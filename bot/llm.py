"""LLM abstraction layer — Claude / Gemini / OpenAI-compatible, with function calling."""

import json
import logging
import os
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

# Gemini models served via Google's OpenAI-compatible endpoint
GEMINI_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _openai_tools_to_claude(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Claude format."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def _pick_provider(model: str) -> str:
    """Decide which provider to use for the given model.

    Returns one of: "gemini", "yunwu", "claude".
    - yunwu: when YUNWU_API_KEY is set and model is NOT a native Gemini model
    - gemini: when model starts with "gemini"
    - claude: fallback to native Anthropic SDK
    """
    is_gemini = model in GEMINI_MODELS or model.startswith("gemini")
    has_yunwu = bool(os.environ.get("YUNWU_API_KEY"))

    if is_gemini:
        return "gemini"
    if has_yunwu:
        return "yunwu"
    return "claude"


async def chat(
    messages: list[dict],
    system: str,
    *,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 150,
    tools: list[dict] | None = None,
    tool_handler=None,  # async (name: str, args: dict) -> str
    max_tool_rounds: int = 2,
    thinking_budget: int = 0,
) -> str:
    """Chat with optional tool-calling loop.

    When *tools* and *tool_handler* are provided, the function will
    automatically execute tool calls and feed results back to the LLM,
    up to *max_tool_rounds* iterations.  Returns the final text response.
    """
    provider = _pick_provider(model)
    use_openai_fmt = provider in ("gemini", "yunwu")

    if provider == "gemini":
        provider_fn = _gemini
    elif provider == "yunwu":
        provider_fn = _yunwu
    else:
        provider_fn = _claude

    # Work on a copy so the caller's list is never mutated
    msgs = list(messages)

    resp = LLMResponse()
    for _round in range(max_tool_rounds + 1):
        resp = await provider_fn(msgs, system, model=model, max_tokens=max_tokens, tools=tools, thinking_budget=thinking_budget)

        if not resp.tool_calls or not tool_handler:
            return resp.text

        # Execute each tool call
        results: dict[str, str] = {}
        for tc in resp.tool_calls:
            results[tc.id] = await tool_handler(tc.name, tc.arguments)

        # Append assistant + tool-result messages in provider-specific format
        if use_openai_fmt:
            # When using text-based tool calls (native_tools=False), avoid
            # role:"tool" messages the relay may not understand — use plain
            # user messages with the tool results instead.
            any_text_tc = any(tc.id.startswith("text_") for tc in resp.tool_calls)
            if any_text_tc:
                if resp.text:
                    msgs.append({"role": "assistant", "content": resp.text})
                result_parts = []
                for tc in resp.tool_calls:
                    result_parts.append(f"[{tc.name}({tc.arguments}) 的查询结果]\n{results[tc.id]}")
                msgs.append({"role": "user", "content": "\n\n".join(result_parts) + "\n\n请根据以上资料，以该角色的身份回复。直接输出角色会说的话。"})
            else:
                msgs.append({
                    "role": "assistant",
                    "content": resp.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in resp.tool_calls
                    ],
                })
                for tc in resp.tool_calls:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": results[tc.id],
                    })
        else:
            # Claude native format
            assistant_content: list[dict] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            msgs.append({"role": "assistant", "content": assistant_content})
            msgs.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": results[tc.id],
                    }
                    for tc in resp.tool_calls
                ],
            })

    # Exhausted rounds — return whatever text we have
    return resp.text


# ── provider implementations ────────────────────────────────────────────────


_TEXT_TOOL_RE = re.compile(
    r'(?:^|\n)\s*(\w+)\(\s*["\'](\w+)["\']\s*\)\s*(?:\n|$)'
)


def _parse_text_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[ToolCall]]:
    """Extract tool calls written as text (e.g. lookup("food")) when the API
    doesn't support structured tool calling.  Returns (cleaned_text, calls)."""
    if not tools or not text:
        return text, []

    # Build set of valid tool names from definitions
    valid_names = {t["function"]["name"] for t in tools if "function" in t}

    calls: list[ToolCall] = []
    for m in _TEXT_TOOL_RE.finditer(text):
        name, arg = m.group(1), m.group(2)
        if name in valid_names:
            calls.append(ToolCall(
                id=f"text_{name}_{len(calls)}",
                name=name,
                arguments={"category": arg},
            ))

    if calls:
        # Remove the matched tool-call text from the response
        cleaned = _TEXT_TOOL_RE.sub("", text).strip()
        return cleaned, calls
    return text, []


def _tools_to_text(tools: list[dict]) -> str:
    """Convert tool definitions to a text block for system prompt injection.

    When a relay (e.g. yunwu) silently drops the `tools` parameter, we inject
    tool descriptions into the system prompt so the model can call them as
    plain text like ``lookup("food")``, which ``_parse_text_tool_calls`` picks up.
    """
    # Build category list from the lookup tool's enum
    categories = ""
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == "lookup":
            cat_info = fn.get("parameters", {}).get("properties", {}).get("category", {})
            if cat_info.get("description"):
                categories = cat_info["description"]

    return (
        "## 重要：回复前必须先查询资料\n"
        "\n"
        "当你准备打字回复（而不是只发🍵）时，必须在第一行写查询调用。"
        "系统会执行它、把结果返回给你，然后你再基于结果生成回复。"
        "这行调用会被系统自动移除，用户看不到。\n"
        "\n"
        "格式（必须独占一行）：\n"
        'lookup("类别名")\n'
        "\n"
        f"可用类别：\n{categories}\n"
        "\n"
        "示例（你的完整输出）：\n"
        'lookup("politics")\n'
        "習近平又要親自部署了❤️"
    )


async def _openai_compat(
    messages: list[dict], system: str, *, model: str, max_tokens: int,
    tools: list[dict] | None = None, api_key: str, base_url: str,
    native_tools: bool = True, thinking_budget: int = 0,
) -> LLMResponse:
    """Shared implementation for all OpenAI-compatible endpoints.

    When *native_tools* is False, tool definitions are injected as text into
    the system prompt instead of being passed via the API ``tools`` parameter.
    The model is expected to write calls like ``lookup("food")`` in its reply,
    which ``_parse_text_tool_calls`` then extracts.
    """
    effective_system = system
    api_tools = tools

    if tools and not native_tools:
        # Inject tool definitions into system prompt; don't send via API
        effective_system = system + "\n\n" + _tools_to_text(tools)
        api_tools = None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    all_msgs = [{"role": "system", "content": effective_system}] + messages
    effective_max = max_tokens
    # Reasoning/thinking models need extra tokens for internal reasoning
    is_reasoning_model = model.endswith("-thinking") or model.endswith("-reasoning")
    if thinking_budget > 0 and is_reasoning_model:
        effective_max = thinking_budget + max_tokens

    kwargs: dict = {
        "model": model,
        "messages": all_msgs,
        "max_completion_tokens": effective_max,
    }
    if api_tools:
        kwargs["tools"] = api_tools
    resp = await client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    tool_calls: list[ToolCall] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))

    text = msg.content or ""

    # Fallback: if API returned no tool_calls, try parsing from text
    if not tool_calls and tools:
        text, tool_calls = _parse_text_tool_calls(text, tools)

    return LLMResponse(text=text, tool_calls=tool_calls)


async def _gemini(
    messages: list[dict], system: str, *, model: str, max_tokens: int,
    tools: list[dict] | None = None, thinking_budget: int = 0,
) -> LLMResponse:
    return await _openai_compat(
        messages, system, model=model, max_tokens=max_tokens, tools=tools,
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        thinking_budget=thinking_budget,
    )


async def _yunwu(
    messages: list[dict], system: str, *, model: str, max_tokens: int,
    tools: list[dict] | None = None, thinking_budget: int = 0,
) -> LLMResponse:
    return await _openai_compat(
        messages, system, model=model, max_tokens=max_tokens, tools=tools,
        api_key=os.environ["YUNWU_API_KEY"],
        base_url="https://yunwu.ai/v1",
        native_tools=False,  # yunwu relay silently drops tools param
        thinking_budget=thinking_budget,
    )


async def _claude(
    messages: list[dict], system: str, *, model: str, max_tokens: int,
    tools: list[dict] | None = None, thinking_budget: int = 0,
) -> LLMResponse:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["max_tokens"] = thinking_budget + max_tokens
    if tools:
        kwargs["tools"] = _openai_tools_to_claude(tools)
    resp = await client.messages.create(**kwargs)

    text = ""
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input,
            ))

    return LLMResponse(text=text, tool_calls=tool_calls)

"""Telegram bot that roleplays as a distilled persona using skill files."""

import asyncio
import base64
import json
import logging
import random
import time
from io import BytesIO
from pathlib import Path

import yaml
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from dotenv import load_dotenv

from bot import llm
from bot.stickers import STICKER_RE, find_sticker, load_stickers

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiogram").setLevel(logging.WARNING)

# Separate file logger for messages only
msg_log = logging.getLogger("bot.messages")
msg_log.setLevel(logging.INFO)
_fh = logging.FileHandler(Path(__file__).resolve().parent.parent / "bot.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
msg_log.addHandler(_fh)

# ── config ──────────────────────────────────────────────────────────────────

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

import os

TARGET_NAME = cfg["target"]["user_name"]
bot_cfg = cfg["bot"]
BOT_TOKEN = os.environ.get("BOT_TOKEN") or bot_cfg.get("token", "")
MODEL = bot_cfg.get("model", "claude-sonnet-4-6-20250514")
TRIGGER = bot_cfg["trigger"]
CONTEXT_WINDOW = bot_cfg.get("context_window", 20)
MAX_TOKENS = bot_cfg.get("max_tokens", 150)
COOLDOWN = bot_cfg.get("cooldown_seconds", 60)
ACTIVE_HOURS = bot_cfg.get("active_hours", [0, 24])
ALLOWED_USERS: list[int] = bot_cfg.get("allowed_users", [])
ALLOWED_GROUPS: list[int] = bot_cfg.get("allowed_groups", [])
SKILL_DIR = ROOT / bot_cfg.get("skill_dir", "data/target/skill")
MEMORY_FILE = SKILL_DIR.parent / "memory.json"
MAX_TOOL_ROUNDS = bot_cfg.get("max_tool_rounds", 2)
THINKING_BUDGET = bot_cfg.get("thinking_budget", 0)
PROACTIVE = bot_cfg.get("proactive", {})

# ── skill loading ───────────────────────────────────────────────────────────


def load_system_prompt() -> str:
    """Load core.md + style.md + examples_core.md as always-loaded system prompt."""
    order = ["core.md", "style.md", "examples_core.md"]
    parts: list[str] = []
    for name in order:
        p = SKILL_DIR / name
        if p.exists():
            parts.append(p.read_text())
    if not parts:
        raise FileNotFoundError(f"No skill files found in {SKILL_DIR}")
    return "\n\n---\n\n".join(parts)


def load_tools() -> list[dict]:
    """Scan ref/ directory to auto-generate lookup tool definition."""
    ref_dir = SKILL_DIR / "ref"
    if not ref_dir.exists():
        return []

    categories: dict[str, str] = {}
    for f in sorted(ref_dir.glob("*.md")):
        first_line = f.read_text().split("\n", 1)[0]
        desc = first_line.lstrip("# ").strip()
        categories[f.stem] = desc

    if not categories:
        return []

    tools = []
    if categories:
        tools.append({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "查询人格详细资料。当话题涉及特定领域时调用获取更详细的背景知识和原话示例，帮助更准确地还原风格。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": list(categories.keys()),
                            "description": "\n".join(f"- {k}: {v}" for k, v in categories.items()),
                        }
                    },
                    "required": ["category"],
                },
            },
        })
    tools.append({
        "type": "function",
        "function": {
            "name": "remember",
            "description": "记住重要信息以供将来参考。当对话中出现值得长期记住的事实时调用，如：某人的重要计划、偏好变化、关键事件、承诺等。不要记录琐碎的日常闲聊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要记住的内容，简洁明了（一句话）",
                    }
                },
                "required": ["content"],
            },
        },
    })
    return tools


def load_memories() -> list[dict]:
    """Load persistent memories from disk."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_memory(content: str) -> str:
    """Save a memory entry to disk."""
    memories = load_memories()
    memories.append({
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "content": content,
    })
    # Keep last 50 memories
    if len(memories) > 50:
        memories = memories[-50:]
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2))
    log.info("Memory saved: %s", content[:60])
    return f"已记住：{content}"


def handle_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool call and return result content."""
    if name == "lookup":
        category = arguments.get("category", "")
        path = SKILL_DIR / "ref" / f"{category}.md"
        if not path.exists():
            return f"Category not found: {category}"
        return path.read_text()
    elif name == "remember":
        return save_memory(arguments.get("content", ""))
    return f"Unknown tool: {name}"


def load_people() -> dict[int, str]:
    """Load per-person profiles from skill/people/ directory.

    Each file has 'user_id: <id>' on the first line, then '---', then content.
    Returns {user_id: profile_content}.
    """
    people_dir = SKILL_DIR / "people"
    if not people_dir.exists():
        return {}
    profiles: dict[int, str] = {}
    for f in sorted(people_dir.glob("*.md")):
        content = f.read_text()
        lines = content.split("\n", 1)
        if lines[0].startswith("user_id:"):
            uid = int(lines[0].split(":", 1)[1].strip())
            # Content after the --- separator
            rest = content.split("---", 1)[-1].strip() if "---" in content else lines[1] if len(lines) > 1 else ""
            profiles[uid] = rest
    return profiles


SYSTEM_PROMPT = load_system_prompt()
TOOLS = load_tools()
PEOPLE = load_people()
load_stickers(SKILL_DIR)


def load_quotes() -> list[str]:
    """Load daily quotes from quotes.md."""
    path = SKILL_DIR / "quotes.md"
    if not path.exists():
        return []
    quotes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("- "):
            quotes.append(line[2:])
    return quotes


QUOTES = load_quotes()
log.info("System prompt loaded: %d chars from %s", len(SYSTEM_PROMPT), SKILL_DIR)
log.info("Tools loaded: %d categories from ref/", len(TOOLS[0]["function"]["parameters"]["properties"]["category"]["enum"]) if TOOLS else 0)
log.info("People loaded: %d profiles from people/", len(PEOPLE))

# ── state ───────────────────────────────────────────────────────────────────

message_buffer: list[dict] = []  # sliding window of recent group messages
last_reply_time: float = 0.0
last_proactive_buf_len: int = 0   # buffer length at last proactive check
bot_user_id: int = 0
bot_username: str = ""

# ── helpers ─────────────────────────────────────────────────────────────────


def is_active_hour() -> bool:
    h = time.localtime().tm_hour
    lo, hi = ACTIVE_HOURS
    return lo <= h < hi


def should_trigger(msg: Message) -> bool:
    """Decide whether to reply to this message."""
    if not is_active_hour():
        return False

    text = msg.text or msg.caption or ""

    # @mention and reply-to-bot always trigger (ignore cooldown)
    is_direct = False
    if TRIGGER.get("on_mention") and msg.entities and bot_username:
        for ent in msg.entities:
            if ent.type == "mention":
                mention = text[ent.offset : ent.offset + ent.length]
                if mention.lower() == f"@{bot_username.lower()}":
                    is_direct = True
            if ent.type == "text_mention" and ent.user and ent.user.id == bot_user_id:
                is_direct = True

    if TRIGGER.get("on_reply") and msg.reply_to_message:
        if msg.reply_to_message.from_user and msg.reply_to_message.from_user.id == bot_user_id:
            is_direct = True

    if is_direct:
        return True

    # Other triggers respect cooldown
    if time.time() - last_reply_time < COOLDOWN:
        return False

    # Photo trigger (e.g. 群主发图)
    photo_cfg = TRIGGER.get("photo_trigger", {})
    if photo_cfg and msg.photo and msg.from_user:
        if msg.from_user.id in photo_cfg.get("user_ids", []):
            if random.random() < photo_cfg.get("probability", 0.5):
                return True

    # Keywords
    for kw in TRIGGER.get("keywords", []):
        if kw.lower() in text.lower():
            return True

    # Random
    prob = TRIGGER.get("random_probability", 0)
    if prob > 0 and random.random() < prob:
        return True

    return False


def format_msg(msg: Message) -> str:
    name = "unknown"
    if msg.from_user:
        name = msg.from_user.first_name or ""
        if msg.from_user.last_name:
            name += " " + msg.from_user.last_name
    if msg.photo:
        caption = msg.caption or ""
        text = f"[发了一张图片] {caption}" if caption else "[发了一张图片]"
    elif msg.sticker:
        emoji = msg.sticker.emoji or "?"
        text = f"[sticker:{emoji}]"
    else:
        text = msg.text or msg.caption or "[media]"
    return f"[{name}]: {text}"


async def download_photo(file_id: str) -> str | None:
    """Download a photo by file_id, return base64 string."""
    try:
        bio = BytesIO()
        await tg.download(file_id, destination=bio)
        bio.seek(0)
        return base64.b64encode(bio.read()).decode()
    except Exception:
        log.warning("Failed to download photo %s", file_id[:20])
        return None


async def send_reply(msg: Message, reply_text: str) -> None:
    """Parse LLM output and send text/sticker messages."""
    lines = [l.strip() for l in reply_text.split("\n") if l.strip()]
    sent = 0
    for line in lines:
        if sent >= 3:
            break
        # Check if line is purely a sticker tag
        m = STICKER_RE.fullmatch(line)
        if m:
            path = find_sticker(m.group(1))
            if path:
                await msg.answer_sticker(FSInputFile(path))
                sent += 1
                await asyncio.sleep(random.uniform(0.5, 1.5))
                continue
        # Check for inline sticker tags mixed with text
        parts = STICKER_RE.split(line)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 1:  # odd index = captured emoji group
                path = find_sticker(part)
                if path:
                    await msg.answer_sticker(FSInputFile(path))
                    sent += 1
            else:
                await msg.answer(part)
                sent += 1
            if sent > 1:
                await asyncio.sleep(random.uniform(0.5, 1.5))
        if sent >= 3:
            break


async def send_to_chat(chat_id: int, reply_text: str) -> None:
    """Send text/sticker messages directly to a chat (for proactive replies)."""
    lines = [l.strip() for l in reply_text.split("\n") if l.strip()]
    sent = 0
    for line in lines:
        if sent >= 3:
            break
        m = STICKER_RE.fullmatch(line)
        if m:
            path = find_sticker(m.group(1))
            if path:
                await tg.send_sticker(chat_id, FSInputFile(path))
                sent += 1
                await asyncio.sleep(random.uniform(0.5, 1.5))
                continue
        parts = STICKER_RE.split(line)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 1:
                path = find_sticker(part)
                if path:
                    await tg.send_sticker(chat_id, FSInputFile(path))
                    sent += 1
            else:
                await tg.send_message(chat_id, part)
                sent += 1
            if sent > 1:
                await asyncio.sleep(random.uniform(0.5, 1.5))
        if sent >= 3:
            break


async def build_llm_messages(trigger_msg: str | None = None) -> list[dict]:
    recent = message_buffer[-CONTEXT_WINDOW:]
    lines = [m["formatted"] for m in recent]
    chat_log = "\n".join(lines)

    if trigger_msg:
        instruction = (
            f"以下是群聊中最近的消息：\n\n{chat_log}\n\n"
            f"触发你回复的是这条消息：「{trigger_msg}」\n"
            f"请以{TARGET_NAME}的身份针对这条消息回复。"
            f"直接输出{TARGET_NAME}会说的话，不要加任何前缀或角色标签。"
        )
    else:
        instruction = (
            f"以下是群聊中最近的消息：\n\n{chat_log}\n\n"
            f"请以{TARGET_NAME}的身份回复最近的话题。"
            f"直接输出{TARGET_NAME}会说的话，不要加任何前缀或角色标签。"
        )

    content: list[dict] = [{"type": "text", "text": instruction}]

    # Attach recent photos (last 3 max to limit tokens)
    photo_msgs = [m for m in recent if m.get("photo_file_id")]
    for m in photo_msgs[-3:]:
        b64 = await download_photo(m["photo_file_id"])
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    return [{"role": "user", "content": content}]


async def generate_reply(messages: list[dict], system: str, active_uids: set[int] | None = None) -> str:
    """Generate reply with optional tool-calling loop."""
    # Inject people profiles for users in the conversation
    if active_uids:
        for uid, profile in PEOPLE.items():
            if uid in active_uids:
                system = system + "\n\n" + profile

    # Inject persistent memories into system prompt
    memories = load_memories()
    if memories:
        mem_lines = [f"- [{m['time']}] {m['content']}" for m in memories[-30:]]
        system = system + "\n\n## 记忆笔记\n以下是你之前记住的重要信息：\n" + "\n".join(mem_lines)

    async def _tool_handler(name: str, args: dict) -> str:
        result = handle_tool_call(name, args)
        log.info("Tool call: %s(%s) → %d chars", name, args, len(result))
        return result

    return await llm.chat(
        messages,
        system,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS or None,
        tool_handler=_tool_handler if TOOLS else None,
        max_tool_rounds=MAX_TOOL_ROUNDS,
        thinking_budget=THINKING_BUDGET,
    )


# ── bot setup ───────────────────────────────────────────────────────────────

tg = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(F.chat.type.in_({"group", "supergroup"}))
async def on_group_message(msg: Message) -> None:
    global last_reply_time, bot_user_id, bot_username

    # Group whitelist — if set, ignore messages from unlisted groups
    if ALLOWED_GROUPS and msg.chat.id not in ALLOWED_GROUPS:
        return

    if not bot_user_id:
        me = await tg.get_me()
        bot_user_id = me.id
        bot_username = me.username or ""

    # Buffer — store file_id for photos and sticker thumbnails
    photo_id = None
    if msg.photo:
        photo_id = msg.photo[-1].file_id
    elif msg.sticker and msg.sticker.thumbnail:
        photo_id = msg.sticker.thumbnail.file_id

    message_buffer.append(
        {
            "msg_id": msg.message_id,
            "from_id": msg.from_user.id if msg.from_user else 0,
            "formatted": format_msg(msg),
            "photo_file_id": photo_id,
            "time": time.time(),
        }
    )
    if len(message_buffer) > 200:
        del message_buffer[: len(message_buffer) - 200]

    if not should_trigger(msg):
        return

    log.info("Triggered by: %s", (msg.text or msg.caption or ("[photo]" if msg.photo else ""))[:60])
    msg_log.info("[群聊·收] %s", format_msg(msg))

    try:
        # Collect user IDs from recent messages for people profile injection
        active_uids = {m["from_id"] for m in message_buffer[-CONTEXT_WINDOW:]}
        trigger_text = format_msg(msg)
        reply_text = await generate_reply(await build_llm_messages(trigger_text), SYSTEM_PROMPT, active_uids)
        reply_text = reply_text.strip()
        if not reply_text:
            return

        await send_reply(msg, reply_text)

        # Add bot's own reply to buffer so future context includes it
        message_buffer.append({
            "msg_id": 0,
            "from_id": bot_user_id,
            "formatted": f"[{TARGET_NAME}]: {reply_text}",
            "photo_file_id": None,
            "time": time.time(),
        })

        last_reply_time = time.time()
        log.info("Replied: %s", reply_text[:80])
        msg_log.info("[群聊·发] %s", reply_text)
    except Exception:
        log.exception("LLM call failed")


# ── private chat ────────────────────────────────────────────────────────────

# per-chat history for private conversations
private_history: dict[int, list[dict]] = {}


@dp.message(F.chat.type == "private", F.text == "/clear")
async def on_clear_history(msg: Message) -> None:
    if ALLOWED_USERS and msg.from_user and msg.from_user.id not in ALLOWED_USERS:
        return
    chat_id = msg.chat.id
    turns = len(private_history.pop(chat_id, []))
    await msg.answer(f"已清空对话记录（{turns} 条）")
    log.info("Private history cleared for %s (%d turns)", chat_id, turns)


@dp.message(F.chat.type == "private")
async def on_private_message(msg: Message) -> None:
    if ALLOWED_USERS and msg.from_user and msg.from_user.id not in ALLOWED_USERS:
        return

    text = msg.text or msg.caption or ""
    has_sticker = bool(msg.sticker)
    if has_sticker:
        emoji = msg.sticker.emoji or "?"
        text = f"[sticker:{emoji}]"

    # Determine file_id for visual content (photo or sticker thumbnail)
    visual_file_id = None
    if msg.photo:
        visual_file_id = msg.photo[-1].file_id
    elif has_sticker and msg.sticker.thumbnail:
        visual_file_id = msg.sticker.thumbnail.file_id

    if not text and not visual_file_id:
        return

    chat_id = msg.chat.id
    history = private_history.setdefault(chat_id, [])

    # Build user message content (text-only or multimodal)
    if visual_file_id:
        b64 = await download_photo(visual_file_id)
        user_content: list[dict] = []
        if text:
            user_content.append({"type": "text", "text": text})
        if b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if not user_content:
            return
        history.append({"role": "user", "content": user_content})
    else:
        history.append({"role": "user", "content": text})

    # Keep last N turns
    if len(history) > CONTEXT_WINDOW * 2:
        del history[: len(history) - CONTEXT_WINDOW * 2]

    log.info("Private msg from %s: %s%s", chat_id, text[:60], " [+photo]" if visual_file_id else "")
    msg_log.info("[私聊·收·%s] %s", chat_id, text)

    try:
        reply_text = await generate_reply(list(history), SYSTEM_PROMPT, {msg.from_user.id} if msg.from_user else None)
        reply_text = reply_text.strip()
        if not reply_text:
            return

        history.append({"role": "assistant", "content": reply_text})

        await send_reply(msg, reply_text)

        log.info("Private reply: %s", reply_text[:80])
        msg_log.info("[私聊·发·%s] %s", chat_id, reply_text)
    except Exception:
        log.exception("Private LLM call failed")


# ── proactive reply ─────────────────────────────────────────────────────────


async def proactive_loop() -> None:
    """Periodically check the message buffer and maybe send a proactive reply."""
    global last_reply_time, last_proactive_buf_len

    interval = PROACTIVE.get("interval_minutes", 15) * 60
    prob = PROACTIVE.get("probability", 0.4)
    min_new = PROACTIVE.get("min_new_messages", 5)

    log.info("Proactive loop started: interval=%dm prob=%.0f%% min_new=%d",
             interval // 60, prob * 100, min_new)

    while True:
        await asyncio.sleep(interval)

        # Check new messages since last check
        new_count = max(0, len(message_buffer) - last_proactive_buf_len)
        last_proactive_buf_len = len(message_buffer)

        if new_count < min_new:
            continue

        if random.random() > prob:
            continue

        # Don't overlap with recent triggered replies
        if time.time() - last_reply_time < COOLDOWN:
            continue

        if not ALLOWED_GROUPS:
            continue

        chat_id = ALLOWED_GROUPS[0]
        log.info("Proactive trigger: %d new messages since last check", new_count)

        try:
            active_uids = {m["from_id"] for m in message_buffer[-CONTEXT_WINDOW:]}
            reply_text = await generate_reply(await build_llm_messages(), SYSTEM_PROMPT, active_uids)
            reply_text = reply_text.strip()
            if not reply_text:
                continue

            await send_to_chat(chat_id, reply_text)

            message_buffer.append({
                "msg_id": 0,
                "from_id": bot_user_id,
                "formatted": f"[{TARGET_NAME}]: {reply_text}",
                "photo_file_id": None,
                "time": time.time(),
            })

            last_reply_time = time.time()
            log.info("Proactive reply: %s", reply_text[:80])
            msg_log.info("[群聊·主动] %s", reply_text)
        except Exception:
            log.exception("Proactive reply failed")


# ── daily quote ─────────────────────────────────────────────────────────────


async def daily_quote_loop() -> None:
    """Send one random quote per day at a random hour."""
    if not QUOTES or not ALLOWED_GROUPS:
        return

    log.info("Daily quote loop started: %d quotes loaded", len(QUOTES))

    while True:
        # Sleep until a random time tomorrow (between 10:00 and 22:00)
        now = time.localtime()
        # Seconds remaining until midnight
        secs_to_midnight = (24 - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec
        # Add random offset: 10:00~22:00 = 36000~79200 seconds into the day
        random_offset = random.randint(10 * 3600, 22 * 3600)
        wait = secs_to_midnight + random_offset
        log.info("Next daily quote in %.1f hours", wait / 3600)
        await asyncio.sleep(wait)

        chat_id = ALLOWED_GROUPS[0]
        quote = random.choice(QUOTES)

        try:
            await send_to_chat(chat_id, quote)
            msg_log.info("[群聊·金句] %s", quote)
            log.info("Daily quote sent: %s", quote[:60])
        except Exception:
            log.exception("Daily quote failed")


# ── entry point ─────────────────────────────────────────────────────────────


async def main() -> None:
    log.info("Starting bot — skill_dir=%s model=%s", SKILL_DIR, MODEL)

    if PROACTIVE.get("enabled"):
        asyncio.create_task(proactive_loop())

    if QUOTES:
        asyncio.create_task(daily_quote_loop())

    await tg.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(tg)


if __name__ == "__main__":
    asyncio.run(main())

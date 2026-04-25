"""Chunking for the RAG index.

Historical: reads preprocessor-produced jsonl (one message per line), splits
the stream into conversation segments (gaps >30 min start a new segment), then
windows each segment into 15-message chunks with 3-message overlap.

Runtime: packages a single bot interaction (context + trigger + reply) as one
chunk for write-back during live conversation.

Noise filtering (stricter than `bot.bot._filter_context` — index quality matters
more than cadence here):
- drop sticker-only lines (they dilute vector signal)
- drop URL-only messages
- drop caption-less media placeholders
- truncate anything over 200 chars
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from bot.rag.store import Chunk

CHUNK_SIZE = 15
CHUNK_OVERLAP = 3
MAX_LINE_LEN = 200
SEGMENT_GAP_SECS = 1800  # 30 min — split conversation segments at this gap

_URL_ONLY_RE = re.compile(r'^https?://\S+$')


def _format_msg(msg: dict) -> str | None:
    """Render one jsonl message as `[name]: content`. Returns None to skip."""
    sender = msg.get("from_name") or "?"
    text = (msg.get("text") or "").strip()
    media_type = msg.get("media_type")

    if media_type == "sticker":
        # Skip — pure noise for vector retrieval (option A in Phase 1 review)
        return None

    if media_type == "photo":
        if text:
            return f"[{sender}]: [发了一张图片] {text}"
        # Skip caption-less photos — pure noise, same as _filter_context
        return None

    if media_type and not text:
        # Other media (video/voice/etc) with no caption — skip
        return None

    if not text:
        return None

    if _URL_ONLY_RE.match(text):
        return None

    if len(text) > MAX_LINE_LEN:
        text = text[:MAX_LINE_LEN] + "...（省略）"

    return f"[{sender}]: {text}"


def _parse_ts(s: str) -> int:
    if not s:
        return 0
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        return 0


def _split_segments(
    formatted: list[tuple[dict, str]],
) -> list[list[tuple[dict, str]]]:
    """Split filtered message stream into conversation segments at >30-min gaps."""
    segments: list[list[tuple[dict, str]]] = []
    current: list[tuple[dict, str]] = []
    prev_ts: int | None = None
    for m, line in formatted:
        cur_ts = _parse_ts(m.get("date", ""))
        if (
            prev_ts is not None
            and cur_ts > 0
            and prev_ts > 0
            and cur_ts - prev_ts > SEGMENT_GAP_SECS
        ):
            if current:
                segments.append(current)
            current = []
        current.append((m, line))
        prev_ts = cur_ts if cur_ts > 0 else prev_ts
    if current:
        segments.append(current)
    return segments


def _window_segment(
    segment: list[tuple[dict, str]],
    chat_id: int | None,
) -> list[Chunk]:
    """Apply 15/3 sliding window to one segment."""
    if len(segment) < 3:
        return []

    step = CHUNK_SIZE - CHUNK_OVERLAP  # 12
    out: list[Chunk] = []
    i = 0
    n = len(segment)
    while i < n:
        window = segment[i : i + CHUNK_SIZE]
        if len(window) < 3:
            break

        lines = [line for _, line in window]
        speakers = sorted({(m.get("from_name") or "?") for m, _ in window})
        timestamps = [_parse_ts(m.get("date", "")) for m, _ in window]
        msg_ids = [int(m.get("id", 0)) for m, _ in window]

        resolved_chat_id = (
            chat_id if chat_id is not None else window[0][0].get("chat_id")
        )

        out.append(Chunk(
            source="historical",
            text="\n".join(lines),
            speakers=speakers,
            timestamp=max(timestamps) if timestamps else 0,
            chat_id=resolved_chat_id,
            msg_id_start=min(msg_ids) if msg_ids else None,
            msg_id_end=max(msg_ids) if msg_ids else None,
        ))

        if i + CHUNK_SIZE >= n:
            break
        i += step
    return out


def chunk_historical(
    jsonl_path: str | Path,
    chat_id: int | None = None,
) -> list[Chunk]:
    """Stream a jsonl file and produce overlapping chunks.

    Pipeline: load → sort by date → noise-filter → split into 30-min-gap
    segments → 15/3 sliding window per segment.
    """
    path = Path(jsonl_path)
    raw: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))

    raw.sort(key=lambda m: m.get("date", ""))

    formatted: list[tuple[dict, str]] = []
    for m in raw:
        line = _format_msg(m)
        if line is None:
            continue
        formatted.append((m, line))

    if len(formatted) < 3:
        return []

    out: list[Chunk] = []
    for seg in _split_segments(formatted):
        out.extend(_window_segment(seg, chat_id))
    return out


def chunk_runtime(
    context_lines: list[str],
    trigger_line: str | None,
    bot_reply: str,
    target_name: str,
    chat_id: int | None = None,
    timestamp: int | None = None,
) -> Chunk:
    """Package one live interaction as a runtime chunk.

    `context_lines` should be pre-formatted (e.g. from `format_msg` in bot.py).
    `trigger_line` is appended if not already the last context line.
    The bot's own reply is appended as `[target_name]: reply`.
    """
    lines: list[str] = list(context_lines)
    if trigger_line and (not lines or lines[-1] != trigger_line):
        lines.append(trigger_line)
    lines.append(f"[{target_name}]: {bot_reply}")

    speakers: set[str] = set()
    for line in lines:
        if line.startswith("[") and "]: " in line:
            speakers.add(line[1 : line.index("]: ")])

    return Chunk(
        source="runtime",
        text="\n".join(lines),
        speakers=sorted(speakers),
        timestamp=timestamp if timestamp is not None else int(time.time()),
        chat_id=chat_id,
    )

"""Sticker mapping — emoji tag → local webp file path, loaded from stats.json."""

import json
import random
import re
from pathlib import Path

# Regex to match [sticker:EMOJI] in LLM output
STICKER_RE = re.compile(r"\[sticker[:：]([^\]]+)\]")

# Populated by load_stickers()
STICKER_MAP: dict[str, list[Path]] = {}


def load_stickers(skill_dir: Path) -> None:
    """Load sticker mapping from stickers/map.json, falling back to stats.json.

    Args:
        skill_dir: Path to the skill directory.
    """
    STICKER_MAP.clear()
    root = Path(__file__).resolve().parent.parent

    # Primary: stickers/map.json (emoji → filename)
    map_file = skill_dir / "stickers" / "map.json"
    if map_file.exists():
        mapping = json.loads(map_file.read_text(encoding="utf-8"))
        sticker_dir = map_file.parent
        for emoji, filename in mapping.items():
            p = sticker_dir / filename
            if p.exists():
                STICKER_MAP.setdefault(emoji, []).append(p)
        return

    # Fallback: stats.json top_stickers
    stats_file = skill_dir.parent / "stats.json"
    if not stats_file.exists():
        return

    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    for entry in stats.get("top_stickers", []):
        emoji = entry.get("emoji")
        path_str = entry.get("path")
        if not emoji or not path_str:
            continue
        p = root / path_str
        if p.exists():
            STICKER_MAP.setdefault(emoji, []).append(p)


def find_sticker(emoji: str) -> Path | None:
    """Return a sticker file path for the given emoji, or None."""
    emoji = emoji.strip()
    paths = STICKER_MAP.get(emoji)
    if not paths:
        return None
    p = random.choice(paths)
    return p if p.exists() else None

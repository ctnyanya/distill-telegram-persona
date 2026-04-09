"""
辅助工具 - 列出 Telegram 账号中的所有聊天及其 ID
用于帮你找到要采集的聊天 ID，填入 config.yaml
"""

import asyncio
import sys
from pathlib import Path

import yaml
from telethon import TelegramClient
from telethon.tl.types import User, Chat, Channel


def load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("错误：找不到 config.yaml，请先复制 config.example.yaml 并填写配置")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
API_ID = config["telegram"]["api_id"]
API_HASH = config["telegram"]["api_hash"]
SESSION_FILE = "telegram_session"


def get_chat_info(dialog):
    entity = dialog.entity
    if isinstance(entity, User):
        name = " ".join(p for p in [entity.first_name or "", entity.last_name or ""] if p)
        return name or entity.username or str(entity.id), "private", entity.id
    if isinstance(entity, (Chat, Channel)):
        return entity.title or str(entity.id), "group", dialog.id
    return str(entity.id), "unknown", dialog.id


async def main():
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    await client.start()

    me = await client.get_me()
    print(f"已登录: {me.first_name} (ID: {me.id})\n")
    print(f"{'类型':<10} {'聊天ID':<20} {'名称'}")
    print("-" * 60)

    async for dialog in client.iter_dialogs():
        name, chat_type, chat_id = get_chat_info(dialog)
        print(f"{chat_type:<10} {chat_id:<20} {name}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

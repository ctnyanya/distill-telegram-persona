"""
数据采集模块 - 使用 Telethon 从 Telegram 拉取聊天记录
支持私聊和群聊，输出 raw_messages.json

群聊策略：找到目标用户的每条消息，并拉取前后 context_radius 条消息作为上下文
媒体支持：sticker（emoji + pack 名）、图片、GIF 均下载保存
断点续传：每个聊天的进度独立保存，中断后重跑自动跳过已完成的部分
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm
from telethon import TelegramClient
from telethon.tl.types import (
    User, Chat, Channel,
    DocumentAttributeSticker,
)

SESSION_FILE = "telegram_session"


def load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("错误：找不到 config.yaml，请先复制 config.example.yaml 并填写配置")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_date(dt: datetime) -> str:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def get_sender_name(sender) -> str:
    if sender is None:
        return "未知"
    if isinstance(sender, User):
        parts = [sender.first_name or "", sender.last_name or ""]
        name = " ".join(p for p in parts if p)
        return name or sender.username or str(sender.id)
    if isinstance(sender, (Chat, Channel)):
        return sender.title or str(sender.id)
    return str(sender.id)


def detect_media_type(message) -> str | None:
    if message.sticker:
        return "sticker"
    if message.gif:
        return "gif"
    if message.photo:
        return "photo"
    if message.video:
        return "video"
    return None


def get_sticker_info(message) -> dict | None:
    if not message.sticker:
        return None
    doc = message.sticker
    emoji = None
    pack_name = None
    for attr in doc.attributes:
        if isinstance(attr, DocumentAttributeSticker):
            emoji = attr.alt
            if attr.stickerset and hasattr(attr.stickerset, 'short_name'):
                pack_name = attr.stickerset.short_name
            break
    return {"emoji": emoji, "pack": pack_name}


async def download_media(client, message, chat_id: int, media_dir: Path) -> str | None:
    media_type = detect_media_type(message)
    if not media_type or media_type == "video":
        return None

    chat_media_dir = media_dir / str(chat_id)
    chat_media_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = await asyncio.wait_for(
            client.download_media(
                message,
                file=str(chat_media_dir / f"{message.id}"),
            ),
            timeout=15,
        )
        if path:
            return str(Path(path))
    except (asyncio.TimeoutError, Exception):
        # 超时后重连，防止连接进入异常状态
        try:
            await client.disconnect()
            await client.connect()
        except Exception:
            pass
    return None


async def build_message_data(client, message, chat_id: int, chat_name: str, chat_type: str, media_dir: Path, enable_download: bool = True) -> dict | None:
    media_type = detect_media_type(message)
    has_text = bool(message.text)

    if not has_text and not media_type:
        return None

    try:
        sender = await asyncio.wait_for(message.get_sender(), timeout=10)
    except (asyncio.TimeoutError, Exception):
        sender = None

    msg_data = {
        "id": message.id,
        "date": format_date(message.date),
        "from_id": message.sender_id,
        "from_name": get_sender_name(sender),
        "chat_id": chat_id,
        "chat_name": chat_name,
        "chat_type": chat_type,
    }

    if message.reply_to_msg_id:
        msg_data["reply_to_msg_id"] = message.reply_to_msg_id

    if has_text:
        msg_data["text"] = message.text

    if media_type:
        msg_data["media_type"] = media_type
        if media_type == "sticker":
            sticker_info = get_sticker_info(message)
            if sticker_info:
                msg_data["sticker"] = sticker_info
        if enable_download:
            media_path = await download_media(client, message, chat_id, media_dir)
            if media_path:
                msg_data["media_path"] = media_path

    if message.raw_text and message.raw_text != message.text:
        msg_data["caption"] = message.raw_text

    return msg_data


# ---- 断点续传 ----

def progress_file(progress_dir: Path, chat_id: int) -> Path:
    return progress_dir / f"{chat_id}.jsonl"


def load_progress(progress_dir: Path, chat_id: int) -> tuple[set[int], list[dict]]:
    pf = progress_file(progress_dir, chat_id)
    if not pf.exists():
        return set(), []
    existing_ids = set()
    messages = []
    with open(pf, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            existing_ids.add(msg["id"])
            messages.append(msg)
    return existing_ids, messages


def append_progress(progress_dir: Path, chat_id: int, msg_data: dict):
    pf = progress_file(progress_dir, chat_id)
    with open(pf, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg_data, ensure_ascii=False) + "\n")


def mark_chat_done(progress_dir: Path, chat_id: int):
    (progress_dir / f"{chat_id}.done").touch()


def is_chat_done(progress_dir: Path, chat_id: int) -> bool:
    return (progress_dir / f"{chat_id}.done").exists()


# ---- 导出逻辑 ----

async def export_private_chat(client: TelegramClient, chat_config: dict, config: dict, dirs: dict) -> list[dict]:
    chat_id = chat_config["chat_id"]

    if is_chat_done(dirs["progress"], chat_id):
        _, messages = load_progress(dirs["progress"], chat_id)
        print(f"\n私聊 {chat_id} 已完成，加载 {len(messages)} 条缓存消息")
        return messages

    print(f"\n正在连接私聊 {chat_id}...")
    try:
        entity = await client.get_entity(chat_id)
    except Exception as e:
        print(f"  无法访问聊天 {chat_id}: {e}")
        return []

    chat_name = get_sender_name(entity)
    print(f"  聊天名称: {chat_name}")

    # 获取聊天总消息数
    total = (await client.get_messages(entity, limit=0)).total
    print(f"  总消息数: {total}")

    exporter_config = config.get("exporter", {})
    max_messages = exporter_config.get("max_messages", 0) or None
    date_from = exporter_config.get("date_from")
    date_to = exporter_config.get("date_to")

    offset_date = None
    if date_to:
        offset_date = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)

    existing_ids, messages = load_progress(dirs["progress"], chat_id)
    if existing_ids:
        print(f"  从断点继续，已有 {len(existing_ids)} 条消息")

    pbar_total = min(max_messages, total) if max_messages else total
    pbar = tqdm(total=pbar_total, desc=f"私聊 {chat_name}", unit="条", initial=len(existing_ids))

    async for message in client.iter_messages(entity, limit=max_messages, offset_date=offset_date):
        if date_from:
            msg_date = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date
            if msg_date < datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc):
                break

        if message.id in existing_ids:
            pbar.update(1)
            continue

        msg_data = await build_message_data(client, message, chat_id, chat_name, "private", dirs["media"], dirs["download"])
        if msg_data:
            messages.append(msg_data)
            append_progress(dirs["progress"], chat_id, msg_data)
        pbar.update(1)

    pbar.close()
    mark_chat_done(dirs["progress"], chat_id)
    print(f"  采集到 {len(messages)} 条有效消息")
    return messages


async def export_group_chat(client: TelegramClient, chat_config: dict, target_user_id: int, config: dict, dirs: dict) -> list[dict]:
    chat_id = chat_config["chat_id"]

    if is_chat_done(dirs["progress"], chat_id):
        _, messages = load_progress(dirs["progress"], chat_id)
        print(f"\n群聊 {chat_id} 已完成，加载 {len(messages)} 条缓存消息")
        return messages

    print(f"\n正在连接群聊 {chat_id}...")
    try:
        entity = await client.get_entity(chat_id)
    except Exception as e:
        print(f"  无法访问聊天 {chat_id}: {e}")
        return []

    chat_name = get_sender_name(entity)
    print(f"  聊天名称: {chat_name}")

    # 获取聊天总消息数
    total = (await client.get_messages(entity, limit=0)).total
    print(f"  总消息数: {total}")

    exporter_config = config.get("exporter", {})
    max_messages = exporter_config.get("max_messages", 0) or None
    date_from = exporter_config.get("date_from")
    date_to = exporter_config.get("date_to")
    context_radius = exporter_config.get("context_radius", 15)

    offset_date = None
    if date_to:
        offset_date = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)

    existing_ids, messages = load_progress(dirs["progress"], chat_id)
    if existing_ids:
        print(f"  从断点继续，已有 {len(existing_ids)} 条消息")

    # 第一遍：扫描目标用户的所有消息 ID 及 reply 关系（结果缓存到文件）
    scan_cache = dirs["progress"] / f"{chat_id}.scan.json"
    target_msgs = None
    if scan_cache.exists():
        raw = json.loads(scan_cache.read_text(encoding="utf-8"))
        if raw and isinstance(raw[0], int):
            # 旧格式（纯 ID 列表），需要重新扫描以获取 reply 信息
            print(f"  检测到旧格式扫描缓存，重新扫描以获取 reply 信息...")
            scan_cache.unlink()
        else:
            target_msgs = raw
            print(f"  从缓存加载扫描结果，{len(target_msgs)} 条目标用户消息")

    if target_msgs is None:
        target_msgs = []
        pbar1_total = min(max_messages, total) if max_messages else total
        pbar1 = tqdm(total=pbar1_total, desc=f"群聊 {chat_name} - 扫描目标消息", unit="条")

        async for message in client.iter_messages(entity, limit=max_messages, offset_date=offset_date):
            if date_from:
                msg_date = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date
                if msg_date < datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc):
                    break
            if message.sender_id == target_user_id:
                target_msgs.append({
                    "id": message.id,
                    "reply_to": message.reply_to_msg_id,
                })
            pbar1.update(1)

        pbar1.close()
        scan_cache.write_text(json.dumps(target_msgs, ensure_ascii=False), encoding="utf-8")
        print(f"  找到 {len(target_msgs)} 条目标用户消息")

    if not target_msgs:
        mark_chat_done(dirs["progress"], chat_id)
        return messages

    # 计算需要拉取的消息 ID（去掉已有的）
    target_msg_ids = [m["id"] for m in target_msgs]
    reply_ids = {m["reply_to"] for m in target_msgs if m.get("reply_to")}

    ids_to_fetch = set()
    for msg_id in target_msg_ids:
        for offset in range(-context_radius, context_radius + 1):
            ids_to_fetch.add(msg_id + offset)

    # 把 reply 引用的消息也加入采集（确保回复链完整）
    reply_extra = reply_ids - ids_to_fetch
    ids_to_fetch.update(reply_ids)
    if reply_extra:
        print(f"  reply 引用额外增加 {len(reply_extra)} 条消息（超出 radius 范围）")
    ids_to_fetch -= existing_ids

    if not ids_to_fetch:
        print(f"  所有上下文消息均已采集")
        mark_chat_done(dirs["progress"], chat_id)
        return messages

    ids_list = sorted(ids_to_fetch)
    batch_size = 100
    pbar2 = tqdm(total=len(ids_list), desc=f"群聊 {chat_name} - 拉取上下文", unit="条")

    for i in range(0, len(ids_list), batch_size):
        batch_ids = ids_list[i:i + batch_size]
        try:
            batch_msgs = await asyncio.wait_for(
                client.get_messages(entity, ids=batch_ids),
                timeout=30,
            )
        except (asyncio.TimeoutError, Exception) as e:
            print(f"\n    批次 {i//batch_size} 超时，等待 5 秒后重试...")
            await asyncio.sleep(5)
            try:
                batch_msgs = await asyncio.wait_for(
                    client.get_messages(entity, ids=batch_ids),
                    timeout=30,
                )
            except (asyncio.TimeoutError, Exception):
                print(f"    批次 {i//batch_size} 重试失败，跳过")
                pbar2.update(len(batch_ids))
                continue
        await asyncio.sleep(1)  # 每批次间隔 1 秒，避免触发限流
        for message in batch_msgs:
            if message is None:
                pbar2.update(1)
                continue
            msg_data = await build_message_data(client, message, chat_id, chat_name, "group", dirs["media"], dirs["download"])
            if msg_data:
                msg_data["is_target"] = (message.sender_id == target_user_id)
                messages.append(msg_data)
                append_progress(dirs["progress"], chat_id, msg_data)
            pbar2.update(1)

    pbar2.close()
    mark_chat_done(dirs["progress"], chat_id)
    print(f"  采集到 {len(messages)} 条有效消息（含上下文）")
    return messages


async def main():
    config = load_config()

    api_id = config["telegram"]["api_id"]
    api_hash = config["telegram"]["api_hash"]
    target_name = config["target"]["user_name"]
    chats = config["target"]["chats"]

    output_dir = Path(config["exporter"].get("output_dir", f"data/{target_name}"))
    media_dir = output_dir / "media"
    progress_dir = output_dir / "progress"

    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(exist_ok=True)
    progress_dir.mkdir(exist_ok=True)

    enable_download = config.get("exporter", {}).get("download_media", True)
    dirs = {"output": output_dir, "media": media_dir, "progress": progress_dir, "download": enable_download}

    print(f"蒸馏目标: {target_name}")
    print(f"输出目录: {output_dir}")
    print(f"待采集聊天数: {len(chats)}")

    client = TelegramClient(SESSION_FILE, api_id, api_hash)
    await client.start()

    me = await client.get_me()
    print(f"已登录: {me.first_name} ({me.phone})")

    target_user_id = config["target"].get("user_id")
    if not target_user_id:
        for chat in chats:
            if chat.get("type") == "private":
                target_user_id = chat["chat_id"]
                break

    all_messages = []
    for chat_config in chats:
        chat_type = chat_config.get("type", "group")
        if chat_type == "private":
            msgs = await export_private_chat(client, chat_config, config, dirs)
        else:
            if not target_user_id:
                print(f"  跳过群聊 {chat_config['chat_id']}：未配置 target_user_id")
                continue
            msgs = await export_group_chat(client, chat_config, target_user_id, config, dirs)
        all_messages.extend(msgs)

    # 按时间排序
    all_messages.sort(key=lambda m: m["date"])

    output = {
        "target": {
            "name": target_name,
            "user_id": target_user_id,
        },
        "export_time": format_date(datetime.now(timezone.utc)),
        "total_messages": len(all_messages),
        "chats_exported": len(chats),
        "messages": all_messages,
    }

    output_path = output_dir / "raw_messages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n导出完成！")
    print(f"  总消息数: {len(all_messages)}")
    print(f"  输出文件: {output_path}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

"""
数据预处理模块 - 将 exporter 采集的 JSONL 数据转为蒸馏输入

输出：
- stats.json: 全量统计（词频、emoji、sticker、活跃时段、TF-IDF top 消息）
- chunks/: 压缩格式的聊天记录分块（每块 ~50 万 token）
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("错误：找不到 config.yaml")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- 数据加载 ----

def load_messages(jsonl_path: Path) -> list[dict]:
    messages = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


# ---- 格式压缩 ----

def format_message(msg: dict) -> str:
    """将一条消息转为紧凑格式"""
    date = msg.get("date", "")
    short_date = date[:16].replace("T", " ") if date else "????-??-?? ??:??"
    sender = msg.get("from_name", "?")

    parts = []
    if "text" in msg:
        parts.append(msg["text"])

    media_type = msg.get("media_type")
    if media_type == "sticker":
        emoji = msg.get("sticker", {}).get("emoji", "")
        media_path = msg.get("media_path", "")
        parts.append(f"[sticker {emoji} {media_path}]".strip())
    elif media_type:
        media_path = msg.get("media_path", "")
        if media_path:
            parts.append(f"[{media_type} {media_path}]")
        else:
            parts.append(f"[{media_type}]")

    if msg.get("caption") and msg.get("caption") != msg.get("text"):
        parts.append(msg["caption"])

    if not parts:
        return ""

    return f"[{short_date}] {sender}: {' '.join(parts)}"


# ---- 统计分析 ----

def extract_words(text: str) -> list[str]:
    """简易分词：按非中文字符和空格切分"""
    # 匹配中文字符序列或英文单词
    return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())


def compute_stats(target_messages: list[dict], all_messages: list[dict], target_name: str) -> dict:
    """计算目标用户的全量统计"""

    # 基础计数
    total_count = len(all_messages)
    target_count = len(target_messages)

    # 词频
    word_counter = Counter()
    for msg in target_messages:
        text = msg.get("text", "")
        if text:
            word_counter.update(extract_words(text))

    # emoji 频率（提取 unicode emoji）
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        "\U0000FE00-\U0000FE0F\U0000200D]+",
        flags=re.UNICODE,
    )
    emoji_counter = Counter()
    for msg in target_messages:
        text = msg.get("text", "")
        if text:
            emoji_counter.update(emoji_pattern.findall(text))

    # sticker 频率 + 高频 sticker 路径
    sticker_emoji_counter = Counter()
    sticker_pack_counter = Counter()
    sticker_path_counter = Counter()  # media_path → 使用次数
    sticker_path_info = {}  # media_path → {emoji, pack}
    for msg in target_messages:
        if msg.get("media_type") == "sticker":
            sticker = msg.get("sticker", {})
            if sticker.get("emoji"):
                sticker_emoji_counter[sticker["emoji"]] += 1
            if sticker.get("pack"):
                sticker_pack_counter[sticker["pack"]] += 1
            media_path = msg.get("media_path")
            if media_path:
                sticker_path_counter[media_path] += 1
                if media_path not in sticker_path_info:
                    sticker_path_info[media_path] = {
                        "emoji": sticker.get("emoji", ""),
                        "pack": sticker.get("pack", ""),
                    }

    # 活跃时段
    hour_counter = Counter()
    for msg in target_messages:
        date = msg.get("date", "")
        if date and len(date) >= 13:
            try:
                hour = int(date[11:13])
                hour_counter[hour] += 1
            except ValueError:
                pass

    # 消息长度分布
    lengths = []
    for msg in target_messages:
        text = msg.get("text", "")
        if text:
            lengths.append(len(text))

    length_dist = {}
    if lengths:
        length_dist = {
            "avg": round(sum(lengths) / len(lengths), 1),
            "median": sorted(lengths)[len(lengths) // 2],
            "max": max(lengths),
            "under_10": sum(1 for l in lengths if l < 10),
            "10_to_30": sum(1 for l in lengths if 10 <= l < 30),
            "30_to_100": sum(1 for l in lengths if 30 <= l < 100),
            "over_100": sum(1 for l in lengths if l >= 100),
        }

    # 媒体类型分布
    media_counter = Counter()
    for msg in target_messages:
        mt = msg.get("media_type")
        if mt:
            media_counter[mt] += 1

    return {
        "target_name": target_name,
        "total_messages": total_count,
        "target_messages": target_count,
        "target_ratio": round(target_count / total_count, 3) if total_count else 0,
        "top_words": word_counter.most_common(100),
        "top_emoji": emoji_counter.most_common(30),
        "top_sticker_emoji": sticker_emoji_counter.most_common(30),
        "top_sticker_packs": sticker_pack_counter.most_common(20),
        "top_stickers": [
            {"path": path, "count": count, **sticker_path_info.get(path, {})}
            for path, count in sticker_path_counter.most_common(30)
        ],
        "active_hours": dict(sorted(hour_counter.items())),
        "message_length": length_dist,
        "media_types": dict(media_counter.most_common()),
    }


def compute_tfidf_scores(target_messages: list[dict], top_n: int = 200) -> list[dict]:
    """TF-IDF 筛选最具信息量的消息（用于 examples.md 候选）"""
    # 构建文档集（每条消息是一个文档）
    docs = []
    doc_indices = []
    for i, msg in enumerate(target_messages):
        text = msg.get("text", "")
        if text and len(text) >= 5:  # 过滤太短的
            words = extract_words(text)
            if words:
                docs.append(words)
                doc_indices.append(i)

    if not docs:
        return []

    # 计算 IDF
    n_docs = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    idf = {}
    for word, count in df.items():
        idf[word] = math.log(n_docs / count)

    # 计算每条消息的 TF-IDF 总分
    scored = []
    for idx, doc in zip(doc_indices, docs):
        tf = Counter(doc)
        doc_len = len(doc)
        score = sum((count / doc_len) * idf.get(word, 0) for word, count in tf.items())
        scored.append((idx, score))

    # 取 top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    top_messages = []
    for idx, score in scored[:top_n]:
        msg = target_messages[idx]
        top_messages.append({
            "text": msg.get("text", ""),
            "date": msg.get("date", ""),
            "score": round(score, 4),
            "chat_type": msg.get("chat_type", ""),
        })

    return top_messages


# ---- 分块 ----

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文 ~1.5 token/字，英文 ~1.3 token/word）"""
    return int(len(text) * 1.4)


def chunk_messages(formatted_lines: list[str], max_tokens: int = 100_000) -> list[list[str]]:
    """将格式化后的消息按 token 上限分块，在对话间隙处切分"""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in formatted_lines:
        line_tokens = estimate_tokens(line)

        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---- 主流程 ----

def main():
    config = load_config()
    target_name = config["target"]["user_name"]
    chats = config["target"]["chats"]
    output_dir = Path(config["exporter"].get("output_dir", f"data/{target_name}"))
    progress_dir = output_dir / "progress"
    chunks_dir = output_dir / "chunks"

    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"预处理目标: {target_name}")
    print(f"数据目录: {output_dir}")

    # 获取 target_user_id
    target_user_id = None
    for chat in chats:
        if chat.get("type") == "private":
            target_user_id = chat["chat_id"]
            break

    # 加载所有消息
    all_target_messages = []
    all_messages_by_chat = {}

    for chat_config in chats:
        chat_id = chat_config["chat_id"]
        chat_type = chat_config.get("type", "group")
        jsonl_path = progress_dir / f"{chat_id}.jsonl"

        if not jsonl_path.exists():
            print(f"  跳过 {chat_id}: 无数据文件")
            continue

        messages = load_messages(jsonl_path)
        messages.sort(key=lambda m: m.get("date", ""))
        all_messages_by_chat[(chat_id, chat_type)] = messages

        # 提取目标用户消息
        for msg in messages:
            if chat_type == "private":
                if msg.get("from_id") == target_user_id:
                    msg["chat_type"] = "private"
                    all_target_messages.append(msg)
            else:
                if msg.get("is_target", False):
                    msg["chat_type"] = "group"
                    all_target_messages.append(msg)

        print(f"  已加载 {chat_type} {chat_id}: {len(messages)} 条消息")

    print(f"\n目标用户消息总数: {len(all_target_messages)}")

    # 全量统计
    print("计算统计数据...")
    all_flat = []
    for msgs in all_messages_by_chat.values():
        all_flat.extend(msgs)

    stats = compute_stats(all_target_messages, all_flat, target_name)

    # TF-IDF top 消息
    print("计算 TF-IDF 分数...")
    tfidf_top = compute_tfidf_scores(all_target_messages, top_n=200)
    stats["tfidf_top_messages"] = tfidf_top

    # 保存 stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计数据已保存: {stats_path}")

    # 格式压缩 + 分块（按聊天类型分开）
    print("\n格式压缩 + 分块...")
    chunk_index = 0

    for (chat_id, chat_type), messages in all_messages_by_chat.items():
        chat_name = messages[0].get("chat_name", str(chat_id)) if messages else str(chat_id)
        prefix = "private" if chat_type == "private" else "group"

        # 格式化所有消息
        formatted = []
        for msg in messages:
            line = format_message(msg)
            if line:
                formatted.append(line)

        print(f"  {prefix} ({chat_name}): {len(formatted)} 条有效消息")

        # 分块
        chunks = chunk_messages(formatted)

        for i, chunk_lines in enumerate(chunks):
            chunk_index += 1
            # 文件头
            first_date = chunk_lines[0].split("]")[0].lstrip("[") if chunk_lines else "?"
            last_date = chunk_lines[-1].split("]")[0].lstrip("[") if chunk_lines else "?"

            header = f"=== {prefix}: {chat_name} ({first_date} ~ {last_date}) ===\n\n"
            content = header + "\n".join(chunk_lines) + "\n"

            chunk_path = chunks_dir / f"{prefix}_{i+1:02d}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(content)

            est_tokens = estimate_tokens(content)
            print(f"    {chunk_path.name}: {len(chunk_lines)} 条, ~{est_tokens:,} token")

    print(f"\n预处理完成！")
    print(f"  统计文件: {stats_path}")
    print(f"  分块目录: {chunks_dir}")
    print(f"  总分块数: {chunk_index}")
    print(f"\n下一步：在 Claude Code 中执行 /distill 开始蒸馏")


if __name__ == "__main__":
    main()

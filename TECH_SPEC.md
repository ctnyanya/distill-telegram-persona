# Telegram Skill Bot - Tech Spec

## 概述

将 Telegram 群聊中特定用户的聊天记录蒸馏为 AI 人格 Skill，再通过 Telegram Bot 以该用户的风格参与群聊。

## 系统架构

```
Telegram 聊天记录 (JSON)
        │
        ▼
  ┌─────────────┐
  │  数据预处理   │  清洗、分段、提取对话对
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  人格蒸馏    │  Claude API 分析风格 → 生成 system prompt
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Skill 文件  │  .md 格式的人格描述 + few-shot 示例
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     ┌───────────┐
  │ Telegram Bot │◄───►│ Claude API │
  └─────────────┘     └───────────┘
```

## 模块设计

### 1. 数据采集模块 (`exporter.py`)

**输入**：Telegram 群聊 ID + 目标用户 ID
**输出**：`raw_messages.json`

- 使用 Telethon 连接 Telegram API
- 拉取目标用户在指定群聊中的全部消息
- 同时拉取被回复的上文，保留对话上下文
- 过滤掉纯图片/贴纸/系统消息等无文本内容

```json
{
  "user": { "id": 123456, "name": "张三" },
  "messages": [
    {
      "id": 1001,
      "text": "这个方案不太行吧，之前踩过坑的",
      "date": "2026-03-15T14:22:00",
      "reply_to": { "id": 999, "text": "我觉得用 Redis 做就行了", "from": "李四" }
    }
  ]
}
```

### 2. 数据预处理模块 (`preprocessor.py`)

**输入**：progress 目录下的 JSONL 文件
**输出**：`stats.json` + `chunks/chunk_01.txt` ~ `chunk_N.txt`

处理逻辑：

**格式压缩**（无损）：将 JSONL 转为紧凑的聊天记录格式，去掉蒸馏无关的字段（chat_id、from_id 等），保留所有文字内容和媒体引用：
```
[03-15 14:22] 张三: 你好
[03-15 14:22] 张三: [sticker 😂 media/123456789/12345.webp]
[03-15 14:23] 李四: 嗯嗯 [photo media/123456789/12346.jpg]
```

**全量统计分析** → `stats.json`：
- 词频 / 口头禅频率
- emoji 使用频率
- sticker 使用频率（按 pack 和 emoji 分组）
- 活跃时间段分布
- 消息长度分布
- 私聊 vs 群聊发言比例
- TF-IDF 筛选 top 消息（最具信息量 / 最能代表风格的发言，用于 examples.md 候选）

**分块**：按时间顺序切分为多个 chunk 文件，每块 ~50 万 token，确保对话片段不被截断（在对话间隙处切分）。私聊和群聊分开切分，chunk 文件名标注来源。

### 3. 人格蒸馏（Claude Code Skill：`/distill`）

**输入**：`stats.json` + `chunks/` 目录
**输出**：`skill/` 目录（多文件，按维度拆分）

通过 Claude Code skill 完成，不使用 API。用户在 Claude Code 中执行 `/distill` 即可启动。

**蒸馏流程**：

**Phase 1 — 逐块分析**：依次读取每个 chunk，结合 stats.json 进行分析，提取人格特征并保存到 `analysis_N.json`。每完成一块，原始数据随 context 压缩释放，但分析结果已持久化到文件。

分析维度（基于 TwinVoice 6 维度框架 + 扩展）：

| 维度 | 说明 | 示例 |
|------|------|------|
| **Lexical Fidelity** 用词习惯 | 口头禅、惯用词汇、造词 | "不太行吧"、"有一说一"、"绷不住了" |
| **Syntactic Style** 句式结构 | 句式特征、标点、长度 | 喜欢反问、常用省略号、短句居多 |
| **Persona Tone** 语气情绪 | emoji/sticker 使用、情绪表达模式 | 高频使用 😅🤔，冷幽默，从不用 ❤️ |
| **Opinion Consistency** 观点一致性 | 话题偏好、立场、价值观 | 技术 > 吐槽 > 八卦，回避政治话题 |
| **Memory Recall** 知识领域 | 展现的知识、经历、兴趣 | 后端开发、数据库优化、日本动漫 |
| **Logical Reasoning** 互动逻辑 | 如何回应不同类型的对话 | 爱抬杠但不恶意、经常帮人纠错 |
| **回复习惯** | 回复频率、长度、分条习惯 | 经常分多条发、群里被 cue 才回 |

**Phase 2 — 合成**：读取所有 `analysis_*.json` + `stats.json`，合成 skill 文件集。

断点续传：已生成的 `analysis_N.json` 不会重复处理，中断后重新执行 `/distill` 从未完成的 chunk 继续。

**Sticker 分析**（可选）：蒸馏完成后可让 Claude 查看高频 sticker 图片，补充表情使用习惯分析。

`skill/` 目录结构：

```
skill/
├── persona.md      # 核心性格、语气、互动风格
├── knowledge.md    # 知识领域、话题偏好、观点倾向
├── style.md        # 语言习惯：口头禅、句式、emoji/sticker 偏好
└── examples.md     # 精选对话示例（few-shot，从原始数据中挑选）
```

Bot 运行时将所有 skill 文件拼接为 system prompt。

### 4. Bot 服务模块 (`bot.py`)

**输入**：`skill/` 目录 + Telegram Bot Token
**运行时**：长驻进程，监听群消息

核心设计：

```
群消息流 ──► 触发判断 ──► 构建上下文 ──► LLM API ──► 发送回复
```

**多模型支持**：

Bot 通过统一的 LLM 抽象层对接不同模型，config 中切换 provider 即可：

| Provider | 模型 | 方式 |
|----------|------|------|
| Claude | claude-sonnet-4-6, claude-haiku-4-5 | API 计费（Anthropic SDK） |
| Gemini | gemini-2.5-pro, gemini-2.5-flash | 免费额度（Google AI Studio，OpenAI 兼容接口） |

实现方式：封装一个 `llm.py` 模块，提供统一的 `chat()` 接口，内部根据 provider 调用对应 SDK。Claude 使用 Anthropic SDK 原生调用，Gemini 使用 OpenAI 兼容接口调用。

**触发机制**（可配置）：
- `@bot` 被直接提及 → 必回
- 被 reply → 必回
- 关键词命中（可配置词表）→ 回
- 随机触发（可配置概率，如 5%）→ 偶尔插嘴

**上下文构建**：
- 维护最近 50 条群消息的滑动窗口（内存中）
- 调用 Claude 时传入最近 10-20 条作为对话历史
- system prompt = skill/ 目录下所有文件拼接

**回复控制**：
- max_tokens 限制在 150 以内，模拟真人碎片化发言
- 对连续触发做频率限制（如 60 秒冷却），避免刷屏
- 可配置"在线时段"，模拟作息

## 配置文件 (`config.yaml`)

```yaml
telegram:
  bot_token: "BOT_TOKEN"
  api_id: 12345
  api_hash: "your_api_hash"

target:
  group_id: -1001234567890
  user_id: 123456
  user_name: "张三"

distill:
  model: "claude-sonnet-4-6-20250514"
  max_sample_conversations: 200

bot:
  llm:
    provider: "gemini"  # "claude" 或 "gemini"
    model: "gemini-2.5-flash"
    api_key: "your_api_key"
    # Gemini 专用（OpenAI 兼容接口）
    # base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  trigger:
    on_mention: true
    on_reply: true
    random_probability: 0.05
    keywords: ["张三", "老张"]
  context_window: 20
  max_tokens: 150
  cooldown_seconds: 60
  active_hours: [9, 23]  # 早 9 到晚 11
```

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 生态最好 |
| Telegram 数据采集 | Telethon | 支持 user API，能拉历史消息 |
| Telegram Bot | aiogram 3.x | 异步原生，群聊支持好 |
| LLM | Claude API / Gemini API | 多模型支持，按需切换 |
| LLM 抽象 | openai SDK（兼容层） + anthropic SDK | Gemini 走 OpenAI 兼容接口，Claude 走原生 SDK |
| 配置 | PyYAML | 简单够用 |
| 部署 | Docker + docker-compose | 方便在 VPS 上跑 |

## 目录结构

```
telegram-skill-bot/
├── config.yaml          # 配置文件（git ignore）
├── config.example.yaml  # 配置示例
├── exporter.py          # 数据采集
├── preprocessor.py      # 数据预处理
├── distiller.py         # 人格蒸馏
├── llm.py               # LLM 统一调用层（Claude / Gemini）
├── bot.py               # Telegram Bot 主进程
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── data/                # 运行时数据（git ignore）
    └── {target_name}/   # 每个蒸馏目标独立目录
        ├── raw_messages.json
        ├── stats.json           # 全量统计
        ├── chunks/              # 压缩格式的聊天记录分块
        │   ├── private_01.txt
        │   ├── group_01.txt
        │   └── ...
        ├── analysis/            # 逐块分析结果
        │   ├── analysis_01.json
        │   └── ...
        ├── media/               # 下载的媒体文件
        ├── progress/            # exporter 断点续传文件
        └── skill/               # 蒸馏输出
            ├── persona.md
            ├── knowledge.md
            ├── style.md
            └── examples.md
```

## 使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 填写配置
cp config.example.yaml config.yaml

# 3. 导出聊天记录（首次需要 Telegram 登录验证）
python exporter.py

# 4. 预处理（本地统计 + 格式压缩 + 分块）
python preprocessor.py

# 5. 蒸馏人格（在 Claude Code 中执行）
/distill

# 6. 查看/手动微调生成的 skill
ls data/{target_name}/skill/

# 7. 启动 Bot
python bot.py
```

## 注意事项

- **隐私与授权**：蒸馏他人的聊天记录前必须获得本人同意
- **Telegram API 限制**：首次使用 Telethon 需要用自己的 Telegram 账号登录，注意不要触发风控
- **费用控制**：蒸馏阶段使用 Claude Code（Max plan），不消耗 API 额度；Bot 运行时可选 Gemini 免费方案
- **效果调优**：生成的 skill 文件建议人工审阅微调，特别是 examples.md 中的对话示例

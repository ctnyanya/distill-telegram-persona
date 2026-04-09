# 蒸馏你的 Telegram 网友

从 Telegram 聊天记录中蒸馏一个人的完整人格画像，生成 AI 角色扮演 Skill 文件，再通过 Telegram Bot 以 TA 的风格参与群聊。

## 系统架构

```
Telegram 聊天记录
        │
        ▼
  ┌─────────────┐
  │  数据采集    │  exporter.py — Telethon 拉取聊天记录
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  数据预处理   │  preprocessor.py — 清洗、统计、分块
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  人格蒸馏    │  Claude Code /distill — 分析风格 → 生成 Skill
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     ┌───────────────────┐
  │ Telegram Bot │◄───►│ LLM (Claude/Gemini)│
  └─────────────┘     └───────────────────┘
```

## 功能特性

- **多聊天源采集**：支持同时从私聊和群聊拉取目标用户的消息，群聊自动拉取上下文
- **断点续传**：数据采集和蒸馏均支持中断后继续，不重复处理
- **TwinVoice 6 维度分析**：用词习惯、句式结构、语气情绪、观点立场、知识领域、互动逻辑
- **分层 Skill 系统**：核心人格 always-loaded + 详细资料 on-demand（通过 function calling 按需查询）
- **多模型支持**：Claude（Anthropic SDK）/ Gemini（OpenAI 兼容接口）/ 云雾中转
- **灵活触发**：@提及、回复、关键词、随机概率、图片触发、主动发言
- **Sticker 支持**：分析并还原目标用户的 sticker 使用习惯
- **Extended Thinking**：支持 Claude thinking 模式，提升角色扮演质量

## 前置条件

- Python 3.11+
- Telegram API 凭证（[my.telegram.org](https://my.telegram.org) 获取 `api_id` 和 `api_hash`）
- Telegram Bot Token（[@BotFather](https://t.me/BotFather) 创建）
- LLM API Key（至少一个）：Anthropic / Google AI Studio / 云雾中转
- [Claude Code](https://claude.ai/claude-code)（用于人格蒸馏步骤）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

编辑 `config.yaml`，填入：
- Telegram API 凭证
- 蒸馏目标的用户名和聊天 ID（可通过 `python list_chats.py` 查看所有聊天）
- Bot 触发规则和模型选择

编辑 `.env`，填入 LLM API Key 和 Bot Token。

### 3. 导出聊天记录

```bash
python exporter.py
```

首次运行需要用你的 Telegram 账号登录（输入手机号和验证码）。

### 4. 预处理

```bash
python preprocessor.py
```

生成统计数据 `stats.json` 和分块聊天记录 `chunks/`。

### 5. 蒸馏人格

在 Claude Code 中执行：

```
/distill
```

这会逐块分析聊天记录，生成 `skill/` 目录下的人格文件。

### 6. 启动 Bot

```bash
python -m bot.bot
```

## 配置说明

### config.yaml

| 字段 | 说明 |
|------|------|
| `telegram.api_id/api_hash` | Telegram API 凭证 |
| `target.user_name` | 蒸馏目标显示名 |
| `target.user_id` | 目标用户 ID（自蒸馏时需显式指定） |
| `target.chats` | 要采集的聊天列表 |
| `exporter.output_dir` | 输出目录 |
| `exporter.download_media` | 是否下载媒体文件 |
| `bot.skill_dir` | Skill 文件目录 |
| `bot.model` | LLM 模型名 |
| `bot.allowed_users` | 私聊白名单 |
| `bot.allowed_groups` | 群聊白名单 |
| `bot.trigger` | 触发规则（关键词、概率、图片触发等） |
| `bot.proactive` | 主动发言配置 |
| `bot.max_tool_rounds` | lookup 工具最大调用轮数 |
| `bot.thinking_budget` | Extended thinking token 预算 |

### .env

| 变量 | 说明 |
|------|------|
| `BOT_TOKEN` | Telegram Bot Token |
| `ANTHROPIC_API_KEY` | Claude API Key |
| `GEMINI_API_KEY` | Google AI Studio API Key |
| `YUNWU_API_KEY` | 云雾中转 API Key |

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.11+ |
| 数据采集 | Telethon（Telegram User API） |
| Bot 框架 | aiogram 3.x |
| LLM | Claude API / Gemini API / OpenAI 兼容接口 |
| 人格蒸馏 | Claude Code（/distill skill） |
| 配置 | PyYAML + python-dotenv |

## 项目结构

```
telegram-skill-bot/
├── exporter.py          # 数据采集（Telethon）
├── preprocessor.py      # 数据预处理（统计 + 分块）
├── list_chats.py        # 辅助工具：列出所有聊天
├── bot/
│   ├── bot.py           # Telegram Bot 主进程
│   ├── llm.py           # LLM 统一抽象层
│   └── stickers.py      # Sticker 映射
├── .claude/
│   └── commands/
│       └── distill.md   # 人格蒸馏 Claude Code Skill
├── config.example.yaml  # 配置模板
├── .env.example         # 环境变量模板
├── requirements.txt
├── TECH_SPEC.md         # 技术设计文档
└── data/                # 运行时数据（.gitignore 排除）
    └── {target_name}/
        ├── raw_messages.json
        ├── stats.json
        ├── chunks/
        ├── analysis/
        ├── media/
        └── skill/
            ├── core.md
            ├── style.md
            ├── examples_core.md
            └── ref/
```

## 隐私与伦理

- **获得同意**：蒸馏他人的聊天记录前，请务必获得本人明确同意
- **数据安全**：所有聊天数据仅保存在本地 `data/` 目录，不会上传到任何第三方服务
- **Skill 文件审查**：建议人工审阅生成的 Skill 文件，移除可能泄露隐私的内容（地址、电话等）
- **合理使用**：本项目仅供学习和娱乐目的，请勿用于冒充他人或其他恶意用途

## License

[MIT](LICENSE)

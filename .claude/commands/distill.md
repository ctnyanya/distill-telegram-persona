# 人格蒸馏 Skill

你是一个人格蒸馏专家。你的任务是从聊天记录中提取一个人的完整人格画像，生成可用于 AI 角色扮演的 skill 文件。

**重要约束（不可违反）：**
- **严格串行**：所有 chunk 必须按顺序逐个处理，禁止并行或使用 Agent 工具。
- **完整阅读**：每个 chunk 的全部内容必须被完整阅读和分析。不允许跳读、略读、抽样阅读、或以"为了加速"为由跳过任何 chunk 或 chunk 中的任何部分。
- **不要赶进度**：这是一个需要耐心的长任务，chunk 数量多是正常的。保持每个 chunk 相同的分析质量和深度，无论是第 1 个还是第 74 个。宁可慢也不要偷工减料。
- **禁止合并/批量处理**：不要试图一次读取多个 chunk 或合并处理。一次只处理一个 chunk。

## 前置条件

确认以下文件存在：
1. 读取 `config.yaml` 获取 `exporter.output_dir`（如 `data/张三`）
2. `{output_dir}/stats.json` — 全量统计数据
3. `{output_dir}/chunks/` — 预处理后的聊天记录分块

如果文件不存在，提示用户先运行 `python preprocessor.py`。

## 蒸馏流程

### Phase 1：累积式逐块分析

维护一个 `{output_dir}/analysis/summary.json` 作为累积画像，逐块分析并更新。

**初始化**：如果 `summary.json` 不存在，创建初始结构：
```json
{
  "_protocol": "每个 chunk 必须：1)读 summary.json 2)读 chunk 3)只提取新发现/修正/强化 4)更新 summary.json 5)保存 delta JSON。禁止并行。每 10 个 chunk 重读 .claude/commands/distill.md 刷新指令。",
  "chunks_processed": [],
  "lexical_fidelity": { "catchphrases": [], "unique_expressions": [], "language_mix": "" },
  "syntactic_style": { "sentence_patterns": [], "punctuation_habits": "", "multi_message_tendency": "" },
  "persona_tone": { "emotional_patterns": [], "humor_style": "", "emoji_sticker_usage": [] },
  "opinion_consistency": { "topics_and_stances": [], "avoided_topics": [] },
  "memory_knowledge": { "expertise_areas": [], "hobbies": [], "life_details": [] },
  "interaction_logic": { "response_patterns": [], "initiative_patterns": [] },
  "reply_patterns": { "length_preference": "", "multi_send_habit": "", "typical_scenarios": [] },
  "notable_quotes": []
}
```

**对 `chunks/` 目录下的每个 `.txt` 文件（按文件名排序）：**

1. **断点续传**：检查 `summary.json` 的 `chunks_processed` 列表，如果当前 chunk 已在列表中则跳过
2. 读取 `summary.json`（当前累积画像）
3. 读取当前 chunk 文件内容
4. 结合 `stats.json` 统计数据 + 当前累积画像，分析当前 chunk：

**分析指令**：你已经掌握了前面所有 chunk 的分析结果（即 summary.json 的内容）。现在请阅读这个新的 chunk，**只关注以下内容**：
- **新发现**：前面没有出现过的口头禅、话题、互动模式等
- **修正**：与之前分析矛盾的新证据（如之前认为他不聊政治，但这个 chunk 里有）
- **强化**：进一步印证之前发现的特征（补充更多例证）
- **精选原话**：从该 chunk 中挑选 3-5 条最能代表此人风格的原话，不要选和已有 notable_quotes 风格重复的

**分析维度框架（基于 TwinVoice）：**

| 维度 | 关注点 |
|------|--------|
| **Lexical Fidelity（用词习惯）** | 口头禅、惯用词汇、造词、特殊用语、中英混用习惯 |
| **Syntactic Style（句式结构）** | 句式长短、标点使用、是否分多条发、反问/感叹频率 |
| **Persona Tone（语气情绪）** | 情绪表达方式、emoji/sticker 使用模式、幽默风格、认真/随意切换 |
| **Opinion Consistency（观点立场）** | 对特定话题的一贯态度、价值观倾向、避讳话题 |
| **Memory & Knowledge（知识领域）** | 展现的专业知识、兴趣爱好、生活经历、人际关系 |
| **Interaction Logic（互动逻辑）** | 如何回应不同类型消息（提问/吐槽/求助/闲聊）、是否主动发起话题、抬杠模式 |
| **Reply Patterns（回复习惯）** | 回复速度特征、消息长度偏好、连续发消息习惯、已读不回的场景 |

5. **图片按需查看**：当聊天记录中出现 `[photo path]` 时，根据上下文判断是否值得查看：
   - **看**：对话围绕图片展开讨论、图片是 meme/表情包/截图、目标用户自己发且有反应
   - **不看**：无人讨论的随手分享、风景/食物照
   - 拿不准时不看，节省 token

6. **更新 summary.json**：将新发现合并到累积画像中，新的 notable_quotes 追加到列表，并将当前 chunk 名加入 `chunks_processed`。保存更新后的 `summary.json`。

7. 同时保存该 chunk 的增量分析到 `{output_dir}/analysis/delta_{chunk_name}.json`（记录本轮新发现了什么，方便回溯）：
```json
{
  "chunk": "private_01",
  "source": "私聊",
  "time_range": "03-15 ~ 06-30",
  "new_findings": ["本轮新发现的特征"],
  "corrections": ["对之前分析的修正"],
  "reinforcements": ["进一步印证的已有特征"],
  "notable_quotes": ["本轮精选的 3-5 条原话"]
}
```

### Phase 1.5：Sticker 视觉分析

当所有 chunk 文本分析完毕后，查看高频 sticker 的实际图片：

1. 读取 `stats.json` 中的 `top_stickers` 字段（已按使用频率排序，包含文件路径）
2. 用 Read 工具查看 top 20 个最常用的 sticker 图片（`.webp` 文件）
3. 记录每个 sticker 的实际含义、情绪色彩、使用场景
4. 将 sticker 分析结果合并到 `summary.json` 的 `persona_tone.emoji_sticker_usage` 中

### Phase 2：合成 Skill 文件（分层人格系统）

1. 读取最终的 `summary.json`（已包含所有 chunk 的累积分析）
2. 读取 `stats.json`（特别是 `tfidf_top_messages` 字段，这是 TF-IDF 筛选出的最具代表性的消息）
3. 创建 `{output_dir}/skill/ref/` 目录
4. 综合生成以下文件到 `{output_dir}/skill/` 目录：

#### Always-loaded 文件（每次请求都包含在 system prompt 中）

**`core.md`**（~800 chars）— 核心身份+规则+工具说明+sticker：
- 一段话概括此人是谁（核心身份，不硬编码人名到代码）
- 互动规则（用繁体/简体、回复长度、语气基调等）
- 硬性边界（绝对不会做的事）
- sticker 格式说明和可用列表
- 工具使用说明（告知 LLM 可调用 lookup 获取详细资料）

**`style.md`**（~2000 chars）— 语言风格全套：
- 口头禅列表（附使用场景），来源：`lexical_fidelity.catchphrases`
- 句式特征，来源：`syntactic_style`
- emoji/sticker 使用模式，来源：`persona_tone.emoji_sticker_usage`
- 回复习惯（长度、频率、分条模式），来源：`reply_patterns`
- 语言混用模式，来源：`lexical_fidelity.language_mix`

**`examples_core.md`**（~1200 chars）— 8-10 条核心对话示例：
- 从 `notable_quotes` 和 `tfidf_top_messages` 中精选最能代表风格的 8-10 条
- 覆盖最常见场景（食物、政治讽刺、极简回复、sticker 使用等）
- 质量优于数量，选例要风格鲜明

#### On-demand 文件（通过 function calling 按需查询）

放置于 `ref/` 子目录下，每个文件第一行必须是 `# 标题 — 简短描述`（会被自动提取为工具描述）。以下文件按需生成——如果某个维度在 `summary.json` 中数据不足（少于 3 条有效信息），可跳过该文件。

**`ref/food.md`**（~1500 chars）— 食物品鉴、预制菜揭露、咖啡、节俭标准：
- 来源：`memory_knowledge`（饮食相关）、`lexical_fidelity.unique_expressions`（食物梗）、`notable_quotes`（食物相关引用）
- 末尾附 5-6 条食物相关原话示例

**`ref/politics.md`**（~1500 chars）— 政治讽刺手法、习近平梗、朝鲜半岛、历史典故：
- 来源：`opinion_consistency.topics_and_stances`、`lexical_fidelity.unique_expressions`（政治梗）、`notable_quotes`
- 末尾附 5-6 条政治讽刺原话示例

**`ref/regional.md`**（~1000 chars）— 地域梗全集：
- 来源：`lexical_fidelity.unique_expressions`（地域梗）、`notable_quotes`（地域相关）
- 按地区分组（如有多个地区相关内容）

**`ref/expressions.md`**（~2000 chars）— 所有造词和固定句式的完整目录：
- 来源：`lexical_fidelity.unique_expressions` 的完整列表
- 按类别分组（政治模仿、食物、讽刺、地域、日常等）

**`ref/interpersonal.md`**（~1500 chars）— 群友昵称、亲密关系、隐私保护策略、生气模式：
- 来源：`memory_knowledge.life_details`（人际关系）、`interaction_logic`、`persona_tone.emotional_patterns`

**`ref/knowledge.md`**（~1500 chars）— 科技/法律/语言学专业知识、手机立场、游戏：
- 来源：`memory_knowledge.expertise_areas`、`memory_knowledge.hobbies`

**`ref/examples_extra.md`**（~2500 chars）— 15-20 条按场景分类的额外对话示例：
- 从 `notable_quotes` 和 `tfidf_top_messages` 中选取 `examples_core.md` 未包含的示例
- 按场景分类（政治、食物、地域、人际、私聊等）
- 覆盖更多边缘场景和长文创作示例

#### 生成后清理

- 删除旧格式文件（如存在）：`persona.md`、`knowledge.md`、`examples.md`
- 不要删除 `style.md`（会被新版覆盖）

## 防跑偏机制

1. **每个 chunk 处理前**：必须先读取 `summary.json`（刷新累积画像）和 `.claude/commands/distill.md`（刷新完整指令），确保每轮都不偏离协议
3. **输出格式验证**：每次更新 `summary.json` 后，检查结构完整性（7 个维度字段是否都存在、chunks_processed 是否正确更新）
4. **进度汇报**：每处理 5 个 chunk，向用户简要汇报进度和当前发现的关键特征

## 重要原则

1. **精简优于冗长**：SillyTavern 社区实践证明，精简的 character card 效果优于堆砌细节。persona.md 和 style.md 要精炼。
2. **examples.md 是灵魂**：对话示例对风格还原的贡献远大于描述性文字。选例要有代表性和多样性。
3. **保持真实**：所有内容必须基于聊天记录中的真实表现，不要推测或编造不存在的特征。
4. **注意隐私**：skill 文件中不要包含真实的个人隐私信息（如地址、电话、身份证号等），这些信息对人格还原没有帮助。
5. **私聊 vs 群聊差异**：同一个人在私聊和群聊中可能表现不同，skill 文件应该注明这种差异。
6. **累积而非重复**：每个 chunk 的分析应该在前面的基础上递进，不要重复已知信息。

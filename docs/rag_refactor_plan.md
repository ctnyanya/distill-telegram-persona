# RAG 重构规划：人格 bot 的记忆升级

> 把 bot 的"记忆"从滑动窗口 + 静态分类检索，升级为向量召回 + 未来可扩展到 hybrid 检索的 RAG 架构。

---

## 1. 背景与目标

### 1.1 痛点

Bot 的人格风格已经调到满意，但**记忆被大幅削弱**。具体表现：

- `message_buffer` 只保留最近 20 条群聊作为上下文
- `memory.json` 只按时间线保留最近 30 条记忆，不按相关性召回
- 过去几年真实聊天记录（`raw_messages.json`，~30 MB）只用来离线蒸馏了一次就躺在硬盘里
- `examples_core.md` 是固定的 few-shot，无法根据当前话题动态调出"他过去聊这个话题时说过的话"

### 1.2 目标

通过 RAG 解决"记忆"问题，**不改变人格风格**。具体：

1. **历史记忆**：让 bot 在回复前，能根据当前话题召回真实本人过去聊过的相关片段
2. **运行时记忆**：让 bot 记住上线后跟群友的长期互动，突破 20 条滑动窗口限制
3. **为 B 方案打好地基**：A 方案的数据结构、接口、存储选型要能无痛升级到 hybrid search + rerank + 时间衰减

### 1.3 非目标（明确不做的事）

- ❌ 替换现有人格 prompt（`core.md / style.md`）——风格是先验，不是检索问题
- ❌ 引入 LangChain / LlamaIndex 等大框架
- ❌ 外部向量数据库（Chroma / Pinecone / Qdrant 托管版等）
- ❌ A 阶段就做 hybrid search、rerank、时间衰减（这些留给 B）

---

## 2. 现状分析

### 2.1 当前的"类 RAG"机制

项目已经有一个简易的静态分类检索：

```
模型侧                     代码侧
──────                     ──────
调用 lookup("politics")  → bot/bot.py:156 handle_tool_call
                         → 读 skill/ref/politics.md
                         → 整份返回给模型
```

这是 **RAG 的最简形态**（静态分类 + 粗粒度），但有两个大问题：
- 颗粒度是"整个文件"，返回的内容里大部分无关
- 依赖模型主动调用，模型可能漏调或分类错

### 2.2 当前的记忆机制

| 机制 | 存储 | 容量 | 策略 | 问题 |
|------|------|------|------|------|
| 群聊上下文 | `message_buffer`（内存） | 200 条 → 注入 20 条 | 滑动窗口 | 重启丢失、不按相关性 |
| 私聊上下文 | `private_history`（内存） | CONTEXT_WINDOW × 2 | 滑动窗口 | 重启丢失 |
| 长期记忆 | `memory.json` | 50 条上限，注入 30 条 | 模型主动 `remember` | 线性堆积，不按相关性 |
| 人物档案 | `skill/people/*.md` | 按 user_id | 活跃用户自动注入 | 静态，不随互动更新 |
| 历史聊天 | `raw_messages.json` | 30 MB | **完全没用上** | 🎯 RAG 的机会 |

---

## 3. 方案总览

### 3.1 核心思路

在现有骨架上**叠加**一层向量召回，不替换任何已有机制：

```
现有：         system prompt = core.md + style.md + examples_core.md
               + people 档案（按活跃用户）
               + 最近 30 条 memory

新增：         + RAG 召回结果（top-k 历史片段 + top-k runtime 交互）
```

### 3.2 Embedding 方案选择

确定方案：**本地 ONNX embedding**（`fastembed` + `BAAI/bge-small-zh-v1.5`）

| 考量维度 | 说明 |
|---------|------|
| 中转站限制 | 用户的 Gemini/OpenAI 都走中转站，embedding endpoint 不一定支持，路径不稳 |
| 隐私 | 聊天记录完全不出本地 |
| 依赖轻量 | `fastembed` 基于 ONNX runtime，不需要 PyTorch（对比 `sentence-transformers` 省 2 GB 依赖） |
| 模型 | `bge-small-zh-v1.5`：512 维，~100 MB，中文对话召回效果足够 |
| 成本 | 零 API 调用、零月费 |

### 3.3 部署位置

两种场景，同一套代码：

| 场景 | 在哪跑 | 做什么 |
|------|--------|--------|
| 离线建索引 | 本地 WSL（一次性） | 读 `raw_messages.json` → chunk → embed → 写入 `rag.db` |
| 运行时召回 | Railway（常驻 bot） | 读 `rag.db` + 加载 fastembed → 每次回复前做 query embedding → 向量检索 → 注入 prompt |
| 运行时增量 | Railway | bot 产生新交互时 embed 一次写进 `rag.db` |

**Railway 内存风险**：fastembed 加载后约 300 MB，叠加 bot 本身 ~200 MB，总计 ~500 MB。Railway 免费套餐（512 MB 起）**可能吃紧**。对策见第 6 节。

---

## 4. 架构设计

### 4.1 数据模型（SQLite）

一张表装两种来源，`source` 字段区分：

```sql
CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL,          -- 'historical' | 'runtime'
    text          TEXT NOT NULL,          -- 拼好的对话片段（带 speaker 前缀）
    speakers      TEXT NOT NULL,          -- JSON 数组：['张三', '小王']
    timestamp     INTEGER NOT NULL,       -- unix 秒，片段末尾时间
    chat_id       INTEGER,                -- Telegram chat id
    msg_id_start  INTEGER,
    msg_id_end    INTEGER,
    embedding     BLOB NOT NULL           -- numpy float32 bytes (512 维)
);

CREATE INDEX idx_source_ts ON chunks(source, timestamp DESC);

-- FTS5 全文索引：A 阶段不用，但建好等 B 阶段直接上 BM25
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);

-- 触发器：chunks 增删时自动同步 FTS
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
-- ...（同样的 au / ad 触发器）
```

**为什么这样设计**：

- 一张表两种来源：召回时可以按 source 过滤（"只要历史"、"只要 runtime"），也可以混合召回后分别注入
- `timestamp / speakers / chat_id`：B 阶段做时间衰减、按说话人过滤、跨群召回时都要用，**现在就得存上**，否则回头补索引痛苦
- `embedding BLOB`：存 numpy float32 bytes，读出来 `np.frombuffer` 即可——A 阶段数据量不大（预计几万 chunk），暴力余弦相似度够用，不需要 ANN（HNSW / IVF）
- FTS5 现在建着不用：B 阶段叠加 BM25 时直接 `SELECT ... MATCH`，零数据迁移

### 4.2 模块划分

```
bot/rag/                         （新建目录）
├── __init__.py
├── schema.sql                   建表语句
├── store.py                     SQLite 读写封装（add_chunk, iter_all, ...）
├── embedder.py                  fastembed 封装（embed(text) → np.ndarray）
├── chunker.py                   两种 chunker：historical / runtime
└── retriever.py                 retrieve(query, k, source=None) → list[Chunk]

scripts/
├── build_rag_index.py           离线建索引脚本（读 raw_messages → chunks 表）
└── query_rag.py                 调试小工具（命令行测召回质量）

bot/bot.py （改动）
  + build_llm_messages 中调用 retriever
  + on_group_message / on_private_message 末尾写 runtime chunk
  + config.yaml 新增 rag 配置块
```

### 4.3 Chunking 策略

#### 4.3.1 Historical chunker（离线）

**规则**：
- 输入：`raw_messages.json`（一个群或一个私聊的完整消息流）
- 每 **15 条消息**切一块，相邻块 **重叠 3 条**
- 过滤：纯 sticker 消息、纯 URL、纯"[发了一张图片]"这类噪音跳过（参考 `bot.py:_filter_context` 的逻辑）
- 太短的片段丢弃（拼完后 <30 字符）
- Chunk 文本格式：每条消息一行，`[发送者]: 内容`（和现有 `format_msg` 对齐）

**为什么**：
- 固定数量简单鲁棒，不用处理"一个大话题连续 100 条该怎么办"
- 重叠保证话题不会正好被切断
- 保留完整对话（含别人的话）：目标人的单句"嗯嗯"脱离上下文没信息量

#### 4.3.2 Runtime chunker（在线）

**规则**：
- 触发点：每次 bot 在群聊/私聊中产生一个回复后
- 一个 chunk = **触发消息** + **前 2 条上下文** + **bot 的回复**
- 立即 embed 并写入 `rag.db`（`source='runtime'`）

**为什么**：
- 粒度以"一次交互"为单位，语义最完整
- 前 2 条上下文让后续召回能还原出"当时在聊什么"

### 4.4 数据流

#### 4.4.1 建索引（一次性，本地）

```
raw_messages.json
    ↓ 按 chat 分组
    ↓ 过滤噪音消息
Historical chunker（15 条 + 重叠 3 条）
    ↓
[chunks: text, speakers, timestamp, ...]
    ↓ 批量 embed（fastembed，batch=32）
[chunks with embedding]
    ↓
SQLite: rag.db
```

预计产出：**几千到几万条 chunk**（取决于原始消息数量），文件 **20-200 MB** 级。

#### 4.4.2 运行时召回（每次 bot 回复）

```
触发消息 "今天喝啥奶茶"
    ↓
embedder.embed(query)   ← fastembed，~30ms
    ↓ 
SELECT * FROM chunks    ← 全表加载（几万条，<100MB numpy 向量）
    ↓ 内存里算余弦 top-k
[top-3 historical + top-3 runtime]
    ↓
build_llm_messages: 注入到 user message 前面
    ↓
LLM 生成回复
    ↓
runtime chunker：打包这轮交互 → embed → 写入 rag.db（异步不阻塞回复）
```

**性能估算**（A 阶段，几万条 chunk）：
- Query embedding：~30 ms
- 向量相似度计算：numpy 矩阵乘法 ~10 ms
- SQLite 加载：首次 ~200 ms，后续缓存 ~30 ms
- **总召回延迟 <100 ms**，可接受

### 4.5 注入 prompt 的格式

在 `build_llm_messages` 构造的 user 消息前插入：

```
## 相关历史对话（来自你过去的真实聊天记录）

[2024-08-12 群聊]
小王: 今天喝什么
张三: 三分糖去冰吧
小王: 又三分糖...
张三: 我怕甜

[2024-11-03 群聊]
...

## 最近的相关互动（你和群友）

[2026-04-10]
大橘: 看见你昨天发的那张图了
你: 嗯嗯那是我猫

---

以下是群聊中最近的消息：
（原来的 chat_log）
```

## 5. 实施计划

### Phase 0：准备（只有你能做）

- [ ] 确认 Railway 套餐（免费/Hobby/Pro）内存上限
- [ ] （可选）在 `yunwu.ai` 文档/控制台搜 `/v1/embeddings` 是否支持——不支持也不影响本方案

### Phase 1：离线建索引（本地）

**代码交付：**

- `bot/rag/schema.sql`
- `bot/rag/store.py`
- `bot/rag/embedder.py`
- `bot/rag/chunker.py`
- `scripts/build_rag_index.py`
- `scripts/query_rag.py`（调试工具）

**你要做的：**

1. 安装依赖：
   ```bash
   pip install fastembed
   ```
2. 跑建索引脚本：
   ```bash
   python scripts/build_rag_index.py --target 张三
   ```
   首次会下载模型（~100 MB）。跑完后会在 `data/<target>/rag.db` 生成索引
3. **检验召回质量**（这一步最重要）：
   ```bash
   python scripts/query_rag.py "奶茶"
   python scripts/query_rag.py "最近在忙什么"
   python scripts/query_rag.py "哈气夙"  # 换成实际话题
   ```
   看返回的 top-5 片段是不是合理。如果 chunker 切得不好，要在这一步调整，**不要**带着坏索引往下走

**验收标准：**

- top-3 里至少 2 条和 query 话题相关
- 片段不会在句中被截断
- 没有大片纯 sticker / URL 的噪音

### Phase 2：接 bot（本地测试）

**代码交付：**

- `bot/rag/retriever.py`
- 改 `bot/bot.py`：
  - `build_llm_messages` 中调用 retriever，注入召回结果
  - `on_group_message` / `on_private_message` 末尾写 runtime chunk（异步）
- 改 `config.yaml`：新增 `rag` 配置块
  ```yaml
  rag:
    enabled: true
    db_path: "data/张三/rag.db"
    historical_k: 3
    runtime_k: 2
    runtime_mode: "vector"   # 备用 "bm25" 见第 6 节降级方案
  ```
- 新增 log：每次召回打印 top-k 片段的 id + 相似度分数（线上验证用）

**你要做的：**

1. 本地启动 bot，用测试账号或测试群发几条消息
2. 观察 log：召回的片段合理吗？
3. 调整 `historical_k` / `runtime_k`（通常 3+2 起步，太多会稀释主任务）
4. 如果效果差，把 log 截图或复制给我，一起分析是 chunker 问题还是注入格式问题
5. 回滚测试：把 `rag.enabled: false`，确认 bot 回到原行为

**验收标准：**

- Bot 回复里能体现"记得"过去聊过的事
- 召回耗时 <200 ms
- rag 关闭后行为完全等同重构前

### Phase 3：部署 Railway

**代码交付：**

- `requirements.txt` 加 `fastembed`
- 可能需要改 Dockerfile（视 fastembed 首次加载模型的路径）

**你要做的：**

1. **关键决定：`rag.db` 怎么上 Railway？**（等 Phase 1 跑完知道文件大小后再定）
   - 方案 A：直接 commit 到 `deploy` 分支（简单，但文件大可能超 git 限制）
   - 方案 B：放 Railway Volume（需要配置）
   - 方案 C：启动时从对象存储拉取（S3/R2，最灵活但要配 credentials）
2. 按 `CLAUDE.md` 约定推送：
   ```bash
   git checkout deploy
   git merge main
   git push deploy deploy:main
   ```
3. Railway 控制台看启动日志，监控：
   - 有无 OOM（内存溢出）
   - 首次冷启动耗时（fastembed 初始化要几秒）
   - 前几条消息的召回表现

**验收标准：**

- Railway 容器稳定运行不 OOM
- 线上 bot 响应延迟没有显著增加（<500 ms 额外）
- 线上 bot 召回效果和本地一致

---

## 6. 风险与应对

### 6.1 风险 A：Railway 免费套餐 OOM

**触发条件**：Railway 512 MB 限制下，fastembed 模型 + Python + aiogram + bot 状态 ≈ 500 MB，没有余量。

**预案（在 A 阶段就埋好开关）**：

代码里 `retriever.py` 支持两种运行模式，通过 `config.yaml` 的 `rag.runtime_mode` 切换：

- `"vector"`（默认）：本地 embed query → 向量召回
- `"bm25"`（降级）：不加载 fastembed，直接用 SQLite FTS5 做关键词召回

这样 Railway 真跑不起来时，**不用改代码**，只改 config 字段一键降级。缺点是降级后失去语义相似匹配（只剩关键词），但对中文短 query 效果其实可接受。

> **注意**：中文 FTS5 unicode61 tokenizer 按字切词，BM25 效果会打折。B 阶段会接 jieba tokenizer，A 阶段先这样。

### 6.2 风险 B：rag.db 太大，git 推不上去

**触发条件**：聊天记录数万条、chunk 数万、每条向量 2 KB → `.db` 文件 100+ MB，超 git 单文件 100 MB 硬限制。

**预案**：

- 建索引时可以省略 FTS5 的 content 副本（只建 external content）
- 真超过 100 MB 就走 Railway Volume 或对象存储方案
- 用 `git-lfs` 是**不推荐**的（双 repo 结构下 LFS 会把敏感数据的 pointer 推到 origin，反而复杂）

### 6.3 风险 C：chunking 切得不好，召回质量差

**触发条件**：Phase 1 验收时发现 top-k 大量无关片段。

**预案**（按成本从低到高）：

1. 调 chunk size：15 → 20 或 10
2. 调重叠：3 → 5
3. 换策略：从"固定消息数"换成"时间间隔 > 30min 切"
4. 重新过滤：加严噪音过滤规则

都在 `chunker.py` 一个模块里改，重跑 `build_rag_index.py` 即可，**不影响其他部分**。

### 6.4 风险 D：运行时召回拖慢 bot 响应

**触发条件**：数据量大到几十万 chunk，线性扫描向量 >1s。

**预案**：A 阶段不考虑（数据量远不到），B 阶段上 `sqlite-vec` 或切到 FAISS。

---

## 7. 未来演进（B 方案）

A 方案的架构决策全部为 B 铺路。B 阶段增量（都是叠加，不改动 A 代码）：

### 7.1 Hybrid search（向量 + BM25）

- A 阶段已经建了 FTS5 表，B 只需要在 `retriever.py` 新增一条 BM25 查询路径
- 用 **RRF (Reciprocal Rank Fusion)** 融合两路 top-k
- 换 jieba tokenizer 改善中文 BM25

### 7.2 Rerank

- 在 retriever 末尾加一层：召回 top-20 → rerank top-5
- 方案：
  - 本地 cross-encoder（`BAAI/bge-reranker-base`）
  - 或调 LLM rerank（用 bot 现有的 yunwu 接口）

### 7.3 时间衰减

- A 已经存了 `timestamp` 字段
- B 在最终分数上乘 `exp(-λ·age)`，新片段加权
- runtime 片段衰减要不要更慢？（可能想让 bot 更记得最近的交互）

### 7.4 召回门控

- A 默认 always-on
- B 可以让 LLM 自己决定"这次要不要召回"——把 retrieve 变回工具，模型不调就省一次 embedding

### 7.5 跨 target 共享索引

- 如果未来蒸馏第二个人格，可以共用同一个群的 historical chunks（只需按 speaker 过滤召回）

---

## 8. 你的手动 checklist

### Phase 0（决策）

- [x] 3 个设计决策（chunk、召回时机等）：全部按推荐默认
- [ ] 告知 Railway 套餐：免费

### Phase 1（本地建索引）

- [ ] `pip install fastembed`
- [ ] 跑 `scripts/build_rag_index.py`（首次会下载 ~100 MB 模型）
- [ ] 用 `scripts/query_rag.py` 测 3-5 个 query，检验召回质量
- [ ] 把召回结果截图/复制给我，一起判断要不要调 chunker

### Phase 2（本地测试 bot）

- [ ] 本地启动 bot，和测试账号对话
- [ ] 观察 log 里的召回片段是否合理
- [ ] 调 `historical_k` / `runtime_k`
- [ ] 验证 `rag.enabled: false` 能干净回滚

### Phase 3（上线）

- [ ] 决定 `rag.db` 上 Railway 的方式（等 Phase 1 跑完看文件大小）
- [ ] 按双 repo 约定推 `deploy` 分支
- [ ] 监控 Railway log（OOM、启动耗时、召回延迟）
- [ ] 观察线上 bot 有没有显著"记性变好"的反馈

**总耗时估算**：你的手动时间 1.5-2 小时，分散在多个检查点。每个 Phase 跑完我会停下来等你验证。

---

## 9. 决策记录（Decision Log）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 1 | Chunk 策略 | 固定 15 条 + 重叠 3 条 | 简单鲁棒，切不出极端大小 |
| 2 | Chunk 内容 | 完整对话（含别人的话） | 单句脱离上下文信息量低 |
| 3 | 召回时机 | Always-on（每次回复都召回） | 不依赖模型判断、反馈直接 |
| 4 | Embedding 模型 | `BAAI/bge-small-zh-v1.5`（fastembed/ONNX） | 轻量、免费、中文效果足够、不依赖中转站 |
| 5 | 向量存储 | SQLite（单文件 + FTS5 预埋） | 零运维、易备份、B 阶段 BM25 零迁移 |
| 6 | 检索算法 | 暴力余弦（全表 numpy） | A 阶段数据量远低于 ANN 收益阈值 |
| 7 | 部署策略 | 离线建索引 + 运行时加载 | 聊天记录不出本地，runtime 只加载模型 |
| 8 | Railway OOM 预案 | config 开关切 bm25-only 降级 | 零代码改动可回滚 |
| 9 | 不用 LangChain | - | 项目小，大框架心智负担大于收益 |
| 10 | 不用外部向量 DB | - | 单文件 SQLite 对 A 阶段绰绰有余，且避免第三方依赖 |

---

## 10. 开始之前的确认清单

开始写代码前，请确认：

- [ ] 这份规划理解无异议
- [ ] Railway 免费套餐确认（影响 Phase 3 风险评估）
- [ ] 可以新增 `bot/rag/` 目录和 `scripts/` 目录
- [ ] 同意在 `config.yaml` 新增 `rag` 配置块
- [ ] 同意在 `requirements.txt` 加 `fastembed` 和 `numpy`

确认后从 **Phase 1** 开始写代码，第一步交付 `bot/rag/schema.sql + store.py + embedder.py + chunker.py + scripts/build_rag_index.py`，约 250-400 行代码。

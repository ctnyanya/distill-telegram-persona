# 项目规范

## 双 Repo 架构（重要！）

本项目有两个 GitHub remote，**绝对不能搞混**：

| Remote | Repo | 可见性 | 用途 | 对应本地分支 |
|--------|------|--------|------|-------------|
| `origin` | `distill-telegram-persona` | **PUBLIC** | 开源项目 | `main` |
| `deploy` | `telegram-susu-skill` | **PRIVATE** | Railway 部署 | `deploy` |

### 敏感文件规则

以下文件**只能出现在 `deploy` 分支，绝对不能推送到 `origin`**：

- `config.yaml` — 含 Telegram API key、user ID、group ID
- `data/` — 人格数据、聊天记录、sticker 文件
- 任何含密钥、token、个人信息的文件

### 推送流程

```bash
# 代码改动 → commit 到 main → 推 origin
git push origin main

# 部署 → 切到 deploy → merge main → 推 deploy remote
git checkout deploy
git merge main
git push deploy deploy:main
```

### 绝对禁止

- **禁止** 对 `data/` 或 `config.yaml` 使用 `git add -f` 然后推到 origin
- **禁止** 把 `main` 直接推到 `deploy` remote（`git push deploy main:main`）
- **禁止** 在 `main` 分支上 commit 任何 `data/` 或 `config.yaml` 文件
- 如果不确定，先 `git ls-files | grep -E '(config\.yaml|data/)'` 检查

## 开发约定

- 始终使用中文交流
- Bot 部署在 Railway，从 `deploy` remote 的 main 分支拉取

# Hammer Code 使用命令

以下命令以 Windows PowerShell、Python 3.11+ 和 uv 为基准。在仓库根目录执行。uv 负责创建和管理 `.venv`，不需要先手动激活虚拟环境。

## uv 与虚拟环境

使用 `uv` 命令，不要在已激活 `.venv` 后使用 `python -m uv`。激活后 `python` 指向 `.venv\Scripts\python.exe`，而 uv 不安装在项目虚拟环境中，因此 `python -m uv` 会报 `No module named uv`。

```powershell
# 推荐：在普通 PowerShell 或已激活的 .venv 中都使用 uv
uv --version
uv sync --all-groups
```

如果刚把 uv 的 Scripts 目录加入用户 PATH，当前已打开的终端不会自动刷新。可关闭并重新打开 PowerShell；或只为当前窗口执行：

```powershell
$env:Path = "C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts;$env:Path"
uv --version
```

PATH 仍不可用时，可直接调用 uv 可执行文件：

```powershell
& "C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts\uv.exe" sync --all-groups
```

## 安装与准备

```powershell
# 创建或同步项目虚拟环境，并安装运行与开发依赖
uv sync --all-groups

# 从无密钥示例创建项目本地配置
Copy-Item .hammer-code\config.example.toml .hammer-code\config.toml
```

`.hammer-code/config.toml` 已被 Git 忽略。请在其中填写模型和 endpoint，但不要写入 API key。

## 查看帮助

这两种入口都不需要配置文件或 API key：

```powershell
uv run python -m hammer_code --help
uv run hammer-code --help
```

当前命令行参数：

| 参数 | 说明 |
| --- | --- |
| `--config PATH` | 明确指定 TOML 配置文件路径。路径不存在时直接报错。 |
| `--profile NAME` | 覆盖配置中的 `default_profile`，选择指定 profile。 |
| `--permission-mode MODE` | 以 `default`、`accept_edits`、`strict` 或 `unattended` 启动权限模式。`unattended` 仍要求启动确认。 |
| `--resume ID_OR_LATEST` | 恢复与当前 profile/协议兼容的本地 Session。 |
| `--list-sessions` | 列出有效 Session 后退出；不初始化模型客户端、MCP 或 Skill。 |
| `-h` / `--help` | 显示命令帮助并退出。 |

## 启动对话

先在当前 PowerShell 会话设置所选 profile 对应的环境变量：

```powershell
$env:OPENAI_API_KEY = "你的密钥"
uv run hammer-code
```

也可以显式指定配置文件和 profile：

```powershell
$env:ANTHROPIC_API_KEY = "你的密钥"
uv run hammer-code --config .hammer-code\config.toml --profile anthropic
```

未提供 `--config` 时，Hammer Code 从当前目录逐级向上查找最近的 `.hammer-code/config.toml`，最多查到 Git 根目录。找不到配置、profile 不存在或环境变量缺失时不会发起网络请求，而是显示错误并以状态码 2 退出。

## 显示思考内容

在 `.hammer-code/config.toml` 中开启 UI 展示开关，然后重新启动 Hammer Code：

```toml
[ui]
show_reasoning = true
```

`show_reasoning = true` 表示把供应商返回的 reasoning/thinking 内容实时输出到终端。设为 `false` 时不会显示真实内容，每轮只显示一个可动态更新的 `Thinking…` 状态，不会按流式分块重复换行。

UI 开关只控制是否展示；所选 profile 也必须请求或支持 reasoning 内容。将下面相应字段合并到已有 profile 中，不要重复声明同一个 TOML 表头：

```toml
# OpenAI Responses：在对应 profile 中请求 reasoning summary
[profiles.openai]
protocol = "openai_responses"
# 保留该 profile 原有的 model、base_url、api_key_env 等字段
reasoning_effort = "medium"
reasoning_summary = "auto"

# Anthropic Messages：在对应 profile 中启用 thinking
[profiles.anthropic]
protocol = "anthropic_messages"
# 保留该 profile 原有的 model、base_url、api_key_env 等字段
thinking_mode = "enabled"
thinking_budget = 1024
```

对于 `openai_chat_completions` 兼容服务，无需额外的 Hammer Code 展示字段；只有服务实际返回 `reasoning_content` 或 `reasoning` 流字段时才会显示。供应商不返回 reasoning 时，即使 `show_reasoning = true` 也没有可展示内容。签名、redacted thinking 等 opaque provider state 始终不会显示。

## 交互内置命令

这些命令只在本地处理，不会发送给模型：

| 命令 | 作用 |
| --- | --- |
| `/help` | 显示内置命令说明。 |
| `/usage` | 显示当前会话中供应商报告的 Token usage；缺失数据会显示为 `unavailable`，不会伪造为 0。 |
| `/clear` | 清除当前进程内的消息历史与 usage，不会删除任何文件。 |
| `/compact` | 请求在下一次模型调用前压缩当前上下文。 |
| `/permission [mode]` | 切换为 `default`、`accept_edits`、`strict` 或 `unattended`；后者需要确认。 |
| `/session list|new|resume|delete` | 管理同一 profile/协议下的本地 Session。 |
| `/memory list|read` | 只读查看项目 Memory 目录与已索引主题。 |
| `/skill <name|scope:name> [arguments]` | 运行一个本地 Skill；可用 `project:name` 或 `user:name` 精确指定被遮蔽的版本。 |
| `/feedback <name|scope:name> <feedback>` | 对指定 Skill 提交维护反馈；仅 Evolution 启用且权限允许自动写入时可用。 |
| `/exit` | 关闭客户端并退出。 |

空输入会被忽略。普通模型 Tool Call 会先经过本地参数校验、路径/命令策略和 PermissionService；未公开、已禁用、无效或未获授权的调用不会执行。`/skill`、`/feedback` 与模型 `use_skill` 的完整说明见 [Skill 使用指南](skills.md)。

## 开发验证与构建

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -m "not live"
uv build
```

默认测试只使用合成事件，不联网、不读取真实密钥。`dist/` 中的 wheel 与 source distribution 是构建输出。
## Session persistence

`hammer-code` creates a local session under `.hammer-code/sessions/`. Use `--list-sessions` to
list valid saved sessions without initializing a model client, or `--resume <id|latest>` to resume
a compatible session. Sessions with a `last_active` timestamp strictly older than 30 days are
deleted at startup and cannot be recovered by Hammer Code; copy a session directory manually before
that boundary if it must be retained.

Project-local `hammer-code.md` may provide instructions and bounded relative `@include(...)` files.
Memory and project instructions are non-authoritative context: they never grant tool permissions or
override current user instructions and verified workspace facts.

Session 切换和退出会先 drain 已排队的后台维护任务；再次中断时会取消尚未提交的维护。Session JSONL 不保存 Skill 候选、自动加载正文或 Evolution 中间输出。

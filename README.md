# Hammer Code

Hammer Code 是面向个人开发者的轻量级、协议中立 CLI 编程助手。首版实现了 OpenAI Responses、OpenAI Chat Completions（含兼容服务）和 Anthropic Messages 的异步流式通信，并把供应商事件隔离在适配器内部。

当前版本提供受权限控制的本地文件、搜索和 PowerShell 工具，以及 MCP Client 接入。所有工具调用仍须通过本地参数校验、公开状态检查和既有权限策略；模型提示词本身不授予任何执行权限。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。Windows PowerShell 中：

```powershell
python -m uv sync --all-groups
Copy-Item .hammer-code\config.example.toml .hammer-code\config.toml
```

不要提交 `.hammer-code/config.toml`；它已被忽略。配置只声明环境变量名，不能包含明文 API key。

## 配置与安全

从示例中选择一个 profile，并通过环境变量提供密钥：

```powershell
$env:OPENAI_API_KEY = "…"
python -m uv run hammer-code
```

可用协议是 `openai_responses`、`openai_chat_completions` 和 `anthropic_messages`。默认会从当前目录向上查找最近的 `.hammer-code/config.toml`，最多到 Git 根；也可明确指定：

```powershell
python -m uv run hammer-code --config .hammer-code/config.toml --profile openai
```

官方 OpenAI、Anthropic 和 DeepSeek HTTPS origin 可直接使用。任何自定义 endpoint 都会先要求本次进程确认，随后才读取密钥或创建客户端。远程 HTTP endpoint 被拒绝；loopback HTTP 仍须确认。

### MCP

在配置中用 `[[mcp]]` 登记 MCP Server。`stdio` Server 只得到 SDK 的安全默认环境和 `env` 中显式映射的变量；`env` 的值是当前进程环境变量名，而不是密钥。`streamable_http` 的远程 endpoint 必须使用 HTTPS；`localhost`、`127.0.0.1` 和 `::1` 可使用 HTTP。示例见 `.hammer-code/config.example.toml`。

需要标准 Bearer 认证的 Streamable HTTP MCP 使用扁平的 `bearer_token_env` 字段。该字段保存的是当前进程的环境变量名，连接时才会读取其值并发送 `Authorization: Bearer`；不要把 token 写入 TOML、URL 或 `env` 映射。PowerShell 示例：

```powershell
$env:EXAMPLE_MCP_TOKEN = "…"
```

```toml
[[mcp]]
name = "example-http"
description = "Example authenticated MCP"
transport = "streamable_http"
endpoint = "https://mcp.example.test/mcp"
bearer_token_env = "EXAMPLE_MCP_TOKEN"
```

`env = { bearer_token_env_var = "EXAMPLE_MCP_TOKEN" }` 不是 HTTP MCP 的有效配置，必须迁移为上面的 `bearer_token_env`。目前不支持任意 headers、OAuth、Basic、mTLS、proxy、cookie 或 token 持久化。

### Hooks

可选的项目本地 Hook 位于 `.hammer-code/hooks.toml`。可从
`.hammer-code/hooks.example.toml` 复制起步；真实配置已被 Git 忽略。每个 Hook 绑定一个确定的
生命周期事件，可按声明顺序执行 `command`、一次性 `prompt` 或匿名 `http` Action。严格 TOML
校验会拒绝未知字段、重复 id、非法条件、认证类 HTTP header 以及 `agent` Action。

Hook command 在展开占位符后，以既有 `shell` 身份经过完整 PermissionService；它不拥有独立或更宽
的权限。Prompt 只注入下一次模型请求（MCP 后、Skill 前），不会写入 Session 或对话历史。
HTTP Action 不经工具审批，但只能使用匿名 HTTP(S)、有超时和有界响应，并且不能配置认证、Cookie、
URL userinfo 或 Secret 引用。`pre_tool_use` Hook 可声明 `reject = true`，仅拒绝当前 Tool Call。

Hook 的 `once`、待注入 Prompt 和后台任务全部属于单个 PrimaryAgent 实例；新建、恢复或切换 Session
都会获得新的状态。删除或重命名 `hooks.toml` 后重启即可停用 Hook；不支持热重载。

MCP Server 在 CLI 启动后按配置顺序后台连接，单个连接失败不会阻塞其他 Server。配置名称、描述和加载状态会进入下一次模型请求的 MCP 提示片段；MCP 工具默认不进入模型工具列表。模型应先调用始终公开的 `toolSearch`，其找到的最多五个工具会从下一次 Agent Loop 请求开始以完整 schema 公开。未发现的工具即使名称被猜中也不能执行；`/clear` 会清除本次会话的发现状态。

## 交互

启动后，外层交互循环只处理输入、Slash Command 与 Session 生命周期；单个普通轮次由当前
Primary Agent 执行。可用的本地命令可通过 `/help` 动态查看，包括：`/status`、`/usage`、
`/clear`、`/compact`、`/permission`、`/session`、`/memory`、`/skill`、`/feedback` 与 `/exit`（`/quit` 是 `/exit` 的
别名）。命令不会进入对话历史或发送给模型。

`/session list|new|resume <id|latest>|delete <id>` 只在同一启动 profile 和协议内切换；新
Session 会立即成为当前会话，旧 Session 只在后台完成已经排队的 Memory 维护。删除会再次
确认，且拒绝当前或 draining Session；删除不可恢复。`/memory list [category]` 与
`/memory read <category> <relative-path>` 仅提供经校验的只读查看。

`/clear` 先等待当前 Session 已有的 Memory 维护完成，再清空当前会话的历史、usage、MCP
已发现工具、恢复线索和该 Session 的临时结果。`/compact` 只能在轮次之间执行；它保留唯一的
强制 Memory flush 入口，并复用自动压缩管线。没有可压缩历史时会提示 `Nothing to compact.`。
`/exit` 或 EOF 停止接收输入、排空已经启动或排队的 Memory 工作并持久化 Session；它不会因
不足五轮而新建 Memory 提取。此期间再次 Ctrl+C 会取消剩余维护并快速收尾。

上下文预算是本地 UTF-8 字节估算，不是供应商 usage。每次普通模型请求前都会检查预算；
达到触发点时，系统只摘要已提交历史，当前 user 输入及当前工具轮次始终原样保留在最终请求尾部。
摘要失败最多重试三次，之后保留会话并安全报错，不会自动清空历史。超大工具结果的临时
捕获位于 `.hammer-code/tmp/<session-id>/tool-results/`，可在成功摘要后定向清理；`/clear`
会同时清除内存历史、usage、MCP 已发现工具、恢复线索和当前会话临时结果。

### Skill

Skill 默认从项目内 `.hammer-code/skill/project/<folder>/SKILL.md` 与
`.hammer-code/skill/user/<folder>/SKILL.md` 的直接子目录发现。project 同名 Skill 遮蔽 user
版本；模型只看到候选元数据，必须调用 `use_skill` 才会加载完整正文。`/skill <name> [arguments]`
使用相同的受限解析与渲染规则，`/feedback <name|scope:name> <feedback>` 提交对特定 Skill 的
维护反馈。Skill 不能授予工具权限，所有实际调用继续经过既有权限服务。

`[skill.evolution]` 默认关闭。开启后，只有 `accept_edits` 或 `unattended` 权限模式允许后台维护
候选；当前目录中的 Skill state、history 与恢复 journal 均为本地数据，已被 Git 忽略。不要在
Skill 或配置中保存凭证。关闭 evolution 或切换至 `default`/`strict` 会阻止新的自动写入；恢复时
保留 history/prune 目录，先备份当前目标再手工恢复完整目录，代码回滚也不应删除这些本地数据。

```powershell
python -m uv run python -m hammer_code --help
python -m uv run hammer-code --help
```

## 验证与构建

所有自动化测试均为合成事件，不会联网，也不需要真实密钥：

```powershell
python -m uv run ruff format --check .
python -m uv run ruff check .
python -m uv run pyright
python -m uv run pytest -m "not live"
python -m uv build
```

## 当前范围

已实现：统一消息/事件模型、持久化本地 Session、项目指令与四类项目 Memory、按 request
快照的 Token usage、严格配置与 endpoint 信任检查、三协议独立流式适配器、受权限控制的
本地工具、MCP stdio/Streamable HTTP Client、延迟工具发现、上下文压缩、Primary Agent、
运行中 Session 切换与 Rich CLI。

未实现：MCP Resources、Prompts、Sampling、Elicitation、SSE、工具列表订阅、自动重试/健康检查、
通用 Subagent、Agent Team 和跨 profile/protocol 的运行中切换。

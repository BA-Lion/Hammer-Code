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

MCP Server 在 CLI 启动后按配置顺序后台连接，单个连接失败不会阻塞其他 Server。配置名称、描述和加载状态会进入下一次模型请求的 MCP 提示片段；MCP 工具默认不进入模型工具列表。模型应先调用始终公开的 `toolSearch`，其找到的最多五个工具会从下一次 Agent Loop 请求开始以完整 schema 公开。未发现的工具即使名称被猜中也不能执行；`/clear` 会清除本次会话的发现状态。

## 交互

启动后支持本地命令：`/help`、`/clear`、`/compact`、`/usage` 和 `/exit`。`/compact`
只能在轮次之间执行：它复用自动压缩的管线，但不受自动阈值限制，成功后只展示估算
节省量，不会发起普通模型请求。没有可压缩历史时会提示 `Nothing to compact.`。

上下文预算是本地 UTF-8 字节估算，不是供应商 usage。每次普通模型请求前都会检查预算；
达到触发点时，系统只摘要已提交历史，当前 user 输入及当前工具轮次始终原样保留在最终请求尾部。
摘要失败最多重试三次，之后保留会话并安全报错，不会自动清空历史。超大工具结果的临时
捕获位于 `.hammer-code/tmp/<session-id>/tool-results/`，可在成功摘要后定向清理；`/clear`
会同时清除内存历史、usage、MCP 已发现工具、恢复线索和当前会话临时结果。

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

已实现：统一消息/事件模型、内存会话事务、按 request 快照的 Token usage、严格配置与 endpoint 信任检查、三协议独立流式适配器、受权限控制的本地工具、MCP stdio/Streamable HTTP Client、延迟工具发现与不可变 ContextWindow，以及 Rich CLI。

未实现：MCP Resources、Prompts、Sampling、Elicitation、SSE、工具列表订阅、自动重试/健康检查、Memory、Skill、Subagent 和持久化会话。

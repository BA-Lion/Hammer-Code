# Hammer Code

Hammer Code 是面向个人开发者的轻量级、协议中立 CLI 编程助手。首版实现了 OpenAI Responses、OpenAI Chat Completions（含兼容服务）和 Anthropic Messages 的异步流式通信，并把供应商事件隔离在适配器内部。

当前版本只提供对话循环：它不会执行工具、命令或文件修改，也不会声称已完成这些操作。

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

## 交互

启动后支持本地命令：`/help`、`/clear`、`/usage` 和 `/exit`。`/clear` 只清除内存中的历史和 usage；对话不会写入磁盘。生成期间取消会回滚该轮消息，但会保留已经收到的 usage 快照。

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

已实现：统一消息/事件模型、内存会话事务、按 request 快照的 Token usage、严格配置与 endpoint 信任检查、三协议独立流式适配器，以及 Rich CLI。

未实现：工具执行、MCP、权限、Memory、Skill、Subagent、持久化会话和上下文裁剪。这些能力没有创建空壳模块。

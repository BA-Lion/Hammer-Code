# Hammer Code 配置指南

Hammer Code 使用项目级 TOML 配置。运行时配置文件是 `.hammer-code/config.toml`；仓库提供无密钥示例 `.hammer-code/config.example.toml`。

```powershell
Copy-Item .hammer-code\config.example.toml .hammer-code\config.toml
```

配置文件可保存模型、协议和 endpoint，但**不得**保存 `api_key`、访问令牌或其他密钥。未知字段会被拒绝，避免拼写错误或明文密钥被静默忽略。

## 完整结构

```toml
default_profile = "openai"

[ui]
show_reasoning = false

[profiles.openai]
protocol = "openai_responses"
model = "gpt-5-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
max_output_tokens = 4096
timeout_seconds = 60
max_retries = 0
```

`default_profile` 必须引用已有的 `[profiles.<名称>]`。profile 名称不可为空。

## 通用 profile 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `protocol` | 是 | `openai_responses`、`openai_chat_completions` 或 `anthropic_messages`。 |
| `model` | 是 | 提供商或兼容服务支持的模型标识。 |
| `base_url` | 是 | 完整的 HTTP(S) API 基地址。 |
| `api_key_env` | 是 | 保存密钥的环境变量名，而不是密钥值。 |
| `max_output_tokens` | 是 | 正整数，单次请求的最大输出 Token。 |
| `timeout_seconds` | 是 | 正数，SDK 请求超时秒数。 |
| `max_retries` | 是 | 首版必须为 `0`；流开始后不会自动重试。 |

设置密钥时，在当前 PowerShell 会话写入与 `api_key_env` 对应的变量：

```powershell
$env:OPENAI_API_KEY = "你的密钥"
```

不要把这条命令或密钥保存到仓库脚本、TOML 文件、日志或截图中。

## 三种协议

### OpenAI Responses

```toml
[profiles.openai]
protocol = "openai_responses"
model = "gpt-5-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
max_output_tokens = 4096
timeout_seconds = 60
max_retries = 0
reasoning_effort = "medium" # 可选：low、medium、high
reasoning_summary = "auto"  # 可选：auto、concise、detailed
```

此协议始终使用流式请求并显式关闭服务端存储；本地内存会话提供完整历史，不使用 `previous_response_id`。

### OpenAI Chat Completions / 兼容服务

```toml
[profiles.openai_compatible]
protocol = "openai_chat_completions"
model = "your-model"
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"
max_output_tokens = 4096
timeout_seconds = 60
max_retries = 0
stream_include_usage = true
```

`stream_include_usage = true` 请求供应商在流末尾返回 usage。若某个兼容服务不支持该选项，请改为 `false`；此时 usage 会标为 `unavailable`，而非显示为零。

### Anthropic Messages

```toml
[profiles.anthropic]
protocol = "anthropic_messages"
model = "claude-sonnet-4-20250514"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
max_output_tokens = 4096
timeout_seconds = 60
max_retries = 0
thinking_mode = "disabled"
```

`thinking_mode` 可为 `disabled`、`enabled` 或 `adaptive`。`thinking_budget` 只能和 `thinking_mode = "enabled"` 一起使用；`effort` 只允许用于 enabled/adaptive 模式。

## Endpoint 信任与网络安全

- 官方 OpenAI、Anthropic 和 DeepSeek 的 HTTPS origin 会直接接受。
- 其他 HTTPS endpoint 会在每次启动时要求确认；确认发生在读取环境变量密钥和构造 SDK 客户端之前。
- 远程 HTTP endpoint 一律拒绝。仅 `localhost`、`127.0.0.1` 和 `::1` 可使用 HTTP，且仍需要本次确认。
- 配置中不存在 `trusted = true` 之类的绕过字段。

自定义 endpoint 仅应在你信任该服务及网络路径时使用；该服务将接收所选 profile 的模型请求与历史上下文。

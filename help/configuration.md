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

## Subagent

预定义 Subagent 从 `.hammer-code/subagents/*.md` 读取；目录不存在时是正常的空 catalog，不会阻止动态 Subagent。只允许当前目录的直接 `.md` 文件，且不会自动创建任何定义。

```toml
[subagent]
default_max_iterations = 20 # 1–50
max_background_tasks = 4    # 1–16
```

这两个数值分别限制单次 Subagent 的模型/工具循环，以及每个会话中运行、等待审批或尚未写入结果的后台任务总数。它们不会授予工具权限，也不会启用 worktree、Agent Team 或跨进程恢复。定义格式、调用方式、审批和后台结果说明见 [Subagent 使用指南](subagents.md)。

## Skill 与自进化

Skill 相关目录位于项目内 `.hammer-code/skill/`，并被 Git 忽略。省略 `[skill]` 时，代码默认启用本地 Skill 发现/调用、但默认关闭 Evolution；不存在 Skill 目录等同于空 catalog，不会在启动时自动创建第一个 Skill。

```toml
[skill]
enabled = true                 # false 会关闭模型、/skill 本地调用和候选检索
retrieval_top_k = 3            # 1–20；每轮最多展示的 metadata 候选数
retrieval_relative_floor = 0.60
retrieval_query_coverage = 0.20

[skill.evolution]
enabled = false                # 建议先保持关闭；启用后才允许后台提炼/维护
max_history_messages = 8
max_input_tokens = 4000
max_maintenance_input_tokens = 64000
merge_candidate_top_k = 10
forced_merge_score = 0.90
forced_merge_margin = 0.10
max_body_chars = 32000
max_corrections = 2
prune_unused_days = 90
prune_min_retrieve = 20
prune_min_relevant = 10
prune_relevant_used_ratio = 5.0
```

Evolution 还要求运行时权限为 `accept_edits` 或 `unattended`；`default` 和 `strict` 不会排队后台写入，也不会显示后台审批。请以 `.hammer-code/config.example.toml` 的实际 `enabled` 值为复制配置后的最终准则：该示例可被项目维护者调整，而代码默认值仍为 `false`。完整目录、调用和恢复说明见 [Skill 使用指南](skills.md)。

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

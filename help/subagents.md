# Hammer Code Subagent 使用指南

Subagent 是由主 Agent 在当前会话内按需发起的受限任务，不是新的终端程序，也不会创建 Git worktree、Agent Team 或可恢复的独立进程。它适合把边界清晰的检索、检查或整理工作交给独立模型循环完成。

你不需要、也不能在终端直接输入 `run_subagent` 或 `subagent_task`；这是提供给模型的受控工具。当任务适合拆分时，模型会使用目录中已定义的 Subagent，或在当前请求内创建一次性的 dynamic Subagent。你可以在普通对话中明确提出意图，例如“请用代码审查 Subagent 检查这次修改”或“把这项检查放到后台，并在完成后告诉我结果”。

## 快速开始

1. 复制并配置 `.hammer-code/config.toml`，保留或按需要调整 `[subagent]`。
2. 创建目录 `.hammer-code/subagents/`。
3. 在该目录创建一个名称与文件名一致的 Markdown 定义，例如 `reviewer.md`。
4. 重启 Hammer Code；第一次加载发现定义无效时会拒绝启动，避免在未知配置下运行。
5. 在对话中说明何时使用该 Subagent。模型在下一次主请求中会看到名称、用途和默认执行模式。

最小示例：

```text
.hammer-code/
  subagents/
    reviewer.md
```

```markdown
---
name: reviewer
description: Review a bounded code change and report actionable findings.
when-to-use: Use before merging a focused code change.
context: fork
execution: inline
allowed-tools: [read_file,grep,glob]
disallowed-tools: [shell]
max-iterations: 12
---
Review only the user-provided change scope. Report findings by severity, with
file paths and concise remediation steps. Do not make edits unless the task
explicitly asks for them.
```

`name` 必须与文件名（去掉 `.md`）完全一致；上例必须保存为 `reviewer.md`。目录中的定义不应提交包含密钥、个人资料或其他敏感内容。

## 定义字段

每个定义都是 UTF-8 的普通 `.md` 文件，位于 `.hammer-code/subagents/` 的直接子层。不能使用子目录、符号链接/reparse point、YAML 嵌套、重复字段或未知字段。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | 小写字母、数字和单个连字符组成，1–64 个字符；必须匹配文件名。 |
| `description` | 是 | 单行描述，最多 500 字符；供主 Agent 选择时展示。 |
| `when-to-use` | 否 | 单行使用条件，最多 1,000 字符。 |
| `context` | 否 | `isolated`（默认）或 `fork`。 |
| `execution` | 否 | `inline`（默认）或 `background`。 |
| `allowed-tools` | 否 | 一行列表，例如 `[read_file,grep]`；只保留该列表与当前已公开工具的交集。 |
| `disallowed-tools` | 否 | 一行列表，例如 `[shell]`；优先从可用工具中移除。 |
| `max-iterations` | 否 | 1–50；未填写时使用 `[subagent].default_max_iterations`。 |

工具列表中的名称不能重复、不能含空白或引号。`allowed-tools` 与 `disallowed-tools` 不能包含同一个工具。没有 `allowed-tools` 时，子 Agent 只能使用当前主请求已公开的工具，再扣除 `disallowed-tools`。

## 上下文与执行方式

| 选项 | 行为 |
| --- | --- |
| `isolated` | 只接收系统/项目/MCP 基线、定义正文和当前任务；不继承主对话消息。适合独立检查。 |
| `fork` | 继承触发时刻的主请求消息快照；不会包含触发该调用的未配对工具调用。适合需要理解当前讨论的工作。 |
| `inline` | 主 Agent 等待完成，并把最终可见文本作为当前工具调用的唯一结果。 |
| `background` | 立即返回短任务 ID；最终结果在后续固定检查点汇总写入当前 Session。 |

动态 Subagent 没有文件定义：主 Agent 会在当前调用中提供任务说明，默认使用 `isolated + inline` 和全局迭代上限。它不会写入 `.hammer-code/subagents/`，也不会成为下次会话可见的定义。

## 后台任务和结果

后台启动后，模型会收到类似 `Subagent task started: ab12cd34` 的回执。你可以在同一会话中让模型列出、查看或取消该任务；短 ID 至少要提供 8 个字符且在当前会话中唯一。

完成、失败或取消的后台任务不会伪造成第二个工具结果。主 Agent 会在主轮次前后、会话切换或退出收尾时，把当时已完成的全部任务作为一个普通的 Session 轮次记录；每批只追加一对 USER/ASSISTANT 消息。因此结果可能在下一次交互或收尾时出现，而不是在后台完成的瞬间插入当前回答。

未被该结果批次取走的终态任务仍占用后台容量。查看任务不会消费结果；完成结果被写入 Session 后才释放容量。使用 `/clear` 会取消并丢弃当前会话尚未写入的 Subagent 终态结果。

## 权限与安全边界

- 子 Agent 使用当前 Session 的同一权限模式、PermissionChecker 和持久化规则；它不会获得额外权限。
- 子 Agent 不能调用 `run_subagent`、`subagent_task`、`use_skill` 或 `toolSearch`，因此不能递归派生任务或加载 Skill。
- 需要人工批准的后台工具调用会显示 `[subagent=<name> task=<short-id>]`，并在等待期间显示为 `waiting_approval`。终端输入由同一个队列串行化，不会与主会话的输入竞争。
- 拒绝一次工具调用只会把该调用作为错误结果返回给子 Agent；不会自动把整个子任务升级权限或改为无人值守执行。
- 子 Agent 的可见输出被视为不可信运行时数据。最终汇总会标明来源、任务 ID、Agent 名称和状态；请像审阅外部工具输出一样核对它。

## 配置上限

```toml
[subagent]
default_max_iterations = 20 # 每次运行的默认模型/工具循环上限，范围 1–50
max_background_tasks = 4    # 每个 PrimaryAgent 的后台任务容量，范围 1–16
```

Subagent 的单次任务最多 16,000 字符，定义正文和 dynamic 提示词最多 32,000 字符，最终可见结果最多 32,000 字符。超过上限、未公开工具、无效定义、未知名称、后台容量已满或异常流都会安全失败，而不是绕过限制继续执行。

## 更新与排障

- 每次主模型请求都会重新检查目录。目录不存在等同于空 catalog，仍允许 dynamic Subagent。
- 已成功加载后，如果后续修改使目录无效，Hammer Code 会发出一次警告并继续使用最后一个完整有效 catalog；修复文件后会在下次请求恢复加载。
- 若程序启动即因 catalog 失败，请检查文件是否为直接子层 `.md`、文件名和 `name` 是否一致、front matter 是否以首行 `---` 开始且闭合，以及工具列表是否采用无引号的一行列表语法。
- 若模型未选择你的定义，请让任务描述与 `description`、`when-to-use` 更具体匹配；定义正文不会直接暴露给主模型，只有被选择后才会加载。

另见 [配置指南](configuration.md)、[使用命令](commands.md) 和 [Skill 使用指南](skills.md)。

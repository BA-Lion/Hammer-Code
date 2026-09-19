# Hammer Code Skill 使用指南

Skill 是项目内、可逐步加载的行为说明。它低于系统规则、当前用户指令和 PermissionService：Skill 内容、检索候选或自进化输出都不能授予工具、文件路径或权限。

## 何时会出现目录

首次启动不会创建 `.hammer-code/skill/`。目录不存在时是正常的空 catalog；因此 `/skill` 和模型 `use_skill` 暂时没有可调用对象。

你可以手动创建第一个 Skill，或在显式启用 Evolution 后由后台维护成功生成第一个 `generated` Skill。测试和开发期间不要把真实 Skill、sessions 或 memory 数据纳入版本控制。

## 手动创建第一个 Skill

每个 Skill 必须位于 project 或 user 根的直接子目录中，文件名固定为 `SKILL.md`：

```text
.hammer-code/
  skill/
    project/
      release-review/
        SKILL.md
    user/
      personal-review/
        SKILL.md
```

`project` 与 `user` 同名时，未限定名称优先使用 project；本地命令可用 `project:name` 或 `user:name` 精确选择。目录、`SKILL.md` 都不能是 symlink/reparse point；无效目录或 front matter 会使该次 catalog 安全关闭，而不是继续使用旧缓存。

最小可用示例：

```markdown
---
name: release-review
description: "Review a release candidate before deployment"
when-to-use: "before releasing a production change"
arguments-required: true
argument-hint: "provide the change or release scope"
---
Review {{arguments}}.

1. Check the scope and rollback plan.
2. Run the relevant offline validation.
3. Report blockers before deployment.
```

名称只允许小写字母、数字与单个短横线；正文不能为空。可选字段包括 `version`、UTC 时间、`source`、调用开关、`context: inline|fork` 与 `allowed-tools`。解析器只接受受限的单行 front matter，不支持 YAML 嵌套、锚点、注释尾随或多文档。

正文只会替换精确的 `{{arguments}}` 和 `{{skill_dir}}`，替换结果不会二次展开。

## 调用方式

```text
/skill release-review current release diff
/skill project:release-review current release diff
```

模型只会收到 name、description 与 when-to-use 的候选元数据；需要完整正文时，它必须调用 `use_skill`。inline Skill 的正文必须完整装入 Tool Result 限额，超限会明确报错而不会截断或提供临时文件路径。

`context: fork` 会启动独立的受限执行循环：它不继承父对话、Memory 或恢复线索；只能使用启动时已经公开的工具与 `allowed-tools` 的交集，且始终排除 `use_skill`、`tool_search`。每个实际工具调用仍进行正常权限检查。

## 自进化与反馈

默认关闭 Evolution。启用 `[skill.evolution].enabled = true` 后，还必须把权限模式设为 `accept_edits` 或 `unattended`，普通顶层输入才会创建独立维护项。Extractor 最多提出一个候选，Maintenance 只会输出 add、merge 或 discard；程序会复核输出、目标、权限、版本和 preimage 后才写入。

对已有 Skill 提供定向反馈：

```text
/feedback project:release-review 在提交前增加变更冻结窗口检查
```

该命令不会作为普通对话消息发送；当 Evolution 未启用或权限不足时，会返回可操作错误。

## 统计、恢复与淘汰

`.hammer-code/skill/state/usage.json` 记录按 `scope:name` 区分的 retrieve、relevant、used 计数及当前版本计数。自动 merge 会保留 source、保存旧完整目录到 `history/`、执行 patch 版本递增并重置当前版本计数。

只有 `source: generated` 的 Skill 可能在满足配置阈值时移动到 `prune/`；不会硬删除。journal 遗留时，启动会尝试前滚已提交事务；歧义 journal 会隔离并关闭自动写入。

恢复前先关闭 Evolution 或切换到 `default`/`strict`，备份当前目标目录，再确认 scope/name 没有冲突后手工从 `history/` 或 `prune/` 恢复完整目录。代码回滚不得删除 `.hammer-code/skill/`。

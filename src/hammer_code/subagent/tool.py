"""Model-facing Subagent invocation and task-inspection tool contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hammer_code.subagent.models import SubagentContext, SubagentExecution
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)


class RunSubagentArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task: str = Field(min_length=1, max_length=16000)
    name: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*"
    )
    prompt: str | None = Field(default=None, max_length=32000)
    context: SubagentContext | None = None
    execution: SubagentExecution | None = None

    @model_validator(mode="after")
    def _shape(self) -> RunSubagentArguments:
        if not self.task.strip():
            raise ValueError("task must not be blank")
        has_name = self.name is not None
        has_prompt = self.prompt is not None and bool(self.prompt.strip())
        if has_name == has_prompt:
            raise ValueError("exactly one of name or non-blank prompt is required")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        return self


class SubagentTaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["list", "get", "cancel"]
    task_id: str | None = Field(default=None, min_length=1, max_length=36)

    @model_validator(mode="after")
    def _shape(self) -> SubagentTaskArguments:
        if (self.action == "list") != (self.task_id is None):
            raise ValueError("list must not include task_id; get and cancel require task_id")
        return self


class RunSubagentTool(Tool):
    name = "run_subagent"
    description = "Run one predefined or dynamic bounded Subagent for a complete task."
    input_model = RunSubagentArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.subagent_invoker is None:
            return ToolExecutionResult("Error: Subagent invocation is unavailable.", True)
        if not isinstance(arguments, RunSubagentArguments):
            return ToolExecutionResult("Error: Subagent arguments are invalid.", True)
        return await context.subagent_invoker.invoke(arguments)


class SubagentTaskTool(Tool):
    name = "subagent_task"
    description = "List, inspect, or cancel Subagent tasks created in the current Session."
    input_model = SubagentTaskArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.subagent_invoker is None:
            return ToolExecutionResult("Error: Subagent task management is unavailable.", True)
        if not isinstance(arguments, SubagentTaskArguments):
            return ToolExecutionResult("Error: Subagent task arguments are invalid.", True)
        return await context.subagent_invoker.task(arguments)

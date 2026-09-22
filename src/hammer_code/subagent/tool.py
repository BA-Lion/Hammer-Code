"""Model-facing Subagent invocation contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hammer_code.subagent.models import SubagentContext, SubagentExecution, SubagentWorkspace
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
    workspace: SubagentWorkspace | None = None

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


class RunSubagentTool(Tool):
    name = "run_subagent"
    description = (
        "Run one bounded Subagent. Use inline when the current answer needs the result; "
        "multiple inline calls in one tool batch run concurrently. Use background only to "
        "start detached work whose result will be delivered on a later user turn; do not wait "
        "or poll for it in the current turn."
    )
    input_model = RunSubagentArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.subagent_invoker is None:
            return ToolExecutionResult("Error: Subagent invocation is unavailable.", True)
        if not isinstance(arguments, RunSubagentArguments):
            return ToolExecutionResult("Error: Subagent arguments are invalid.", True)
        return await context.subagent_invoker.invoke(arguments)

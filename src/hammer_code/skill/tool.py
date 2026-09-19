"""The narrow model-facing tool for progressive Skill loading."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)


class UseSkillArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=64, pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*")
    arguments: str = ""


class UseSkillTool(Tool):
    name = "use_skill"
    description = (
        "Load the complete instructions for a suggested Skill by exact name. "
        "For a fork Skill, include the complete task in arguments."
    )
    input_model = UseSkillArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.skill_invoker is None:
            return ToolExecutionResult("Error: Skill invocation is unavailable.", True)
        if not isinstance(arguments, UseSkillArguments):
            return ToolExecutionResult("Error: Skill arguments are invalid.", True)
        return await context.skill_invoker.invoke(arguments.name, arguments.arguments)

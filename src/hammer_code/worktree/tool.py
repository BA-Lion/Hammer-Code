"""Primary-only model tools for bounded worktree inspection and resolution."""
# ruff: noqa: E501

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)


class InspectSubagentWorktreeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    detail: Literal["summary", "diff", "merge_inputs"] = "summary"
    paths: tuple[str, ...] = ()

    @field_validator("task_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        return str(UUID(value))

    @field_validator("paths")
    @classmethod
    def _paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 20 or any(not _relative(path) for path in value):
            raise ValueError("paths must be normalized repository-relative paths")
        return value


class ResolveEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    expected_current_hash: str = Field(pattern=r"[0-9a-f]{64}|missing")
    action: Literal["write", "delete"]
    content: str | None = Field(default=None, max_length=4 * 1024 * 1024)

    @model_validator(mode="after")
    def _content(self) -> ResolveEdit:
        if not _relative(self.path) or (self.action == "write") != (self.content is not None):
            raise ValueError("merge edit is invalid")
        return self


class ResolveSubagentWorktreeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    resolution: Literal["integrate", "agent_merge", "discard"]
    inspection_id: str | None = None
    edits: tuple[ResolveEdit, ...] = ()

    @field_validator("task_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        return str(UUID(value))

    @model_validator(mode="after")
    def _shape(self) -> ResolveSubagentWorktreeArguments:
        if self.resolution == "agent_merge" and (not self.inspection_id or not self.edits):
            raise ValueError("agent_merge requires an inspection id and complete edits")
        if self.resolution != "agent_merge" and (self.inspection_id or self.edits):
            raise ValueError("only agent_merge accepts inspection data")
        return self


class InspectSubagentWorktreeTool(Tool):
    name = "inspect_subagent_worktree"
    description = "Inspect one current Agent's pending isolated Subagent worktree draft."
    input_model = InspectSubagentWorktreeArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.worktree_invoker is None or not isinstance(
            arguments, InspectSubagentWorktreeArguments
        ):
            return ToolExecutionResult("Error: Subagent worktree inspection is unavailable.", True)
        return await context.worktree_invoker.inspect(arguments)


class ResolveSubagentWorktreeTool(Tool):
    name = "resolve_subagent_worktree"
    description = (
        "Integrate, semantically merge, or discard one current Agent's pending worktree draft."
    )
    input_model = ResolveSubagentWorktreeArguments
    category = ToolCategory.WRITE
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.worktree_invoker is None or not isinstance(
            arguments, ResolveSubagentWorktreeArguments
        ):
            return ToolExecutionResult("Error: Subagent worktree resolution is unavailable.", True)
        return await context.worktree_invoker.resolve(arguments)


def _relative(value: str) -> bool:
    return (
        bool(value)
        and "\\" not in value
        and not value.startswith("/")
        and ":" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )

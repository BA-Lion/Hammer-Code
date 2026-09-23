"""Stable tool contract, independent of any model SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from hammer_code.domain.events import ToolDefinition

if TYPE_CHECKING:
    from hammer_code.agent_team.tool import (
        AgentTeamPrimaryArguments,
        TeamRuntimeArguments,
    )
    from hammer_code.subagent.tool import RunSubagentArguments
    from hammer_code.worktree.tool import (
        InspectSubagentWorktreeArguments,
        ResolveSubagentWorktreeArguments,
    )


class ToolCategory(StrEnum):
    READ = "read"
    WRITE = "write"
    COMMAND = "command"


class ConcurrencyPolicy(StrEnum):
    PARALLEL_READ = "parallel_read"
    SERIAL = "serial"


class SkillInvocationPort(Protocol):
    """Per-Agent bridge used by the built-in ``use_skill`` tool only."""

    async def invoke(self, name: str, arguments: str) -> ToolExecutionResult: ...


class SubagentInvocationPort(Protocol):
    """Per-Agent bridge used by the Subagent tools during an active main turn."""

    async def invoke(self, arguments: RunSubagentArguments) -> ToolExecutionResult: ...


class SubagentWorktreeInvocationPort(Protocol):
    """Primary-only bridge for inspecting and resolving its own worktree drafts."""

    async def inspect(self, arguments: InspectSubagentWorktreeArguments) -> ToolExecutionResult: ...

    async def resolve(self, arguments: ResolveSubagentWorktreeArguments) -> ToolExecutionResult: ...


class AgentTeamInvocationPort(Protocol):
    """Primary-only bridge for catalog, create, and run Team tools."""

    async def invoke(self, arguments: AgentTeamPrimaryArguments) -> ToolExecutionResult: ...


class TeamRuntimeInvocationPort(Protocol):
    """Assignment-bound bridge; no call accepts a filesystem or run path."""

    async def invoke(self, arguments: TeamRuntimeArguments) -> ToolExecutionResult: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    workspace_root: Path
    cwd: Path
    runtime_dir: Path
    sanitized_env: Mapping[str, str]
    skill_invoker: SkillInvocationPort | None = None
    subagent_invoker: SubagentInvocationPort | None = None
    worktree_invoker: SubagentWorktreeInvocationPort | None = None
    agent_team_invoker: AgentTeamInvocationPort | None = None
    team_runtime_invoker: TeamRuntimeInvocationPort | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    is_error: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


class Tool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    category: ToolCategory
    concurrency_policy: ConcurrencyPolicy
    should_defer: bool = False

    def definition(self) -> ToolDefinition:
        """Project the local Pydantic contract into a model-facing tool definition."""
        schema = _remove_titles(self.input_model.model_json_schema())
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"Tool {self.name} must have an object input schema")
        schema["additionalProperties"] = False
        return ToolDefinition(self.name, self.description, schema)

    @abstractmethod
    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult: ...


def _remove_titles(value: object) -> object:
    if isinstance(value, dict):
        return {key: _remove_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_remove_titles(item) for item in value]
    return value

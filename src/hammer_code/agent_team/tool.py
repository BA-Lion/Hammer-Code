"""Model contracts for the Primary, Leader, and Teammate AgentTeam tools."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hammer_code.subagent.models import SubagentContext, SubagentExecution, SubagentWorkspace
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.worktree.tool import (
    InspectSubagentWorktreeArguments,
    ResolveSubagentWorktreeArguments,
)


class LeaderIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*", max_length=64)
    description: str = Field(min_length=1, max_length=500)
    when_to_use: str | None = Field(default=None, max_length=1000)
    prompt: str = Field(min_length=1, max_length=32_000)
    context: SubagentContext = SubagentContext.ISOLATED
    execution: SubagentExecution = SubagentExecution.INLINE
    workspace: SubagentWorkspace = SubagentWorkspace.WORKTREE
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()

    @field_validator("description", "when_to_use")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or any(char in value for char in "\r\n\0")):
            raise ValueError("identity text must be a non-empty single line")
        return value.strip() if value is not None else None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("identity prompt must not be blank or contain NUL")
        return value

    @model_validator(mode="after")
    def _tools_do_not_overlap(self) -> LeaderIdentity:
        if set(self.allowed_tools or ()).intersection(self.disallowed_tools):
            raise ValueError("allowed_tools and disallowed_tools must not overlap")
        return self


class GetAgentTeamsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreateAgentTeamArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task: str = Field(min_length=1, max_length=16_000)
    description: str = Field(min_length=1, max_length=500)
    predefined_leader: str | None = Field(default=None, max_length=64)
    leader: LeaderIdentity | None = None

    @model_validator(mode="after")
    def _shape(self) -> CreateAgentTeamArguments:
        if (self.predefined_leader is None) == (self.leader is None):
            raise ValueError("exactly one of predefined_leader or leader is required")
        return self


class RunAgentTeamArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*", max_length=64)
    task: str = Field(min_length=1, max_length=16_000)
    context: SubagentContext | None = None
    execution: SubagentExecution | None = None
    workspace: SubagentWorkspace | None = None


AgentTeamPrimaryArguments: TypeAlias = (
    GetAgentTeamsArguments | CreateAgentTeamArguments | RunAgentTeamArguments
)


class CreateTeammateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    existing_template: str | None = Field(default=None, max_length=64)
    template: LeaderIdentity | None = None

    @model_validator(mode="after")
    def _shape(self) -> CreateTeammateArguments:
        if (self.existing_template is None) == (self.template is None):
            raise ValueError("exactly one of existing_template or template is required")
        return self


class TaskCreateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=8000)
    dependencies: tuple[str, ...] = ()


class TaskUpdateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    state: Literal["pending", "in_progress", "completed", "failed", "cancelled"] | None = None
    assignment_id: str | None = None


class GetAllTasksArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GetAllTeammatesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WaitForMessageArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SendMessageArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recipient_assignment_id: str
    body: str = Field(min_length=1, max_length=16_000)


class FinishTaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    summary: str = Field(min_length=1, max_length=8000)


class FinishTeamArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "failed", "cancelled"]
    summary: str = Field(min_length=1, max_length=8000)


TeamRuntimeArguments: TypeAlias = (
    CreateTeammateArguments
    | TaskCreateArguments
    | TaskUpdateArguments
    | GetAllTasksArguments
    | GetAllTeammatesArguments
    | SendMessageArguments
    | FinishTaskArguments
    | FinishTeamArguments
    | WaitForMessageArguments
    | InspectSubagentWorktreeArguments
    | ResolveSubagentWorktreeArguments
)


class _PrimaryTool(Tool):
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.agent_team_invoker is None:
            return ToolExecutionResult("Error: AgentTeam invocation is unavailable.", True)
        return await context.agent_team_invoker.invoke(arguments)  # type: ignore[arg-type]


class GetAgentTeamsTool(_PrimaryTool):
    name = "get_agent_teams"
    description = "List reusable AgentTeams by their Team name, description, and Leader identity."
    input_model = GetAgentTeamsArguments
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ


class CreateAgentTeamTool(_PrimaryTool):
    name = "create_agent_team"
    description = "Create a self-contained AgentTeam and immediately run its root task."
    input_model = CreateAgentTeamArguments
    category = ToolCategory.WRITE


class RunAgentTeamTool(_PrimaryTool):
    name = "run_agent_team"
    description = "Run an existing AgentTeam on a new root task."
    input_model = RunAgentTeamArguments


class _RuntimeTool(Tool):
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if context.team_runtime_invoker is None:
            return ToolExecutionResult("Error: AgentTeam runtime invocation is unavailable.", True)
        return await context.team_runtime_invoker.invoke(arguments)  # type: ignore[arg-type]


def _runtime_tool(
    name: str, description: str, input_model: type[BaseModel], *, write: bool = False
) -> type[Tool]:
    return type(
        name.title().replace("_", "") + "Tool",
        (_RuntimeTool,),
        {
            "name": name,
            "description": description,
            "input_model": input_model,
            "category": ToolCategory.WRITE if write else ToolCategory.READ,
        },
    )


CreateTeammateTool = _runtime_tool(
    "create_teammate",
    "Create or reuse a Teammate template for one Team task.",
    CreateTeammateArguments,
    write=True,
)
TaskCreateTool = _runtime_tool(
    "task_create", "Create a Team task; Leader only.", TaskCreateArguments
)
TaskUpdateTool = _runtime_tool(
    "task_update", "Update a Team task; Leader only.", TaskUpdateArguments
)
GetAllTasksTool = _runtime_tool(
    "get_all_tasks", "Read the current Team task board.", GetAllTasksArguments
)
GetAllTeammatesTool = _runtime_tool(
    "get_all_teammates", "Read current Team Assignment runtimes.", GetAllTeammatesArguments
)
SendMessageTool = _runtime_tool(
    "send_message", "Send one bounded message to an active Team Assignment.", SendMessageArguments
)
WaitForMessageTool = _runtime_tool(
    "wait_for_message", "Wait for a message addressed to this Assignment.", WaitForMessageArguments
)
FinishTaskTool = _runtime_tool(
    "finish_task", "Finish this Teammate Assignment.", FinishTaskArguments
)
FinishTeamTool = _runtime_tool("finish_team", "Finish this Leader's TeamRun.", FinishTeamArguments)
InspectAssignmentWorktreeTool = _runtime_tool(
    "inspect_assignment_worktree",
    "Inspect a Team Assignment child worktree.",
    InspectSubagentWorktreeArguments,
)
ResolveAssignmentWorktreeTool = _runtime_tool(
    "resolve_assignment_worktree",
    "Resolve one Team Assignment child worktree into the Team root.",
    ResolveSubagentWorktreeArguments,
    write=True,
)

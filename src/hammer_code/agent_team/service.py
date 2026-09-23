"""Per-PrimaryAgent AgentTeam orchestration and assignment-bound ports."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from hammer_code.agent_team.models import (
    AgentTeamCatalogSnapshot,
    AgentTeamDefinition,
    TeamAgentTemplate,
)
from hammer_code.agent_team.prompts import build_assignment_prompt, build_leader_prompt
from hammer_code.agent_team.runner import AgentTeamRunner
from hammer_code.agent_team.runtime import (
    AgentTeamRun,
    AgentTeamRuntimeStore,
    AssignmentPhase,
    AssignmentRuntime,
    TeamTaskState,
)
from hammer_code.agent_team.tool import (
    CreateAgentTeamArguments,
    CreateTeammateArguments,
    CreateTeammateTool,
    FinishTaskArguments,
    FinishTaskTool,
    FinishTeamArguments,
    FinishTeamTool,
    GetAgentTeamsArguments,
    GetAllTasksArguments,
    GetAllTasksTool,
    GetAllTeammatesArguments,
    GetAllTeammatesTool,
    InspectAssignmentWorktreeTool,
    ResolveAssignmentWorktreeTool,
    RunAgentTeamArguments,
    SendMessageArguments,
    SendMessageTool,
    TaskCreateArguments,
    TaskCreateTool,
    TaskUpdateArguments,
    TaskUpdateTool,
    TeamRuntimeArguments,
    WaitForMessageArguments,
    WaitForMessageTool,
)
from hammer_code.config import AgentTeamConfig, SubagentConfig
from hammer_code.permissions.models import PermissionRequest
from hammer_code.subagent.models import (
    ParentRequestSnapshot,
    SubagentContext,
    SubagentExecution,
    SubagentWorkspace,
)
from hammer_code.subagent.service import BackgroundTaskManager, SubagentService
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult
from hammer_code.tools.registry import ToolRegistry
from hammer_code.worktree.manager import WorktreeManager, WorktreeUnavailableError
from hammer_code.worktree.tool import (
    InspectSubagentWorktreeArguments,
    ResolveSubagentWorktreeArguments,
)


class AgentTeamService:
    """Session-owned coordinator. Long-lived definitions live only in Repository."""

    def __init__(
        self,
        repository,
        config: AgentTeamConfig,
        subagent_config: SubagentConfig,
        tasks: BackgroundTaskManager,
        runtime_store: AgentTeamRuntimeStore,
        worktree_manager: WorktreeManager | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.subagent_config = subagent_config
        self.tasks = tasks
        self.runtime_store = runtime_store
        self.worktree_manager = worktree_manager
        self._parent: ParentRequestSnapshot | None = None
        self._catalog: AgentTeamCatalogSnapshot | None = None
        self._runner: AgentTeamRunner | None = None
        self._registry: ToolRegistry | None = None
        self._context: ToolExecutionContext | None = None
        self._runs: dict[str, AgentTeamRun] = {}
        self._handles: dict[tuple[str, str], asyncio.Task[ToolExecutionResult]] = {}
        self._root_leases: dict[str, str] = {}

    def bind_runtime(
        self, runner: AgentTeamRunner, registry: ToolRegistry, context: ToolExecutionContext
    ) -> None:
        self._runner, self._registry, self._context = runner, registry, context

    def bind_request(
        self, parent: ParentRequestSnapshot, catalog: AgentTeamCatalogSnapshot
    ) -> None:
        self._parent, self._catalog = parent, catalog

    def clear_request(self) -> None:
        self._parent = self._catalog = None

    async def invoke(self, arguments) -> ToolExecutionResult:
        if not self.config.coordination_mode:
            return ToolExecutionResult("Error: AgentTeam coordination is disabled.", True)
        if isinstance(arguments, GetAgentTeamsArguments):
            catalog = self._catalog or await self.repository.snapshot_for_request()
            return ToolExecutionResult(json.dumps(catalog.catalog_views(), ensure_ascii=False))
        if isinstance(arguments, CreateAgentTeamArguments):
            try:
                leader = self._leader_from_create(arguments)
                definition = await self.repository.create_team(
                    arguments.description.strip(),
                    leader,
                    execution=(
                        arguments.leader.execution
                        if arguments.leader is not None
                        else self._predefined_execution(arguments.predefined_leader)
                    ),
                )
                return await self._launch(definition, arguments.task, None, None, None)
            except ValueError as exc:
                return ToolExecutionResult(f"Error: {exc}", True)
        if isinstance(arguments, RunAgentTeamArguments):
            catalog = self._catalog or await self.repository.snapshot_for_request()
            definition = catalog.definitions.get(arguments.name)
            if definition is None:
                return ToolExecutionResult("Error: Unknown AgentTeam.", True)
            return await self._launch(
                definition,
                arguments.task,
                arguments.context,
                arguments.execution,
                arguments.workspace,
            )
        return ToolExecutionResult("Error: AgentTeam arguments are invalid.", True)

    def _leader_from_create(self, arguments: CreateAgentTeamArguments) -> TeamAgentTemplate:
        if arguments.leader is not None:
            source = arguments.leader
            return TeamAgentTemplate(
                source.name,
                source.description,
                source.when_to_use,
                source.prompt,
                source.context,
                source.workspace,
                source.allowed_tools,
                source.disallowed_tools,
            )
        if self._parent is None or arguments.predefined_leader is None:
            raise ValueError("predefined AgentTeam Leader is unavailable")
        source = self._parent.catalog.definitions.get(arguments.predefined_leader)
        if source is None:
            raise ValueError("Unknown predefined Subagent Leader")
        return TeamAgentTemplate(
            source.name,
            source.description,
            source.when_to_use,
            source.prompt,
            source.context,
            source.workspace,
            source.allowed_tools,
            source.disallowed_tools,
        )

    def _predefined_execution(self, name: str | None) -> SubagentExecution:
        if self._parent is None or name is None:
            return SubagentExecution.INLINE
        definition = self._parent.catalog.definitions.get(name)
        return definition.execution if definition is not None else SubagentExecution.INLINE

    async def _launch(
        self,
        definition: AgentTeamDefinition,
        task: str,
        context: SubagentContext | None,
        execution: SubagentExecution | None,
        workspace: SubagentWorkspace | None,
    ) -> ToolExecutionResult:
        if self._runner is None or self._registry is None or self._context is None:
            return ToolExecutionResult(
                "Error: AgentTeam is unavailable outside an active turn.", True
            )
        leader = replace(
            definition.leader,
            context=context or definition.leader.context,
            workspace=workspace or definition.leader.workspace,
        )
        definition = replace(definition, leader=leader, execution=execution or definition.execution)
        run = AgentTeamRun(
            definition,
            task.strip(),
            self.runtime_store,
            self.config.max_active_assignments,
            self.config.max_team_tokens,
            definition.execution,
        )
        try:
            await run.initialize()
            await self._prepare_root_workspace(run)
        except (OSError, ValueError, WorktreeUnavailableError) as exc:
            await run.cleanup()
            return ToolExecutionResult(f"Error: AgentTeam run could not be created: {exc}", True)
        self._runs[run.id] = run
        handle = asyncio.create_task(self._run_leader(run))
        self._handles[(run.id, run.leader_assignment_id)] = handle
        if definition.execution is SubagentExecution.BACKGROUND:

            async def background_operation(_) -> ToolExecutionResult:
                result = await handle
                if result.is_error:
                    await self._cleanup_run(run.id)
                    return result
                payload = {
                    "team": definition.name,
                    "run_id": run.id,
                    "status": run.terminal,
                    "summary": run.summary,
                }
                await self._finalize_root(run, payload)
                await self._cleanup_run(run.id, keep_root=True)
                return ToolExecutionResult(json.dumps(payload, ensure_ascii=False))

            try:
                item = await self.tasks.start(
                    definition.name,
                    task,
                    background_operation,
                    source="agent_team",
                    usage_tracker=run.usage,
                )
            except ValueError as exc:
                handle.cancel()
                await self._cleanup_run(run.id)
                return ToolExecutionResult(f"Error: {exc}", True)
            return ToolExecutionResult(f"AgentTeam task started: {item.id[:8]}")
        result = await handle
        if result.is_error:
            await self._cleanup_run(run.id)
            return result
        payload = {
            "team": definition.name,
            "run_id": run.id,
            "status": run.terminal,
            "summary": run.summary,
        }
        await self._finalize_root(run, payload)
        await self._cleanup_run(run.id, keep_root=True)
        return ToolExecutionResult(json.dumps(payload, ensure_ascii=False))

    async def _prepare_root_workspace(self, run: AgentTeamRun) -> None:
        if run.definition.leader.workspace is not SubagentWorkspace.WORKTREE:
            return
        if (
            self.worktree_manager is None
            or not self.worktree_manager.available
            or self._context is None
        ):
            raise WorktreeUnavailableError("worktree isolation is unavailable")
        relative = self._context.cwd.resolve().relative_to(self._context.workspace_root.resolve())
        lease = await self.worktree_manager.create(
            run.id, relative, self.subagent_config.worktree.initialization_files
        )
        self._root_leases[run.id] = lease.task_id
        run.assignments[run.leader_assignment_id].lease_id = lease.task_id
        invoker = self._context.worktree_invoker
        if isinstance(invoker, SubagentService):
            invoker.adopt_worktree_draft(lease.task_id)

    async def _run_leader(self, run: AgentTeamRun) -> ToolExecutionResult:
        template = run.definition.leader
        port = _AssignmentPort(self, run, run.leader_assignment_id, True)
        registry = self._role_registry(template, leader=True)
        context = self._assignment_context(run, run.assignments[run.leader_assignment_id], port)
        members = tuple(
            await asyncio.gather(
                *(
                    self.repository.member(run.definition.name, item)
                    for item in run.definition.members
                )
            )
        )
        runner = self._runner
        assert runner is not None
        return await runner.run(
            operation_id=f"agent_team:{run.id}:{run.leader_assignment_id}",
            system_prompt=build_leader_prompt(
                self._parent.system_prompt if self._parent else "", run, members
            ),
            task=run.tasks[run.root_task_id].description,
            registry=registry,
            context=context,
            usage=run.usage,
            terminal=lambda: (
                run.terminal is not None
                or run.assignments[run.leader_assignment_id].phase is AssignmentPhase.FINISHED
            ),
        )

    async def _run_assignment(
        self, run: AgentTeamRun, assignment: AssignmentRuntime, template: TeamAgentTemplate
    ) -> ToolExecutionResult:
        port = _AssignmentPort(self, run, assignment.id, False)
        registry = self._role_registry(template, leader=False)
        context = self._assignment_context(run, assignment, port)
        task = run.tasks[assignment.task_id].description
        runner = self._runner
        assert runner is not None
        result = await runner.run(
            operation_id=f"agent_team:{run.id}:{assignment.id}",
            system_prompt=build_assignment_prompt(
                self._parent.system_prompt if self._parent else "", template, task
            ),
            task=task,
            registry=registry,
            context=context,
            usage=run.usage,
            terminal=lambda: (
                assignment.phase is AssignmentPhase.FINISHED or run.terminal is not None
            ),
        )
        if assignment.phase is not AssignmentPhase.FINISHED:
            await run.finish_assignment(assignment.id, "failed", result.content)
        return result

    def _assignment_context(
        self, run: AgentTeamRun, assignment: AssignmentRuntime, port: _AssignmentPort
    ) -> ToolExecutionContext:
        assert self._context is not None
        root = self._context.workspace_root
        cwd = self._context.cwd
        if assignment.lease_id and self.worktree_manager is not None:
            # lease metadata is only exposed internally; a model never receives its path.
            lease = self.worktree_manager._record(assignment.lease_id).lease
            root, cwd = lease.path, (lease.path / lease.relative_cwd).resolve()
        return replace(
            self._context,
            workspace_root=root,
            cwd=cwd,
            skill_invoker=None,
            subagent_invoker=None,
            worktree_invoker=None,
            agent_team_invoker=None,
            team_runtime_invoker=port,
        )

    def _role_registry(self, template: TeamAgentTemplate, *, leader: bool) -> ToolRegistry:
        assert self._registry is not None
        allowed = set(self._registry.exposed_names())
        if template.allowed_tools is not None:
            allowed.intersection_update(template.allowed_tools)
        allowed.difference_update(template.disallowed_tools)
        prohibited = {
            "run_subagent",
            "subagent_task",
            "use_skill",
            "toolSearch",
            "get_agent_teams",
            "create_agent_team",
            "run_agent_team",
            "inspect_subagent_worktree",
            "resolve_subagent_worktree",
        }
        if leader:
            prohibited.update({"shell", "edit_file", "create_file"})
        allowed.difference_update(prohibited)
        registry = self._registry.restricted_view(allowed)
        tools = (
            (
                CreateTeammateTool(),
                TaskCreateTool(),
                TaskUpdateTool(),
                GetAllTasksTool(),
                GetAllTeammatesTool(),
                SendMessageTool(),
                WaitForMessageTool(),
                FinishTeamTool(),
                InspectAssignmentWorktreeTool(),
                ResolveAssignmentWorktreeTool(),
            )
            if leader
            else (
                GetAllTasksTool(),
                GetAllTeammatesTool(),
                SendMessageTool(),
                WaitForMessageTool(),
                FinishTaskTool(),
            )
        )
        for tool in tools:
            registry.register(tool)
        return registry

    async def _finalize_root(self, run: AgentTeamRun, payload: dict[str, object]) -> None:
        root_id = self._root_leases.get(run.id)
        if root_id is None or self.worktree_manager is None:
            return
        metadata = await self.worktree_manager.finalize(root_id)
        payload["worktree"] = {
            "change_state": metadata.change_state,
            "changed_files": metadata.changed_files,
        }

    async def _cleanup_run(self, run_id: str, *, keep_root: bool = False) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        for key, handle in tuple(self._handles.items()):
            if key[0] == run_id and not handle.done():
                handle.cancel()
        await run.cleanup()
        root_id = self._root_leases.pop(run_id, None)
        if root_id and self.worktree_manager is not None and not keep_root:
            try:
                await self.worktree_manager.discard(root_id)
            except WorktreeUnavailableError:
                pass

    async def cancel_all(self, *, close: bool) -> None:
        for run_id in tuple(self._runs):
            await self._cleanup_run(run_id)


class _AssignmentPort:
    def __init__(
        self, service: AgentTeamService, run: AgentTeamRun, assignment_id: str, leader: bool
    ) -> None:
        self.service, self.run, self.assignment_id, self.leader = (
            service,
            run,
            assignment_id,
            leader,
        )

    async def invoke(self, arguments: TeamRuntimeArguments) -> ToolExecutionResult:
        try:
            if isinstance(arguments, GetAllTasksArguments):
                return await self._read_tasks()
            if isinstance(arguments, GetAllTeammatesArguments):
                return await self._read_assignments()
            if isinstance(arguments, WaitForMessageArguments):
                return await self._wait()
            if isinstance(arguments, SendMessageArguments):
                return await self._send(arguments)
            if isinstance(arguments, FinishTaskArguments) and not self.leader:
                await self.run.finish_assignment(self.assignment_id, "completed", arguments.summary)
                assignment = self.run.assignments[self.assignment_id]
                if assignment.lease_id and self.service.worktree_manager is not None:
                    await self.service.worktree_manager.finalize(assignment.lease_id)
                return ToolExecutionResult("Assignment finished.")
            if isinstance(arguments, FinishTeamArguments) and self.leader:
                await self.run.finish_team(arguments.status, arguments.summary)
                await self.run.finish_assignment(
                    self.assignment_id, arguments.status, arguments.summary
                )
                return ToolExecutionResult("Team finished.")
            if self.leader and isinstance(arguments, TaskCreateArguments):
                task = await self.run.create_task(
                    arguments.title, arguments.description, arguments.dependencies
                )
                return ToolExecutionResult(json.dumps({"task_id": task.id}, ensure_ascii=False))
            if self.leader and isinstance(arguments, TaskUpdateArguments):
                state = TeamTaskState(arguments.state) if arguments.state else None
                task = await self.run.update_task(
                    arguments.task_id, state=state, assignment_id=arguments.assignment_id
                )
                return ToolExecutionResult(
                    json.dumps({"task_id": task.id, "state": task.state.value})
                )
            if self.leader and isinstance(arguments, CreateTeammateArguments):
                return await self._create_teammate(arguments)
            if self.leader and isinstance(arguments, InspectSubagentWorktreeArguments):
                return await self._inspect(arguments)
            if self.leader and isinstance(arguments, ResolveSubagentWorktreeArguments):
                return await self._resolve(arguments)
            return ToolExecutionResult("Error: AgentTeam role cannot use this tool.", True)
        except (ValueError, WorktreeUnavailableError) as exc:
            return ToolExecutionResult(f"Error: {exc}", True)

    async def _read_tasks(self) -> ToolExecutionResult:
        snapshot = await self.run.snapshot()
        return ToolExecutionResult(
            json.dumps(
                {
                    "tasks": [item.__dict__ for item in snapshot.tasks],
                },
                ensure_ascii=False,
                default=str,
            )
        )

    async def _read_assignments(self) -> ToolExecutionResult:
        snapshot = await self.run.snapshot()
        return ToolExecutionResult(
            json.dumps(
                {
                    "assignments": [
                        {
                            "id": item.id,
                            "template_id": item.template_id,
                            "task_id": item.task_id,
                            "phase": item.phase.value,
                        }
                        for item in snapshot.assignments
                    ],
                },
                ensure_ascii=False,
            )
        )

    async def _wait(self) -> ToolExecutionResult:
        assignment = self.run.assignments[self.assignment_id]
        await self.run.set_phase(self.assignment_id, AssignmentPhase.IDLE_WAITING)
        from hammer_code.agent_team.mailbox import receive_messages

        while True:
            messages = receive_messages(self.run.path, self.assignment_id)
            if messages:
                await self.run.set_phase(self.assignment_id, AssignmentPhase.RUNNING_TOOL)
                return ToolExecutionResult(
                    json.dumps([item.__dict__ for item in messages], ensure_ascii=False)
                )
            await assignment.mail_event.wait()
            assignment.mail_event.clear()

    async def _send(self, arguments: SendMessageArguments) -> ToolExecutionResult:
        async with self.run.lock:
            sender = self.run.assignments.get(self.assignment_id)
            recipient = self.run.assignments.get(arguments.recipient_assignment_id)
            if (
                sender is None
                or recipient is None
                or sender.phase is AssignmentPhase.FINISHED
                or recipient.phase is AssignmentPhase.FINISHED
            ):
                raise ValueError("message sender or recipient is unavailable")
            self.run._send_locked(sender.id, recipient.id, arguments.body)
        return ToolExecutionResult("Message sent.")

    async def _create_teammate(self, arguments: CreateTeammateArguments) -> ToolExecutionResult:
        if arguments.existing_template:
            template = await self.service.repository.member(
                self.run.definition.name, arguments.existing_template
            )
        else:
            assert arguments.template is not None
            data = arguments.template
            template = TeamAgentTemplate(
                data.name,
                data.description,
                data.when_to_use,
                data.prompt,
                data.context,
                data.workspace,
                data.allowed_tools,
                data.disallowed_tools,
            )
            await self.service.repository.create_member(self.run.definition.name, template)
        assignment = await self.run.create_assignment(template, arguments.task_id)
        await self._prepare_assignment_workspace(assignment)
        handle = asyncio.create_task(self.service._run_assignment(self.run, assignment, template))
        self.service._handles[(self.run.id, assignment.id)] = handle
        return ToolExecutionResult(json.dumps({"assignment_id": assignment.id}, ensure_ascii=False))

    async def _prepare_assignment_workspace(self, assignment: AssignmentRuntime) -> None:
        if assignment.workspace is not SubagentWorkspace.WORKTREE:
            return
        parent_id = self.service._root_leases.get(self.run.id)
        if parent_id is None or self.service.worktree_manager is None:
            raise WorktreeUnavailableError("Team root worktree is unavailable")
        lease = await self.service.worktree_manager.create_child(
            assignment.id,
            parent_id,
            Path("."),
            self.service.subagent_config.worktree.initialization_files,
        )
        assignment.lease_id = lease.task_id

    async def _inspect(self, arguments: InspectSubagentWorktreeArguments) -> ToolExecutionResult:
        if self.service.worktree_manager is None or arguments.task_id not in self.run.assignments:
            raise ValueError("Team Assignment worktree is unavailable")
        value = await self.service.worktree_manager.inspect(
            arguments.task_id, arguments.detail, arguments.paths
        )
        return ToolExecutionResult(json.dumps(value, ensure_ascii=False, default=str))

    async def _resolve(self, arguments: ResolveSubagentWorktreeArguments) -> ToolExecutionResult:
        if self.service.worktree_manager is None or arguments.task_id not in self.run.assignments:
            raise ValueError("Team Assignment worktree is unavailable")
        await self._authorize_resolution(arguments)
        if arguments.resolution == "integrate":
            result = await self.service.worktree_manager.integrate(arguments.task_id)
        elif arguments.resolution == "discard":
            result = await self.service.worktree_manager.discard(arguments.task_id)
        else:
            result = await self.service.worktree_manager.agent_merge(
                arguments.task_id,
                arguments.inspection_id or "",
                tuple(item.model_dump() for item in arguments.edits),
            )
        return ToolExecutionResult(
            json.dumps(result.__dict__, ensure_ascii=False, default=str),
            not result.main_workspace_changed and result.change_state not in {"discarded"},
        )

    async def _authorize_resolution(self, arguments: ResolveSubagentWorktreeArguments) -> None:
        if arguments.resolution == "discard":
            return
        manager = self.service.worktree_manager
        runner = self.service._runner
        root_id = self.service._root_leases.get(self.run.id)
        if manager is None or runner is None or root_id is None:
            raise ValueError("Team root worktree is unavailable")
        root = manager._record(root_id).lease.path
        if arguments.resolution == "agent_merge":
            paths = tuple((root / item.path).resolve() for item in arguments.edits)
            actions = tuple(
                {"path": item.path, "action": item.action, "sha256": item.expected_current_hash}
                for item in arguments.edits
            )
        else:
            metadata = manager.metadata(arguments.task_id)
            paths = tuple((root / item).resolve() for item in metadata.changed_files)
            actions = tuple(
                {"path": item, "action": "integrate"} for item in metadata.changed_files
            )
        request = PermissionRequest.for_tool(
            "resolve_assignment_worktree",
            "write",
            {
                "assignment_id": arguments.task_id,
                "resolution": arguments.resolution,
                "actions": actions,
            },
            paths,
            root,
            root,
        )
        await runner.permissions.authorize(request)

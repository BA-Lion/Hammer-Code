"""One-turn Agent implementations used by the outer interactive loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from hammer_code.app.context_manager import ContextManager
from hammer_code.app.context_window import ContextWindow
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelRequest,
    ReasoningDelta,
    ResponseCompleted,
    ResponseStarted,
    TextDelta,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus, UsageSummary
from hammer_code.errors import ContextCompactionError, HammerCodeError, StreamInterruptedError
from hammer_code.hooks.manager import HookManager
from hammer_code.hooks.models import HookContext, LifecycleEvent
from hammer_code.llm.client import ModelClient
from hammer_code.mcp.manager import McpManager
from hammer_code.memory.service import MemoryService
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.prompts import build_stale_restore_prompt
from hammer_code.session.session import SessionCoordinator
from hammer_code.skill.evolution import SkillEvolutionService
from hammer_code.skill.models import SkillObservation
from hammer_code.skill.prompts import build_skill_request_prompt
from hammer_code.skill.repository import SkillRepositoryError
from hammer_code.skill.service import SkillInvocationService
from hammer_code.subagent.models import ParentRequestSnapshot
from hammer_code.subagent.prompts import build_subagent_catalog_prompt
from hammer_code.subagent.service import (
    BackgroundTaskManager,
    SubagentService,
    SubagentTaskSnapshot,
    format_background_results,
)
from hammer_code.tools.executor import ToolBatchCancelled, ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore
from hammer_code.ui.console import ConsolePort


class Agent(Protocol):
    """The narrow capability required to execute one ordinary user turn."""

    async def run_turn(self, text: str) -> None: ...


@dataclass(frozen=True)
class AgentStatus:
    profile: str
    protocol: str
    model: str
    session_id: str
    permission_mode: PermissionMode
    persistence_degraded: bool
    usage: UsageSummary


class PrimaryAgent:
    """Owns the current model, tool, context, and persistence turn pipeline."""

    def __init__(
        self,
        manager: ConversationManager,
        client: ModelClient,
        ui: ConsolePort,
        system_prompt: str,
        max_output_tokens: int,
        show_reasoning: bool = False,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        context_window: ContextWindow | None = None,
        mcp_manager: McpManager | None = None,
        context_manager: ContextManager | None = None,
        session_coordinator: SessionCoordinator | None = None,
        memory_service: MemoryService | None = None,
        project_instructions: str = "",
        stale_restore: bool = False,
        permissions: PermissionService | None = None,
        prompt_builder: Callable[[PermissionMode], str] | None = None,
        runtime: RuntimeStore | None = None,
        skill_service: SkillInvocationService | None = None,
        skill_evolution: SkillEvolutionService | None = None,
        hooks: HookManager | None = None,
        subagent_tasks: BackgroundTaskManager | None = None,
        subagent_service: SubagentService | None = None,
    ) -> None:
        (
            self.manager,
            self.client,
            self.ui,
            self.system_prompt,
            self.max_output_tokens,
            self.show_reasoning,
            self.registry,
            self.executor,
            self.context_window,
            self.mcp_manager,
            self.context_manager,
            self.session_coordinator,
            self.memory_service,
            self.project_instructions,
            self.stale_restore,
            self.permissions,
            self._prompt_builder,
            self.runtime,
            self.skill_service,
            self.skill_evolution,
            self.hooks,
            self.subagent_tasks,
            self.subagent_service,
        ) = (
            manager,
            client,
            ui,
            system_prompt,
            max_output_tokens,
            show_reasoning,
            registry,
            executor,
            context_window or ContextWindow(system_prompt),
            mcp_manager,
            context_manager,
            session_coordinator,
            memory_service,
            project_instructions,
            stale_restore,
            permissions,
            prompt_builder,
            runtime,
            skill_service,
            skill_evolution,
            hooks,
            subagent_tasks,
            subagent_service,
        )
        self._accepting_turns = True
        self._closed = False
        self._finalization_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._previous_skill_observation: SkillObservation | None = None
        self._started = False

    @property
    def session_id(self) -> str:
        if self.session_coordinator is not None:
            return self.session_coordinator.session.meta.id
        if self.manager.conversation is not None:
            return self.manager.conversation.id
        return "unavailable"

    def status(self) -> AgentStatus:
        conversation = self.manager.conversation
        if conversation is None:
            raise HammerCodeError("Agent has no active conversation")
        return AgentStatus(
            profile=conversation.profile_name,
            protocol=conversation.protocol,
            model=conversation.model,
            session_id=self.session_id,
            permission_mode=self.permissions.mode if self.permissions else PermissionMode.DEFAULT,
            persistence_degraded=bool(
                self.session_coordinator and self.session_coordinator.persistence_degraded
            ),
            usage=conversation.usage_ledger.for_conversation(),
        )

    async def clear(self) -> None:
        async with self._lifecycle_lock:
            if self.subagent_tasks is not None:
                await self.subagent_tasks.cancel_all(close=False, discard_results=True)
            if self.skill_evolution is not None:
                await self.skill_evolution.wait_idle()
            if self.memory_service is not None:
                await self.memory_service.wait_idle()
            self.manager.clear()
            if self.session_coordinator is not None:
                await self.session_coordinator.clear_history()
            cleanup_ok = self.context_manager.clear() if self.context_manager else True
            if self.registry:
                self.registry.clear_discovered()
            if cleanup_ok:
                self.ui.info("Conversation cleared.")
            else:
                self.ui.error(
                    "Conversation cleared, but some temporary tool results could not be removed."
                )

    async def compact(self) -> None:
        if self.context_manager is None:
            self.ui.error("Context compaction is unavailable.")
            return
        try:
            prompt, tools = self._sample_context()
            before = self.manager.snapshot_committed()
            if self.memory_service is not None:
                await self.memory_service.flush_before_compaction()
            batch = self.hooks.snapshot_prompt() if self.hooks else None
            preparation = await self.context_manager.compact_now(
                mcp_prompt=prompt, hook_prompt=batch.text if batch else "", tools=tools
            )
            await self._persist_preparation(before, preparation)
            if preparation.compact_event is not None and self.hooks is not None:
                await self.hooks.dispatch(LifecycleEvent.COMPACT)
            if preparation.compact_event is None:
                self.ui.info("Nothing to compact.")
            else:
                self.ui.compact(preparation.compact_event)
                if preparation.cleanup_failed:
                    self.ui.error(
                        "Context compacted, but some temporary tool results could not be removed."
                    )
        except ContextCompactionError as exc:
            self.ui.error(str(exc))

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if self.permissions is None or self._prompt_builder is None:
            raise HammerCodeError("Runtime permission switching is unavailable")
        prompt = self._prompt_builder(mode)
        if not prompt.strip():
            raise HammerCodeError("Runtime permission prompt is unavailable")
        self.permissions.set_mode(mode)
        self.context_window.set_base_system_prompt(prompt)
        if self.context_manager is not None:
            self.context_manager.set_base_system_prompt(prompt)
        self.system_prompt = prompt

    async def drain(self) -> None:
        self._accepting_turns = False
        await self._finish(cancel_memory=False)

    async def cancel(self) -> None:
        self._accepting_turns = False
        await self._finish(cancel_memory=True)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.hooks is not None:
            await self.hooks.dispatch(LifecycleEvent.SESSION_START)

    async def run_skill(self, name: str, arguments: str) -> None:
        """Run a local inline Skill as an ordinary durable turn."""
        if self.skill_service is None:
            raise HammerCodeError("Skill invocation is unavailable")
        await self.run_turn(
            f"/skill {name}{(' ' + arguments) if arguments else ''}", skill=(name, arguments)
        )

    async def feedback_skill(self, name: str, feedback: str) -> None:
        if self.skill_evolution is None:
            raise HammerCodeError("Skill feedback is unavailable")
        await self.skill_evolution.feedback(name, feedback)
        self.ui.info("Skill feedback queued for controlled maintenance.")

    async def list_subagent_tasks(self) -> tuple[SubagentTaskSnapshot, ...]:
        if self.subagent_tasks is None:
            raise HammerCodeError("Subagent task management is unavailable")
        return await self.subagent_tasks.list()

    async def get_subagent_task(self, task_id: str) -> SubagentTaskSnapshot:
        if self.subagent_tasks is None:
            raise HammerCodeError("Subagent task management is unavailable")
        try:
            return await self.subagent_tasks.get(task_id)
        except ValueError as exc:
            raise HammerCodeError(str(exc)) from exc

    async def cancel_subagent_task(self, task_id: str) -> SubagentTaskSnapshot:
        if self.subagent_tasks is None:
            raise HammerCodeError("Subagent task management is unavailable")
        try:
            return await self.subagent_tasks.cancel(task_id)
        except ValueError as exc:
            raise HammerCodeError(str(exc)) from exc

    async def run_turn(self, text: str, skill: tuple[str, str] | None = None) -> None:
        async with self._lifecycle_lock:
            await self._run_turn(text, skill)

    async def _run_turn(self, text: str, skill: tuple[str, str] | None = None) -> None:
        if not self._accepting_turns:
            raise HammerCodeError("Agent is draining and cannot accept another turn")
        turn = None
        request_count = 0
        call_count = 0
        unknown_count = 0
        skill_prompt = ""
        skill_started = False
        turn_error = ""
        try:
            await self._flush_subagent_results_locked()
            if self.hooks is not None:
                await self.hooks.dispatch(LifecycleEvent.TURN_START, HookContext(message=text))
            if self.skill_service is not None:
                try:
                    skill_snapshot = await self.skill_service.repository.snapshot_for_turn()
                    if skill is None:
                        index = skill_snapshot.index
                        retrieved = (
                            index.retrieve(
                                text,
                                skill_snapshot.effective.values(),
                                top_k=self.skill_service.skill_config.retrieval_top_k,
                                relative_floor=self.skill_service.skill_config.retrieval_relative_floor,
                                coverage_floor=self.skill_service.skill_config.retrieval_query_coverage,
                            )
                            if self.skill_service.skill_config.enabled and index is not None
                            else ()
                        )
                        if retrieved:
                            await self.skill_service.repository.record_retrieved(
                                tuple(item.ref for item in retrieved)
                            )
                        skill_prompt = build_skill_request_prompt(skill_snapshot, retrieved)
                    else:
                        retrieved = ()
                    self.skill_service.begin_turn(
                        skill_snapshot, SkillObservation(skill_snapshot.generation, retrieved)
                    )
                    skill_started = True
                    if self.skill_evolution is not None and skill is None:
                        self.skill_evolution.schedule(
                            self.manager.snapshot_committed(),
                            text,
                            self._previous_skill_observation,
                        )
                    if skill is not None:
                        prepared = await self.skill_service.invoke_user(*skill)
                        if prepared.is_fork:
                            result = await self.skill_service.invoke_prepared_fork(
                                prepared, skill[1]
                            )
                            turn = self.manager.begin_turn(text)
                            if result.is_error:
                                self.ui.error(result.content.removeprefix("Error: ").strip())
                                committed = self.manager.interrupt(turn)
                                await self._persist_turn_outcome(committed, completed=False)
                            else:
                                self.ui.text_delta(result.content)
                                committed = self.manager.commit(
                                    turn, Message(Role.ASSISTANT, (TextBlock(result.content),))
                                )
                                await self._persist_turn_outcome(committed, completed=True)
                            return
                        skill_prompt = prepared.prompt
                except SkillRepositoryError:
                    self.ui.error("Skill catalog is unavailable for this turn.")
            turn = self.manager.begin_turn(text)
            while request_count < 50:
                request_count += 1
                request_id = str(uuid4())
                completed = None
                announced_calls: set[str] = set()
                reasoning_status_started = False
                prompt, tools = self._sample_context()
                subagent_catalog = None
                subagent_prompt = ""
                if self.subagent_service is not None:
                    subagent_catalog = await self.subagent_service.repository.snapshot_for_request()
                    subagent_prompt = build_subagent_catalog_prompt(subagent_catalog)
                hooks = self.hooks
                if hooks is not None:
                    await hooks.dispatch(LifecycleEvent.PRE_SEND, HookContext(message=text))
                    hook_batch = hooks.snapshot_prompt()
                else:
                    hook_batch = None
                if self.context_manager is not None:
                    before = self.manager.snapshot_committed()
                    preparation = await self.context_manager.prepare_before_request(
                        current_suffix=self.manager.snapshot_staged(turn),
                        mcp_prompt=prompt,
                        subagent_prompt=subagent_prompt,
                        hook_prompt=hook_batch.text if hook_batch else "",
                        skill_prompt=skill_prompt,
                        tools=tools,
                        before_compaction=(
                            self.memory_service.flush_before_compaction
                            if self.memory_service is not None
                            else None
                        ),
                    )
                    self._show_preparation(preparation)
                    await self._persist_preparation(before, preparation)
                    if preparation.compact_event is not None and hooks is not None:
                        await hooks.dispatch(LifecycleEvent.COMPACT)
                        hook_batch = hooks.snapshot_prompt()
                memory_prompt = (
                    await self.memory_service.read_index_prompt()
                    if self.memory_service is not None
                    else ""
                )
                snapshot = self.context_window.snapshot(
                    mcp_prompt=prompt,
                    project_instructions=self.project_instructions,
                    subagent_prompt=subagent_prompt,
                    hook_prompt=hook_batch.text if hook_batch else "",
                    skill_prompt=skill_prompt,
                    memory_prompt=memory_prompt,
                    stale_prompt=build_stale_restore_prompt() if self.stale_restore else "",
                    recovery_prompt=self.context_manager.recovery_prompt
                    if self.context_manager
                    else "",
                    messages=self.manager.snapshot_for_request(turn),
                    tools=tools,
                )
                request = ModelRequest(
                    request_id,
                    turn.id,
                    snapshot.system_prompt,
                    snapshot.messages,
                    snapshot.tools,
                    self.max_output_tokens,
                )
                if self.subagent_service is not None and subagent_catalog is not None:
                    self.subagent_service.bind_request(
                        ParentRequestSnapshot(
                            subagent_catalog,
                            snapshot.system_prompt,
                            snapshot.messages,
                            snapshot.tools,
                            frozenset(tool.name for tool in snapshot.tools),
                            self.system_prompt,
                            self.project_instructions,
                            prompt,
                        )
                    )
                if hook_batch is not None and hooks is not None:
                    hooks.commit_prompt(hook_batch)
                self.manager.record_usage(
                    turn,
                    request_id,
                    TokenUsage(None, None, status=UsageStatus.UNAVAILABLE),
                )
                async for event in self.client.stream(request):
                    if event.request_id != request_id:
                        raise StreamInterruptedError(
                            "Client emitted an event for a different request"
                        )
                    if isinstance(event, ResponseStarted):
                        continue
                    if isinstance(event, TextDelta):
                        self.ui.text_delta(event.text)
                    elif isinstance(event, ReasoningDelta):
                        if self.show_reasoning:
                            self.ui.reasoning_delta(event.text, event.visibility)
                        elif not reasoning_status_started:
                            self.ui.reasoning_status()
                            reasoning_status_started = True
                    elif isinstance(event, ToolCallCompleted):
                        announced_calls.add(event.tool_call.call_id)
                        self.ui.tool_call_notice(event.tool_call)
                    elif isinstance(event, UsageUpdated):
                        self.manager.record_usage(turn, request_id, event.usage)
                    elif isinstance(event, ResponseCompleted):
                        if completed is not None:
                            raise StreamInterruptedError("Client emitted duplicate completion")
                        completed = event
                if completed is None:
                    raise StreamInterruptedError("Client ended without a completion")
                response = completed.response
                if self.hooks is not None:
                    visible = "".join(
                        block.text
                        for block in response.message.content
                        if isinstance(block, TextBlock)
                    )
                    await self.hooks.dispatch(
                        LifecycleEvent.POST_RECEIVE, HookContext(message=visible)
                    )
                calls = tuple(
                    block for block in response.message.content if isinstance(block, ToolCallBlock)
                )
                if response.stop_reason.value != "tool_call":
                    committed = self.manager.commit(turn, response.message)
                    await self._persist_turn_outcome(committed, completed=True)
                    return
                if not calls or {call.call_id for call in calls} != announced_calls:
                    raise StreamInterruptedError(
                        "Completed tool response does not match tool call events"
                    )
                call_count += len(calls)
                unknown_count += sum(
                    self.registry is None or not self.registry.exposed(call.name) for call in calls
                )
                if call_count > 200 or unknown_count > 3 or self.executor is None:
                    self.manager.stage_tool_call(turn, response.message)
                    self.ui.error("Tool loop limit reached or tool executor is unavailable")
                    committed = self.manager.interrupt(turn)
                    await self._persist_turn_outcome(committed, completed=False)
                    return
                self.manager.stage_tool_call(turn, response.message)
                results = await self.executor.execute_batch(calls)
                self.manager.stage_tool_results(turn, Message(Role.USER, tuple(results)))
            self.ui.error("Tool loop request limit reached")
            committed = self.manager.interrupt(turn)
            await self._persist_turn_outcome(committed, completed=False)
        except ToolBatchCancelled as exc:
            turn_error = "tool execution cancelled"
            if turn is not None:
                self.manager.stage_tool_results(turn, Message(Role.USER, exc.results))
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            raise asyncio.CancelledError from exc
        except asyncio.CancelledError:
            turn_error = "turn cancelled"
            if turn is not None:
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            raise
        except Exception as exc:
            turn_error = (
                str(exc) if isinstance(exc, HammerCodeError) else "unexpected model failure"
            )
            if turn is not None:
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            self.ui.error(
                str(exc) if isinstance(exc, HammerCodeError) else "Unexpected model failure"
            )
            if self.hooks is not None:
                await self.hooks.dispatch(LifecycleEvent.ERROR, HookContext(error=turn_error))
        finally:
            if skill_started and self.skill_service is not None:
                observation = self.skill_service.end_turn(
                    completed=turn is not None and turn.status.value == "committed"
                )
                if observation is not None and observation.completed:
                    self._previous_skill_observation = observation
            if self.subagent_service is not None:
                self.subagent_service.clear_request()
            if self.manager.conversation and turn is not None:
                self.ui.usage(self.manager.conversation.usage_ledger.for_turn(turn.id))
            if self.hooks is not None:
                await self.hooks.dispatch(
                    LifecycleEvent.TURN_END, HookContext(message=text, error=turn_error)
                )
            await self._flush_subagent_results_locked()

    async def _finish(self, *, cancel_memory: bool) -> None:
        async with self._finalization_lock:
            if self._closed:
                return
            async with self._lifecycle_lock:
                if self.subagent_tasks is not None:
                    await self.subagent_tasks.cancel_all(close=True, discard_results=False)
                    await self._flush_subagent_results_locked()
                if self.hooks is not None:
                    await self.hooks.dispatch(LifecycleEvent.SESSION_END)
                    if cancel_memory:
                        await self.hooks.cancel()
                    else:
                        await self.hooks.drain()
                if self.memory_service is not None:
                    if cancel_memory:
                        await self.memory_service.cancel()
                    else:
                        await self.memory_service.drain()
                if self.skill_evolution is not None:
                    if cancel_memory:
                        await self.skill_evolution.cancel()
                    else:
                        await self.skill_evolution.drain()
                if self.session_coordinator is not None:
                    await self.session_coordinator.close()
                if self.runtime is not None:
                    await asyncio.to_thread(self.runtime.cleanup)
                self._closed = True

    async def _flush_subagent_results(self) -> None:
        """Persist one atomic batch of completed background results between main turns."""
        if self.subagent_tasks is None:
            return
        async with self._lifecycle_lock:
            await self._flush_subagent_results_locked()

    async def _flush_subagent_results_locked(self) -> None:
        """Flush one batch while the caller owns ``_lifecycle_lock``."""
        if self.subagent_tasks is None:
            return
        items = await self.subagent_tasks.take_completed()
        if not items:
            return
        try:
            messages = self.manager.append_completed_runtime_turn(
                format_background_results(items), "Subagent results recorded."
            )
        except Exception:
            await self.subagent_tasks.requeue_front(items)
            raise
        if self.session_coordinator is not None:
            await self.session_coordinator.append_turn(messages, completed=True)
        if self.memory_service is not None and not (
            self.session_coordinator and self.session_coordinator.persistence_degraded
        ):
            self.memory_service.maybe_schedule()

    def _sample_context(self) -> tuple[str, tuple]:
        return (
            self.mcp_manager.prompt if self.mcp_manager else "",
            self.registry.definitions() if self.registry else (),
        )

    def _show_preparation(self, preparation: object) -> None:
        event = getattr(preparation, "compact_event", None)
        if event is not None:
            self.ui.compact(event)
        if getattr(preparation, "cleanup_failed", False):
            self.ui.error(
                "Context compacted, but some temporary tool results could not be removed."
            )

    async def _persist_turn_outcome(
        self, messages: tuple[Message, ...], *, completed: bool
    ) -> None:
        if self.session_coordinator is None or not messages:
            return
        await self.session_coordinator.append_turn(messages, completed=completed)
        if self.session_coordinator.persistence_degraded:
            self.ui.persistence_warning(
                "Conversation continues in memory; recent history may be lost after interruption."
            )
            return
        if self.memory_service is not None:
            self.memory_service.maybe_schedule()

    async def _persist_preparation(self, before: tuple[Message, ...], preparation: object) -> None:
        if self.session_coordinator is None or not getattr(preparation, "history_replaced", False):
            return
        summary = getattr(preparation, "summary", None)
        positions = getattr(preparation, "summarized_turn_positions", ())
        if not isinstance(summary, str) or not isinstance(positions, tuple):
            self.ui.persistence_warning("Compacted history could not be saved.")
            return
        await self.session_coordinator.rewrite_after_compaction(
            before,
            self.manager.snapshot_committed(),
            summary=summary,
            summarized_turn_positions=positions,
        )
        if self.session_coordinator.persistence_degraded:
            self.ui.persistence_warning(
                "Context compacted in memory, but compacted history could not be saved."
            )

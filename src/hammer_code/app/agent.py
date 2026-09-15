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
from hammer_code.llm.client import ModelClient
from hammer_code.mcp.manager import McpManager
from hammer_code.memory.service import MemoryService
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.prompts import build_stale_restore_prompt
from hammer_code.session.session import SessionCoordinator
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
        )
        self._accepting_turns = True
        self._closed = False
        self._finalization_lock = asyncio.Lock()

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
            preparation = await self.context_manager.compact_now(mcp_prompt=prompt, tools=tools)
            await self._persist_preparation(before, preparation)
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

    async def run_turn(self, text: str) -> None:
        if not self._accepting_turns:
            raise HammerCodeError("Agent is draining and cannot accept another turn")
        turn = None
        request_count = 0
        call_count = 0
        unknown_count = 0
        try:
            if self.context_manager is not None:
                pending = Message(Role.USER, (TextBlock(text),))
                prompt, tools = self._sample_context()
                before = self.manager.snapshot_committed()
                preparation = await self.context_manager.prepare_before_request(
                    current_suffix=(pending,),
                    mcp_prompt=prompt,
                    tools=tools,
                    before_compaction=(
                        self.memory_service.flush_before_compaction
                        if self.memory_service is not None
                        else None
                    ),
                )
                self._show_preparation(preparation)
                await self._persist_preparation(before, preparation)
            turn = self.manager.begin_turn(text)
            while request_count < 50:
                request_count += 1
                request_id = str(uuid4())
                completed = None
                announced_calls: set[str] = set()
                reasoning_status_started = False
                prompt, tools = self._sample_context()
                if self.context_manager is not None and request_count > 1:
                    before = self.manager.snapshot_committed()
                    preparation = await self.context_manager.prepare_before_request(
                        current_suffix=self.manager.snapshot_staged(turn),
                        mcp_prompt=prompt,
                        tools=tools,
                        before_compaction=(
                            self.memory_service.flush_before_compaction
                            if self.memory_service is not None
                            else None
                        ),
                    )
                    self._show_preparation(preparation)
                    await self._persist_preparation(before, preparation)
                memory_prompt = (
                    await self.memory_service.read_index_prompt()
                    if self.memory_service is not None
                    else ""
                )
                snapshot = self.context_window.snapshot(
                    mcp_prompt=prompt,
                    project_instructions=self.project_instructions,
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
            if turn is not None:
                self.manager.stage_tool_results(turn, Message(Role.USER, exc.results))
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            raise asyncio.CancelledError from exc
        except asyncio.CancelledError:
            if turn is not None:
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            raise
        except Exception as exc:
            if turn is not None:
                committed = self.manager.interrupt(turn)
                await self._persist_turn_outcome(committed, completed=False)
            self.ui.error(
                str(exc) if isinstance(exc, HammerCodeError) else "Unexpected model failure"
            )
        finally:
            if self.manager.conversation and turn is not None:
                self.ui.usage(self.manager.conversation.usage_ledger.for_turn(turn.id))

    async def _finish(self, *, cancel_memory: bool) -> None:
        async with self._finalization_lock:
            if self._closed:
                return
            if self.memory_service is not None:
                if cancel_memory:
                    await self.memory_service.cancel()
                else:
                    await self.memory_service.drain()
            if self.session_coordinator is not None:
                await self.session_coordinator.close()
            if self.runtime is not None:
                await asyncio.to_thread(self.runtime.cleanup)
            self._closed = True

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

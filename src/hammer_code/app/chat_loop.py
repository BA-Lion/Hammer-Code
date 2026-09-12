"""The single conversation loop consuming only internal ModelEvent objects."""

from __future__ import annotations

import asyncio
from contextlib import suppress
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
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import ContextCompactionError, HammerCodeError, StreamInterruptedError
from hammer_code.llm.client import ModelClient
from hammer_code.mcp.manager import McpManager
from hammer_code.tools.executor import ToolBatchCancelled, ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.ui.console import ConsolePort


class ChatLoop:
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
        )
        self._exit = False

    async def run(self) -> None:
        while not self._exit:
            try:
                text = await self.ui.prompt()
            except (EOFError, KeyboardInterrupt):
                return
            if not text.strip():
                continue
            if text.startswith("/"):
                await self._command(text)
            else:
                task = asyncio.create_task(self.run_turn(text))
                try:
                    await task
                except KeyboardInterrupt:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    self.ui.error("Request cancelled.")

    async def _command(self, text: str) -> None:
        if text == "/help":
            self.ui.help()
        elif text == "/clear":
            try:
                self.manager.clear()
                cleanup_ok = self.context_manager.clear() if self.context_manager else True
                if self.registry:
                    self.registry.clear_discovered()
                if cleanup_ok:
                    self.ui.info("Conversation cleared.")
                else:
                    self.ui.error(
                        "Conversation cleared, but some temporary tool "
                        "results could not be removed."
                    )
            except HammerCodeError as exc:
                self.ui.error(str(exc))
        elif text == "/usage":
            if self.manager.conversation:
                self.ui.usage(self.manager.conversation.usage_ledger.for_conversation())
        elif text == "/exit":
            self._exit = True
        elif text == "/compact":
            if self.context_manager is None:
                self.ui.error("Context compaction is unavailable.")
                return
            try:
                prompt, tools = self._sample_context()
                preparation = await self.context_manager.compact_now(mcp_prompt=prompt, tools=tools)
                if preparation.compact_event is None:
                    self.ui.info("Nothing to compact.")
                else:
                    self.ui.compact(preparation.compact_event)
                    if preparation.cleanup_failed:
                        self.ui.error(
                            "Context compacted, but some temporary tool "
                            "results could not be removed."
                        )
            except ContextCompactionError as exc:
                self.ui.error(str(exc))
        else:
            self.ui.error("Unknown command. Use /help.")

    async def run_turn(self, text: str) -> None:
        turn = None
        request_count = 0
        call_count = 0
        unknown_count = 0
        try:
            if self.context_manager is not None:
                pending = Message(Role.USER, (TextBlock(text),))
                prompt, tools = self._sample_context()
                preparation = await self.context_manager.prepare_before_request(
                    current_suffix=(pending,), mcp_prompt=prompt, tools=tools
                )
                self._show_preparation(preparation)
            turn = self.manager.begin_turn(text)
            while request_count < 50:
                request_count += 1
                request_id = str(uuid4())
                completed = None
                announced_calls: set[str] = set()
                reasoning_status_started = False
                prompt, tools = self._sample_context()
                if self.context_manager is not None and request_count > 1:
                    preparation = await self.context_manager.prepare_before_request(
                        current_suffix=self.manager.snapshot_staged(turn),
                        mcp_prompt=prompt,
                        tools=tools,
                    )
                    self._show_preparation(preparation)
                snapshot = self.context_window.snapshot(
                    mcp_prompt=prompt,
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
                    self.manager.commit(turn, response.message)
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
                    self.manager.interrupt(turn)
                    return
                self.manager.stage_tool_call(turn, response.message)
                results = await self.executor.execute_batch(calls)
                self.manager.stage_tool_results(turn, Message(Role.USER, tuple(results)))
            self.ui.error("Tool loop request limit reached")
            self.manager.interrupt(turn)
        except ToolBatchCancelled as exc:
            if turn is not None:
                self.manager.stage_tool_results(turn, Message(Role.USER, exc.results))
                self.manager.interrupt(turn)
            raise asyncio.CancelledError from exc
        except asyncio.CancelledError:
            if turn is not None:
                self.manager.interrupt(turn)
            raise
        except Exception as exc:
            if turn is not None:
                self.manager.interrupt(turn)
            self.ui.error(
                str(exc) if isinstance(exc, HammerCodeError) else "Unexpected model failure"
            )
        finally:
            if self.manager.conversation and turn is not None:
                self.ui.usage(self.manager.conversation.usage_ledger.for_turn(turn.id))

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

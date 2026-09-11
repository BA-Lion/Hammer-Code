"""The single conversation loop consuming only internal ModelEvent objects."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import uuid4

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
from hammer_code.domain.messages import Message, Role, ToolCallBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import HammerCodeError, StreamInterruptedError
from hammer_code.llm.client import ModelClient
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
        ) = (
            manager,
            client,
            ui,
            system_prompt,
            max_output_tokens,
            show_reasoning,
            registry,
            executor,
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
                self._command(text)
            else:
                task = asyncio.create_task(self.run_turn(text))
                try:
                    await task
                except KeyboardInterrupt:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    self.ui.error("Request cancelled.")

    def _command(self, text: str) -> None:
        if text == "/help":
            self.ui.help()
        elif text == "/clear":
            try:
                self.manager.clear()
                self.ui.info("Conversation cleared.")
            except HammerCodeError as exc:
                self.ui.error(str(exc))
        elif text == "/usage":
            if self.manager.conversation:
                self.ui.usage(self.manager.conversation.usage_ledger.for_conversation())
        elif text == "/exit":
            self._exit = True
        else:
            self.ui.error("Unknown command. Use /help.")

    async def run_turn(self, text: str) -> None:
        turn = self.manager.begin_turn(text)
        request_count = 0
        call_count = 0
        unknown_count = 0
        try:
            while request_count < 50:
                request_count += 1
                request_id = str(uuid4())
                completed = None
                announced_calls: set[str] = set()
                reasoning_status_started = False
                request = ModelRequest(
                    request_id,
                    turn.id,
                    self.system_prompt,
                    self.manager.snapshot_for_request(turn),
                    self.registry.definitions() if self.registry else (),
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
                    self.registry is None or not self.registry.enabled(call.name) for call in calls
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
            self.manager.stage_tool_results(turn, Message(Role.USER, exc.results))
            self.manager.interrupt(turn)
            raise asyncio.CancelledError from exc
        except asyncio.CancelledError:
            self.manager.interrupt(turn)
            raise
        except Exception as exc:
            self.manager.interrupt(turn)
            self.ui.error(
                str(exc) if isinstance(exc, HammerCodeError) else "Unexpected model failure"
            )
        finally:
            if self.manager.conversation:
                self.ui.usage(self.manager.conversation.usage_ledger.for_turn(turn.id))

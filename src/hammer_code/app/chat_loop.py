"""The single conversation loop consuming only internal ModelEvent objects."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import uuid4

from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ResponseCompleted,
    ResponseStarted,
    TextDelta,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import HammerCodeError, StreamInterruptedError
from hammer_code.llm.client import ModelClient
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
    ) -> None:
        (
            self.manager,
            self.client,
            self.ui,
            self.system_prompt,
            self.max_output_tokens,
            self.show_reasoning,
        ) = (
            manager,
            client,
            ui,
            system_prompt,
            max_output_tokens,
            show_reasoning,
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
        request_id = str(uuid4())
        completed = None
        reasoning_status_started = False
        try:
            from hammer_code.domain.events import ModelRequest, ReasoningDelta

            request = ModelRequest(
                request_id,
                turn.id,
                self.system_prompt,
                self.manager.snapshot_for_request(turn),
                (),
                self.max_output_tokens,
            )
            self.manager.record_usage(
                turn,
                request_id,
                TokenUsage(None, None, status=UsageStatus.UNAVAILABLE),
            )
            async for event in self.client.stream(request):
                if event.request_id != request_id:
                    raise StreamInterruptedError("Client emitted an event for a different request")
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
                    self.ui.tool_call_notice(event.tool_call)
                elif isinstance(event, UsageUpdated):
                    self.manager.record_usage(turn, request_id, event.usage)
                elif isinstance(event, ResponseCompleted):
                    if completed is not None:
                        raise StreamInterruptedError("Client emitted duplicate completion")
                    completed = event
            if completed is None:
                raise StreamInterruptedError("Client ended without a completion")
            self.manager.commit(turn, completed.response.message)
        except asyncio.CancelledError:
            self.manager.abort(turn)
            raise
        except Exception as exc:
            self.manager.abort(turn)
            self.ui.error(
                str(exc) if isinstance(exc, HammerCodeError) else "Unexpected model failure"
            )
        finally:
            if self.manager.conversation:
                self.ui.usage(self.manager.conversation.usage_ledger.for_turn(turn.id))

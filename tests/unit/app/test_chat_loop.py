from collections.abc import AsyncIterator

import pytest

from hammer_code.app.chat_loop import ChatLoop
from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    ResponseStarted,
    StopReason,
    TextDelta,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.llm.client import ClientCapabilities, ModelClient


class FakeUI:
    def __init__(self) -> None:
        self.text = []
        self.errors = []
        self.shown_usage = []

    async def prompt(self) -> str:
        return "/exit"

    def text_delta(self, text: str) -> None:
        self.text.append(text)

    def reasoning_status(self) -> None:
        pass

    def reasoning_delta(self, text, visibility) -> None:
        pass

    def tool_call_notice(self, call) -> None:
        pass

    def error(self, message: str) -> None:
        self.errors.append(message)

    def usage(self, summary) -> None:
        self.shown_usage.append(summary)

    def help(self) -> None:
        pass

    def info(self, message: str) -> None:
        pass


class FakeClient(ModelClient):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ResponseStarted(request.request_id, "provider", "model")
        yield TextDelta(request.request_id, 0, "hello")
        yield UsageUpdated(request.request_id, TokenUsage(1, 2, status=UsageStatus.FINAL))
        if self.fail:
            raise RuntimeError("boom")
        response = ModelResponse(
            "provider",
            "model",
            Message(Role.ASSISTANT, (TextBlock("hello"),)),
            StopReason.END_TURN,
            TokenUsage(1, 2, status=UsageStatus.FINAL),
        )
        yield ResponseCompleted(request.request_id, response)


def _manager() -> ConversationManager:
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_chat_completions",
                    "model": "m",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "K",
                    "max_output_tokens": 5,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    manager = ConversationManager()
    manager.create(resolve_profile(config, None, {"K": "x"}))
    return manager


@pytest.mark.asyncio
async def test_chat_loop_renders_and_commits_only_completed_response() -> None:
    manager = _manager()
    ui = FakeUI()
    await ChatLoop(manager, FakeClient(), ui, "system", 5).run_turn("hi")
    assert ui.text == ["hello"]
    assert manager.conversation and len(manager.conversation.messages) == 2


@pytest.mark.asyncio
async def test_chat_loop_rolls_back_failed_response_but_keeps_usage() -> None:
    manager = _manager()
    ui = FakeUI()
    await ChatLoop(manager, FakeClient(True), ui, "system", 5).run_turn("hi")
    assert manager.conversation and manager.conversation.messages == []
    assert manager.conversation.usage_ledger.for_conversation().usage.total_tokens == 3

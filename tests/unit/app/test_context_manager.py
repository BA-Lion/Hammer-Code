from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hammer_code.app.context_manager import ContextManager, RecoveryState
from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import AppConfig, ContextConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    StopReason,
)
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import ContextCompactionError, TransportError
from hammer_code.llm.client import ClientCapabilities, ModelClient
from hammer_code.tools.runtime import RuntimeStore


class SummaryClient(ModelClient):
    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        message = Message(
            Role.ASSISTANT,
            (
                TextBlock("<"),
                TextBlock("analysis"),
                TextBlock(">plan</"),
                TextBlock("analysis"),
                TextBlock("><summary>compact "),
                TextBlock("state</summary>"),
            ),
        )
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                message,
                StopReason.END_TURN,
                TokenUsage(1, 1, status=UsageStatus.FINAL),
            ),
        )


class InvalidSummaryClient(SummaryClient):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, (TextBlock("summary without the required envelope"),)),
                StopReason.END_TURN,
                TokenUsage(1, 1, status=UsageStatus.FINAL),
            ),
        )


class FailingSummaryClient(SummaryClient):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        if not request.request_id:  # pragma: no cover - keeps this an async generator
            yield ResponseCompleted(
                request.request_id,
                ModelResponse(
                    "provider",
                    "model",
                    Message(Role.ASSISTANT, (TextBlock("unused"),)),
                    StopReason.END_TURN,
                    TokenUsage(1, 1, status=UsageStatus.FINAL),
                ),
            )
        raise TransportError("Model connection failed; check network settings")


def _manager(texts: tuple[str, ...] = ("one", "middle " * 200, "three")) -> ConversationManager:
    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_chat_completions",
                    "model": "model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "KEY",
                    "max_output_tokens": 100,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    manager = ConversationManager()
    manager.create(resolve_profile(config, None, {"KEY": "secret"}))
    for text in texts:
        turn = manager.begin_turn(text)
        manager.commit(turn, Message(Role.ASSISTANT, (TextBlock("answer " + text),)))
    return manager


@pytest.mark.asyncio
async def test_manual_compaction_replaces_only_committed_middle_history(tmp_path: Path) -> None:
    manager = _manager()
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        max_context_tokens=1000, compact_trigger_tokens=900, summary_target_tokens=100
    )
    context = ContextManager(
        manager,
        SummaryClient(),
        runtime,
        config,
        TokenEstimator(),
        RecoveryState(config),
        "base",
        100,
    )
    try:
        before = manager.snapshot_committed()
        result = await context.compact_now(mcp_prompt="", tools=())
        after = manager.snapshot_committed()
        assert result.compact_event is not None
        assert after[0] == before[0]
        summary_block = after[2].content[0]
        assert isinstance(summary_block, TextBlock)
        assert "Conversation summary" in summary_block.text
        assert TokenEstimator().estimate_messages(after) < TokenEstimator().estimate_messages(
            before
        )
        assert context.recovery_prompt == ""
    finally:
        runtime.cleanup()


@pytest.mark.asyncio
async def test_manual_compaction_without_middle_turn_is_a_noop(tmp_path: Path) -> None:
    manager = _manager(("one", "two"))
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        max_context_tokens=1000, compact_trigger_tokens=900, summary_target_tokens=100
    )
    context = ContextManager(
        manager,
        SummaryClient(),
        runtime,
        config,
        TokenEstimator(),
        RecoveryState(config),
        "base",
        100,
    )
    try:
        result = await context.compact_now(mcp_prompt="", tools=())
        assert result.compact_event is None
    finally:
        runtime.cleanup()


@pytest.mark.asyncio
async def test_manual_compaction_reports_last_failure_and_preserves_history(
    tmp_path: Path,
) -> None:
    manager = _manager()
    before = manager.snapshot_committed()
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        max_context_tokens=1000, compact_trigger_tokens=900, summary_target_tokens=100
    )
    client = InvalidSummaryClient()
    context = ContextManager(
        manager,
        client,
        runtime,
        config,
        TokenEstimator(),
        RecoveryState(config),
        "base",
        100,
    )
    try:
        with pytest.raises(
            ContextCompactionError,
            match=(
                "three compaction attempts failed; last reason: the model response did not "
                "contain one valid"
            ),
        ):
            await context.compact_now(mcp_prompt="", tools=())
        assert client.calls == 3
        assert manager.snapshot_committed() == before
    finally:
        runtime.cleanup()


@pytest.mark.asyncio
async def test_manual_compaction_reports_safe_model_request_failure(tmp_path: Path) -> None:
    manager = _manager()
    before = manager.snapshot_committed()
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        max_context_tokens=1000, compact_trigger_tokens=900, summary_target_tokens=100
    )
    client = FailingSummaryClient()
    context = ContextManager(
        manager,
        client,
        runtime,
        config,
        TokenEstimator(),
        RecoveryState(config),
        "base",
        100,
    )
    try:
        with pytest.raises(
            ContextCompactionError,
            match=(
                "last reason: the summary request failed: Model connection failed; "
                "check network settings"
            ),
        ):
            await context.compact_now(mcp_prompt="", tools=())
        assert client.calls == 3
        assert manager.snapshot_committed() == before
    finally:
        runtime.cleanup()

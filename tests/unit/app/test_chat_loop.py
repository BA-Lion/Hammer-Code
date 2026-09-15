import asyncio
from collections.abc import AsyncIterator

import pytest

from hammer_code.app.agent import PrimaryAgent
from hammer_code.app.context_manager import ContextPreparation
from hammer_code.app.context_window import ContextWindow
from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ResponseCompleted,
    ResponseStarted,
    StopReason,
    TextDelta,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import (
    Message,
    ReasoningVisibility,
    Role,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import TransportError
from hammer_code.llm.client import ClientCapabilities, ModelClient
from hammer_code.permissions.models import ApprovalChoice
from hammer_code.tools.builtin.files import ReadFileTool
from hammer_code.tools.executor import ToolBatchCancelled
from hammer_code.tools.registry import ToolRegistry


class FakeUI:
    def __init__(self) -> None:
        self.text = []
        self.errors = []
        self.shown_usage = []
        self.reasoning_status_count = 0
        self.reasoning = []

    async def prompt(self) -> str:
        return "/exit"

    def text_delta(self, text: str) -> None:
        self.text.append(text)

    def reasoning_status(self) -> None:
        self.reasoning_status_count += 1

    def reasoning_delta(self, text, visibility) -> None:
        self.reasoning.append((text, visibility))

    def tool_call_notice(self, call) -> None:
        pass

    def error(self, message: str) -> None:
        self.errors.append(message)

    def usage(self, summary) -> None:
        self.shown_usage.append(summary)

    def help(self) -> None:
        pass

    async def approve(self, request, reason: str) -> ApprovalChoice:
        return ApprovalChoice.DENY

    async def confirm(self, prompt: str) -> bool:
        return False

    async def choose(self, prompt: str, options: tuple[str, ...]) -> str | None:
        return None

    def info(self, message: str) -> None:
        pass

    def mcp_status(self, name: str, status: str, detail: str | None = None) -> None:
        pass

    def compact(self, event) -> None:
        pass

    def sessions(self, items: tuple[object, ...]) -> None:
        pass

    def persistence_warning(self, message: str, *, final: bool = False) -> None:
        self.errors.append(message)

    def memory_warning(self, message: str) -> None:
        self.errors.append(message)


class FakeClient(ModelClient):
    def __init__(
        self,
        fail: bool = False,
        fail_before_events: bool = False,
        reasoning_chunks: tuple[str, ...] = (),
    ) -> None:
        self.fail = fail
        self.fail_before_events = fail_before_events
        self.reasoning_chunks = reasoning_chunks

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if self.fail_before_events:
            raise TransportError("network failure")
        yield ResponseStarted(request.request_id, "provider", "model")
        for chunk in self.reasoning_chunks:
            yield ReasoningDelta(request.request_id, 0, chunk, ReasoningVisibility.SUMMARY)
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


class ToolCallClient(ModelClient):
    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        call = ToolCallBlock("call", "read_file", {"path": "a.txt"}, '{"path":"a.txt"}')
        yield ResponseStarted(request.request_id, "provider", "model")
        yield ToolCallCompleted(request.request_id, 0, call)
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, (call,)),
                StopReason.TOOL_CALL,
                TokenUsage(1, 1, status=UsageStatus.FINAL),
            ),
        )


class CancelledExecutor:
    async def execute_batch(self, calls: tuple[ToolCallBlock, ...]) -> tuple[ToolResultBlock, ...]:
        raise ToolBatchCancelled(
            (ToolResultBlock(calls[0].call_id, (TextBlock("cancelled"),), True),)
        )


class _NonCompactingContext:
    recovery_prompt = ""

    async def prepare_before_request(self, **kwargs) -> ContextPreparation:
        assert kwargs["before_compaction"] is not None
        return ContextPreparation()


class _MemoryMustNotFlush:
    async def flush_before_compaction(self) -> bool:
        raise AssertionError("ordinary main requests must not flush Memory")

    async def read_index_prompt(self) -> str:
        return ""


class _FailedCompactionPersistence:
    persistence_degraded = False

    async def rewrite_after_compaction(self, *args, **kwargs) -> None:
        self.persistence_degraded = True


@pytest.mark.asyncio
async def test_ordinary_main_request_does_not_wait_for_memory_flush() -> None:
    manager = _manager()
    await PrimaryAgent(
        manager,
        FakeClient(),
        FakeUI(),
        "system",
        5,
        context_manager=_NonCompactingContext(),  # type: ignore[arg-type]
        memory_service=_MemoryMustNotFlush(),  # type: ignore[arg-type]
    ).run_turn("hi")
    assert manager.conversation and len(manager.conversation.messages) == 2


@pytest.mark.asyncio
async def test_compaction_persistence_failure_is_reported_immediately() -> None:
    manager = _manager()
    ui = FakeUI()
    coordinator = _FailedCompactionPersistence()
    loop = PrimaryAgent(
        manager,
        FakeClient(),
        ui,
        "system",
        5,
        session_coordinator=coordinator,  # type: ignore[arg-type]
    )

    await loop._persist_preparation(
        manager.snapshot_committed(),
        ContextPreparation(
            history_replaced=True,
            summary="summary",
            summarized_turn_positions=(1,),
        ),
    )

    assert ui.errors == ["Context compacted in memory, but compacted history could not be saved."]


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
async def test_primary_agent_renders_and_commits_only_completed_response() -> None:
    manager = _manager()
    ui = FakeUI()
    await PrimaryAgent(manager, FakeClient(), ui, "system", 5).run_turn("hi")
    assert ui.text == ["hello"]
    assert manager.conversation and len(manager.conversation.messages) == 2


@pytest.mark.asyncio
async def test_primary_agent_announces_hidden_reasoning_only_once_per_turn() -> None:
    manager = _manager()
    ui = FakeUI()
    await PrimaryAgent(
        manager,
        FakeClient(reasoning_chunks=("one", "two", "three")),
        ui,
        "system",
        5,
    ).run_turn("hi")
    assert ui.reasoning_status_count == 1
    assert ui.reasoning == []


@pytest.mark.asyncio
async def test_primary_agent_streams_real_reasoning_when_enabled() -> None:
    manager = _manager()
    ui = FakeUI()
    await PrimaryAgent(
        manager,
        FakeClient(reasoning_chunks=("one", "two")),
        ui,
        "system",
        5,
        show_reasoning=True,
    ).run_turn("hi")
    assert ui.reasoning_status_count == 0
    assert ui.reasoning == [
        ("one", ReasoningVisibility.SUMMARY),
        ("two", ReasoningVisibility.SUMMARY),
    ]


@pytest.mark.asyncio
async def test_primary_agent_rolls_back_failed_response_but_keeps_usage() -> None:
    manager = _manager()
    ui = FakeUI()
    await PrimaryAgent(manager, FakeClient(True), ui, "system", 5).run_turn("hi")
    assert manager.conversation and manager.conversation.messages == []
    summary = manager.conversation.usage_ledger.for_conversation()
    assert summary.usage.total_tokens == 3
    assert summary.final_requests == 1
    assert summary.unavailable_requests == 0


@pytest.mark.asyncio
async def test_primary_agent_recovers_after_failure_before_first_event() -> None:
    manager = _manager()
    ui = FakeUI()
    loop = PrimaryAgent(manager, FakeClient(fail_before_events=True), ui, "system", 5)
    await loop.run_turn("hi")
    assert manager.conversation and manager.conversation.messages == []
    summary = manager.conversation.usage_ledger.for_conversation()
    assert summary.usage.total_tokens is None
    assert summary.unavailable_requests == 1
    assert ui.errors == ["network failure"]

    loop.client = FakeClient()
    await loop.run_turn("hello again")
    assert manager.conversation.messages and len(manager.conversation.messages) == 2


@pytest.mark.asyncio
async def test_cancelled_tool_batch_keeps_the_completed_exchange() -> None:
    manager = _manager()
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    loop = PrimaryAgent(
        manager,
        ToolCallClient(),
        FakeUI(),
        "system",
        5,
        registry=registry,
        executor=CancelledExecutor(),  # type: ignore[arg-type]
    )

    with pytest.raises(asyncio.CancelledError):
        await loop.run_turn("inspect")
    assert manager.conversation is not None
    assert [message.role for message in manager.conversation.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]


class _DeferredTool(ReadFileTool):
    name = "deferred"
    should_defer = True


class _McpState:
    prompt: str

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class _PromptChangingExecutor:
    def __init__(self, registry: ToolRegistry, mcp: _McpState) -> None:
        self.registry = registry
        self.mcp = mcp

    async def execute_batch(self, calls: tuple[ToolCallBlock, ...]) -> tuple[ToolResultBlock, ...]:
        self.registry.discover("deferred")
        self.mcp.prompt = "# MCP\n\nOverall status: connected\n"
        return (ToolResultBlock(calls[0].call_id, (TextBlock("found"),), False),)


class _TwoRequestClient(ModelClient):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ResponseStarted(request.request_id, "provider", "model")
        if len(self.requests) == 1:
            call = ToolCallBlock("search", "toolSearch", {"query": "find"}, '{"query":"find"}')
            yield ToolCallCompleted(request.request_id, 0, call)
            yield ResponseCompleted(
                request.request_id,
                ModelResponse(
                    "provider",
                    "model",
                    Message(Role.ASSISTANT, (call,)),
                    StopReason.TOOL_CALL,
                    TokenUsage(1, 1, status=UsageStatus.FINAL),
                ),
            )
            return
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, (TextBlock("done"),)),
                StopReason.END_TURN,
                TokenUsage(1, 1, status=UsageStatus.FINAL),
            ),
        )


@pytest.mark.asyncio
async def test_next_request_uses_new_context_snapshot_after_tool_execution() -> None:
    manager = _manager()
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(_DeferredTool())
    client = _TwoRequestClient()
    mcp = _McpState("# MCP\n\nOverall status: connecting\n")
    loop = PrimaryAgent(
        manager,
        client,
        FakeUI(),
        "base",
        5,
        registry=registry,
        executor=_PromptChangingExecutor(registry, mcp),  # type: ignore[arg-type]
        context_window=ContextWindow("base"),
        mcp_manager=mcp,  # type: ignore[arg-type]
    )
    await loop.run_turn("find")
    assert "connecting" in client.requests[0].system_prompt
    assert "connected" in client.requests[1].system_prompt
    assert "connecting" in client.requests[0].system_prompt
    assert "deferred" not in {tool.name for tool in client.requests[0].tools}
    assert "deferred" in {tool.name for tool in client.requests[1].tools}

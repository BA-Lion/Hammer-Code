from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.domain.events import (
    ModelEvent,
    ModelResponse,
    ResponseCompleted,
    StopReason,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.llm.client import ClientCapabilities, ModelClient
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.rules import RuleStore
from hammer_code.permissions.service import PermissionService
from hammer_code.subagent.models import (
    ParentRequestSnapshot,
    SubagentCatalogSnapshot,
    SubagentContext,
    SubagentExecution,
    SubagentInvocation,
    SubagentSource,
)
from hammer_code.subagent.runner import SubagentRunner
from hammer_code.subagent.usage import SubagentUsageTracker
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.builtin.shell import sanitized_environment
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore


class _Client(ModelClient):
    def __init__(self, reason: StopReason, text: str) -> None:
        self.reason, self.text = reason, text

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        return None

    async def _events(self, request_id: str) -> AsyncIterator[ModelEvent]:
        response = ModelResponse(
            "response",
            "fake",
            Message(Role.ASSISTANT, (TextBlock(self.text),)),
            self.reason,
            TokenUsage(1, 1, status=UsageStatus.FINAL),
        )
        yield UsageUpdated(request_id, TokenUsage(1, 1))
        yield ResponseCompleted(request_id, response)

    def stream(self, request) -> AsyncIterator[ModelEvent]:
        return self._events(request.request_id)


def _invocation(max_iterations: int = 1) -> SubagentInvocation:
    parent = ParentRequestSnapshot(
        SubagentCatalogSnapshot(1, "f", {}), "system", (), (), frozenset(), "base", "project", "mcp"
    )
    return SubagentInvocation(
        SubagentSource.DYNAMIC,
        "dynamic",
        "task",
        "prompt",
        SubagentContext.ISOLATED,
        SubagentExecution.INLINE,
        None,
        (),
        max_iterations,
        parent,
    )


@pytest.mark.asyncio
async def test_runner_returns_only_visible_end_turn_text_and_clips_it(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path)
    permissions = PermissionService(
        PermissionChecker(PermissionMode.DEFAULT, RuleStore(tmp_path)),
        _Approval(),
        RuleStore(tmp_path),
    )
    usages: list[tuple[str, str, TokenUsage]] = []
    runner = SubagentRunner(
        _Client(StopReason.END_TURN, "x" * 33000),
        runtime,
        100,
        permissions,
        lambda operation_id, request_id, usage: usages.append((operation_id, request_id, usage)),
    )
    context = ToolExecutionContext(tmp_path, tmp_path, runtime.session_dir, sanitized_environment())

    tracker = SubagentUsageTracker(1000)
    result = await runner.run(_invocation(), ToolRegistry(), context, "subagent:test", tracker)

    assert not result.is_error
    assert len(result.content) <= 32000
    assert result.content.endswith("[Subagent result truncated.]")
    assert len(usages) == 2
    assert usages[0][0] == "subagent:test"
    assert tracker.snapshot().reported_total == 2


@pytest.mark.asyncio
async def test_runner_rejects_non_normal_stop_reason(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path)
    permissions = PermissionService(
        PermissionChecker(PermissionMode.DEFAULT, RuleStore(tmp_path)),
        _Approval(),
        RuleStore(tmp_path),
    )
    runner = SubagentRunner(
        _Client(StopReason.MAX_TOKENS, "partial"), runtime, 100, permissions, lambda *_: None
    )
    context = ToolExecutionContext(tmp_path, tmp_path, runtime.session_dir, sanitized_environment())

    result = await runner.run(
        _invocation(),
        ToolRegistry(),
        context,
        "subagent:test",
        SubagentUsageTracker(1000),
    )

    assert result.is_error
    assert "did not complete normally" in result.content


class _Approval:
    async def approve(self, request, reason):
        del request, reason
        raise AssertionError("No tool approval expected")


class _FixedEstimator(TokenEstimator):
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def estimate_request(self, system_prompt, messages, tools) -> int:
        del system_prompt, messages, tools
        return self.values.pop(0)


class _SequenceClient(ModelClient):
    def __init__(self, responses: tuple[ModelResponse, ...]) -> None:
        self.responses = responses
        self.requests = []

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        return None

    async def _events(self, request) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        for index, block in enumerate(response.message.content):
            if isinstance(block, ToolCallBlock):
                yield ToolCallCompleted(request.request_id, index, block)
        yield UsageUpdated(request.request_id, response.usage)
        yield ResponseCompleted(request.request_id, response)

    def stream(self, request) -> AsyncIterator[ModelEvent]:
        return self._events(request)


class _NoArguments(BaseModel):
    pass


class _EchoTool(Tool):
    name = "echo"
    description = "Return a fixed test value."
    input_model = _NoArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(self, context, arguments) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult("echoed")


def _response(
    reason: StopReason,
    usage: TokenUsage,
    *content: TextBlock | ToolCallBlock,
) -> ModelResponse:
    return ModelResponse("response", "fake", Message(Role.ASSISTANT, content), reason, usage)


def _runner(
    tmp_path: Path, client: ModelClient, estimator: TokenEstimator
) -> tuple[SubagentRunner, ToolExecutionContext]:
    runtime = RuntimeStore(tmp_path)
    permissions = PermissionService(
        PermissionChecker(PermissionMode.UNATTENDED, RuleStore(tmp_path)),
        _Approval(),
        RuleStore(tmp_path),
    )
    return (
        SubagentRunner(client, runtime, 100, permissions, lambda *_: None, estimator),
        ToolExecutionContext(tmp_path, tmp_path, runtime.session_dir, sanitized_environment()),
    )


@pytest.mark.asyncio
async def test_runner_refuses_before_client_call_and_applies_dynamic_output_limit(
    tmp_path: Path,
) -> None:
    rejected_client = _SequenceClient(
        (_response(StopReason.END_TURN, TokenUsage(1, 1), TextBlock("unused")),)
    )
    rejected, context = _runner(tmp_path, rejected_client, _FixedEstimator(10))
    rejected_result = await rejected.run(
        _invocation(),
        ToolRegistry(),
        context,
        "subagent:reject",
        SubagentUsageTracker(10),
    )
    assert rejected_result.is_error
    assert "budget exhausted" in rejected_result.content
    assert rejected_client.requests == []

    client = _SequenceClient(
        (
            _response(
                StopReason.END_TURN,
                TokenUsage(3, 2, status=UsageStatus.FINAL),
                TextBlock("done"),
            ),
        )
    )
    runner, context = _runner(tmp_path, client, _FixedEstimator(3))
    tracker = SubagentUsageTracker(10)
    result = await runner.run(_invocation(), ToolRegistry(), context, "subagent:limited", tracker)
    assert result.content == "done"
    assert client.requests[0].max_output_tokens == 7
    assert tracker.snapshot().accounted_tokens == 5


@pytest.mark.asyncio
async def test_runner_accumulates_rounds_and_stops_after_tool_call_exhausts_budget(
    tmp_path: Path,
) -> None:
    call = ToolCallBlock("call", "echo", {}, "{}")
    client = _SequenceClient(
        (
            _response(
                StopReason.TOOL_CALL,
                TokenUsage(2, 1, status=UsageStatus.FINAL),
                call,
            ),
            _response(
                StopReason.END_TURN,
                TokenUsage(3, 2, status=UsageStatus.FINAL),
                TextBlock("complete"),
            ),
        )
    )
    runner, context = _runner(tmp_path, client, _FixedEstimator(1, 2))
    tracker = SubagentUsageTracker(10)
    registry = ToolRegistry()
    registry.register(_EchoTool())

    result = await runner.run(_invocation(2), registry, context, "subagent:rounds", tracker)

    assert result.content == "complete"
    assert [request.max_output_tokens for request in client.requests] == [9, 5]
    assert tracker.snapshot().reported_total == 8

    exhausted_client = _SequenceClient(
        (
            _response(
                StopReason.TOOL_CALL,
                TokenUsage(8, 2, status=UsageStatus.FINAL),
                call,
            ),
        )
    )
    exhausted, context = _runner(tmp_path, exhausted_client, _FixedEstimator(1))
    result = await exhausted.run(
        _invocation(),
        registry,
        context,
        "subagent:exhausted",
        SubagentUsageTracker(10),
    )
    assert result.is_error
    assert "budget exhausted" in result.content
    assert len(exhausted_client.requests) == 1


@pytest.mark.asyncio
async def test_runner_keeps_complete_end_turn_when_reported_usage_exceeds_budget(
    tmp_path: Path,
) -> None:
    client = _SequenceClient(
        (
            _response(
                StopReason.END_TURN,
                TokenUsage(8, 4, status=UsageStatus.FINAL),
                TextBlock("paid result"),
            ),
        )
    )
    runner, context = _runner(tmp_path, client, _FixedEstimator(1))
    tracker = SubagentUsageTracker(10)

    result = await runner.run(_invocation(), ToolRegistry(), context, "subagent:over", tracker)

    assert result.content == "paid result"
    assert not result.is_error
    assert tracker.snapshot().exceeded


@pytest.mark.asyncio
async def test_runner_keeps_conservative_charge_when_provider_usage_is_unavailable(
    tmp_path: Path,
) -> None:
    client = _SequenceClient(
        (
            _response(
                StopReason.END_TURN,
                TokenUsage(None, None, status=UsageStatus.UNAVAILABLE),
                TextBlock("done"),
            ),
        )
    )
    runner, context = _runner(tmp_path, client, _FixedEstimator(2))
    tracker = SubagentUsageTracker(10)

    result = await runner.run(_invocation(), ToolRegistry(), context, "subagent:unknown", tracker)

    assert result.content == "done"
    assert tracker.snapshot().reported_total is None
    assert tracker.snapshot().accounted_tokens == 10
    assert tracker.snapshot().unavailable_requests == 1

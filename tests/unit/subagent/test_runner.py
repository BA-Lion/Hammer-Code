from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hammer_code.domain.events import (
    ModelEvent,
    ModelResponse,
    ResponseCompleted,
    StopReason,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage
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
from hammer_code.tools.base import ToolExecutionContext
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
            TokenUsage(1, 1),
        )
        yield UsageUpdated(request_id, TokenUsage(1, 1))
        yield ResponseCompleted(request_id, response)

    def stream(self, request) -> AsyncIterator[ModelEvent]:
        return self._events(request.request_id)


def _invocation() -> SubagentInvocation:
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
        1,
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

    result = await runner.run(_invocation(), ToolRegistry(), context)

    assert not result.is_error
    assert len(result.content) <= 32000
    assert result.content.endswith("[Subagent result truncated.]")
    assert len(usages) == 2
    assert usages[0][0].startswith("subagent-")


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

    result = await runner.run(_invocation(), ToolRegistry(), context)

    assert result.is_error
    assert "did not complete normally" in result.content


class _Approval:
    async def approve(self, request, reason):
        del request, reason
        raise AssertionError("No tool approval expected")

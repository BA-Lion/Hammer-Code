import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from hammer_code.domain.messages import ToolCallBlock
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import ApprovalChoice, PermissionMode, PermissionRequest
from hammer_code.permissions.service import PermissionService
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.builtin.files import ReadFileTool
from hammer_code.tools.executor import ToolBatchCancelled, ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore


class AllowApproval:
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice:
        return ApprovalChoice.ALLOW_ONCE


class _NoArguments(BaseModel):
    pass


class SlowReadTool(Tool):
    name = "slow_read"
    description = "Test double that waits until cancelled."
    input_model = _NoArguments
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("The test tool must be cancelled")


@pytest.mark.asyncio
async def test_executor_keeps_call_order_and_reports_unknown(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    runtime = RuntimeStore(tmp_path)
    try:
        executor = ToolExecutor(
            registry,
            PermissionService(PermissionChecker(PermissionMode.DEFAULT), AllowApproval(), None),
            ToolExecutionContext(tmp_path, tmp_path, runtime.session_dir, {}),
            runtime,
        )
        results = await executor.execute_batch(
            (
                ToolCallBlock("one", "read_file", {"path": "a.txt"}, '{"path":"a.txt"}'),
                ToolCallBlock("two", "missing", {}, "{}"),
            )
        )
        assert [result.call_id for result in results] == ["one", "two"]
        assert not results[0].is_error
        assert results[1].is_error
    finally:
        runtime.cleanup()


@pytest.mark.asyncio
async def test_cancelled_batch_returns_a_result_for_every_call(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = SlowReadTool()
    registry.register(tool)
    runtime = RuntimeStore(tmp_path)
    try:
        executor = ToolExecutor(
            registry,
            PermissionService(PermissionChecker(PermissionMode.DEFAULT), AllowApproval(), None),
            ToolExecutionContext(tmp_path, tmp_path, runtime.session_dir, {}),
            runtime,
        )
        calls = (
            ToolCallBlock("one", "slow_read", {}, "{}"),
            ToolCallBlock("two", "slow_read", {}, "{}"),
        )
        task = asyncio.create_task(executor.execute_batch(calls))
        await tool.started.wait()
        task.cancel()

        with pytest.raises(ToolBatchCancelled) as cancelled:
            await task
        assert [result.call_id for result in cancelled.value.results] == ["one", "two"]
        assert all(result.is_error for result in cancelled.value.results)
    finally:
        runtime.cleanup()

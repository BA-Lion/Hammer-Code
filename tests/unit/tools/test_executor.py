from pathlib import Path

import pytest

from hammer_code.domain.messages import ToolCallBlock
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import ApprovalChoice, PermissionMode, PermissionRequest
from hammer_code.permissions.service import PermissionService
from hammer_code.tools.base import ToolExecutionContext
from hammer_code.tools.builtin.files import ReadFileTool
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore


class AllowApproval:
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice:
        return ApprovalChoice.ALLOW_ONCE


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

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hammer_code.config import SubagentConfig
from hammer_code.subagent.models import BackgroundTaskStatus
from hammer_code.subagent.service import (
    BackgroundTaskManager,
    SubagentApprovalPort,
    format_background_results,
)
from hammer_code.tools.base import ToolExecutionResult


@pytest.mark.asyncio
async def test_background_tasks_are_bounded_and_drained_as_one_ordered_batch() -> None:
    manager = BackgroundTaskManager(SubagentConfig(max_background_tasks=2))

    async def successful(_task) -> ToolExecutionResult:
        return ToolExecutionResult("done")

    async def failing(_task) -> ToolExecutionResult:
        await asyncio.sleep(0)
        return ToolExecutionResult("failed safely", True)

    first = await manager.start("review", "first", successful)
    second = await manager.start("review", "second", failing)
    with pytest.raises(ValueError, match="capacity"):
        await manager.start("review", "third", successful)
    await manager.wait_idle()
    items = await manager.take_completed()

    assert {item.task_id for item in items} == {first.id, second.id}
    assert {item.status for item in items} == {
        BackgroundTaskStatus.COMPLETED,
        BackgroundTaskStatus.FAILED,
    }
    rendered = format_background_results(items)
    assert rendered.count("--- subagent-result") == 2
    assert '"source":"subagent"' in rendered


@pytest.mark.asyncio
async def test_requeue_preserves_a_failed_precommit_batch() -> None:
    manager = BackgroundTaskManager(SubagentConfig())

    async def completed(_task) -> ToolExecutionResult:
        return ToolExecutionResult("result")

    task = await manager.start("dynamic", "work", completed)
    await manager.wait_idle()
    batch = await manager.take_completed()
    await manager.requeue_front(batch)

    restored = await manager.take_completed()
    assert restored == batch
    assert restored[0].task_id == task.id


@pytest.mark.asyncio
async def test_task_lookup_requires_an_unambiguous_eight_character_prefix() -> None:
    manager = BackgroundTaskManager(SubagentConfig())

    async def waiting(_task) -> ToolExecutionResult:
        await asyncio.Future()
        return ToolExecutionResult("unreachable")

    task = await manager.start("dynamic", "work", waiting)
    with pytest.raises(ValueError, match="at least 8"):
        await manager.get(task.id[:7])
    assert (await manager.get(task.id[:8])).id == task.id
    await manager.cancel(task.id)
    await manager.wait_idle()
    assert (await manager.get(task.id)).status is BackgroundTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_background_approval_is_labeled_and_marks_the_task_waiting(tmp_path: Path) -> None:
    manager = BackgroundTaskManager(SubagentConfig())
    entered = asyncio.Event()
    release = asyncio.Event()

    class Approval:
        async def approve(self, request, reason: str):
            del request
            assert reason.startswith("[subagent=review task=")
            entered.set()
            await release.wait()
            from hammer_code.permissions.models import ApprovalChoice

            return ApprovalChoice.ALLOW_ONCE

    async def operation(task) -> ToolExecutionResult:
        port = SubagentApprovalPort(Approval(), agent="review", task_id=task.id, tasks=manager)
        from hammer_code.permissions.models import PermissionRequest

        await port.approve(
            PermissionRequest.for_command("shell", "Get-Date", tmp_path, tmp_path),
            "tool requires approval",
        )
        return ToolExecutionResult("done")

    task = await manager.start("review", "work", operation)
    await entered.wait()
    assert (await manager.get(task.id)).status is BackgroundTaskStatus.WAITING_APPROVAL
    release.set()
    await manager.wait_idle()
    assert (await manager.get(task.id)).status is BackgroundTaskStatus.COMPLETED

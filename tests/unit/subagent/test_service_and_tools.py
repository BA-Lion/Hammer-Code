from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import hammer_code.subagent.service as service_module
from hammer_code.config import SubagentConfig
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.subagent.models import (
    BackgroundTaskStatus,
    ParentRequestSnapshot,
    SubagentCatalogSnapshot,
    SubagentExecution,
)
from hammer_code.subagent.prompts import build_subagent_catalog_prompt
from hammer_code.subagent.service import (
    BackgroundTaskManager,
    SubagentApprovalPort,
    SubagentService,
    format_background_results,
)
from hammer_code.subagent.tool import RunSubagentArguments, RunSubagentTool
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult
from hammer_code.tools.registry import ToolRegistry


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
    await manager.cancel(task.id)
    await manager.wait_idle()
    assert (await manager.get(task.id)).status is BackgroundTaskStatus.CANCELLED
    completed = await manager.take_completed()
    assert len(completed) == 1
    assert completed[0].status is BackgroundTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_task_lookup_rejects_unknown_and_ambiguous_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            "aaaaaaaa-0000-0000-0000-000000000002",
        )
    )
    monkeypatch.setattr(service_module, "uuid4", lambda: next(ids))
    manager = BackgroundTaskManager(SubagentConfig())

    async def waiting(_task) -> ToolExecutionResult:
        await asyncio.Event().wait()
        raise AssertionError("cancelled tasks must not resume")

    await manager.start("dynamic", "one", waiting)
    await manager.start("dynamic", "two", waiting)
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        await manager.get("bbbbbbbb")
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        await manager.get("aaaaaaaa")
    await manager.cancel_all(close=True, discard_results=True)


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


@pytest.mark.asyncio
async def test_task_queries_return_bounded_immutable_usage_snapshots_without_draining() -> None:
    manager = BackgroundTaskManager(SubagentConfig(max_task_tokens=1_000))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def operation(task) -> ToolExecutionResult:
        assert task.operation_id == f"subagent:{task.id}"
        assert task.usage_tracker.reserve("request", 100, 200) == 200
        task.usage_tracker.observe("request", TokenUsage(80, 20, status=UsageStatus.FINAL))
        entered.set()
        await release.wait()
        return ToolExecutionResult("x" * 25_000)

    task = await manager.start("review", "t" * 1_000, operation)
    await entered.wait()
    running = await manager.get(task.id[:8])

    assert running.task == "t" * 500
    assert running.usage.reported_total == 100
    assert await manager.take_completed() == ()

    release.set()
    await manager.wait_idle()
    completed = await manager.get(task.id)
    assert completed.status is BackgroundTaskStatus.COMPLETED
    assert completed.result == "x" * 20_000
    assert len(await manager.take_completed()) == 1


def test_model_surface_has_only_run_subagent_and_documents_execution_modes() -> None:
    registry = ToolRegistry(("subagent_task",))
    registry.register(RunSubagentTool())

    assert registry.exposed_names() == ("run_subagent",)
    assert "inline" in registry.definitions()[0].description
    assert "background" in registry.definitions()[0].description
    prompt = build_subagent_catalog_prompt(SubagentCatalogSnapshot(0, "", {}))
    assert "multiple inline" in prompt
    assert "Do not wait or poll" in prompt
    assert "subagent_task" not in prompt
    with pytest.raises(ValueError):
        RunSubagentArguments.model_validate(
            {"task": "work", "prompt": "p", "max_task_tokens": 999_999}
        )


@pytest.mark.asyncio
async def test_service_gives_inline_and_background_independent_configured_budgets(
    tmp_path: Path,
) -> None:
    config = SubagentConfig(max_task_tokens=1_234)
    tasks = BackgroundTaskManager(config)
    service = SubagentService(None, config, tasks)  # type: ignore[arg-type]
    seen: list[tuple[str, int]] = []

    class Runner:
        permissions = SimpleNamespace(approvals=object())

        async def run(
            self,
            invocation,
            registry,
            context,
            operation_id,
            usage_tracker,
            approval_port,
        ) -> ToolExecutionResult:
            del invocation, registry, context, approval_port
            seen.append((operation_id, usage_tracker.limit))
            return ToolExecutionResult("done")

    parent = ParentRequestSnapshot(
        SubagentCatalogSnapshot(0, "", {}),
        "system",
        (),
        (),
        frozenset(),
        "base",
        "project",
        "mcp",
    )
    service.bind_runtime(
        Runner(),  # pyright: ignore[reportArgumentType]
        ToolRegistry(),
        ToolExecutionContext(tmp_path, tmp_path, tmp_path, {}),
    )
    service.bind_request(parent)

    inline = await service.invoke(RunSubagentArguments(task="inline", prompt="p"))
    background = await service.invoke(
        RunSubagentArguments(task="background", prompt="p", execution=SubagentExecution.BACKGROUND)
    )
    await tasks.wait_idle()

    assert inline.content == "done"
    assert background.content.startswith("Subagent task started:")
    assert seen[0][0].startswith("subagent:inline:")
    assert seen[1][0].startswith("subagent:") and not seen[1][0].startswith("subagent:inline:")
    assert [limit for _, limit in seen] == [1_234, 1_234]

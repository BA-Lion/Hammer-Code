from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from hammer_code.worktree.manager import WorktreeManager


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Hammer Code Test")
    (root / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "base")
    return root


@pytest.mark.asyncio
async def test_snapshot_finalizes_and_integrates_without_changing_main_index(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    index_before = _git(root, "write-tree")
    manager = WorktreeManager(root)
    assert await manager.initialize()

    lease = await manager.create(str(uuid4()), Path("."))
    assert (lease.path / "tracked.txt").read_text(encoding="utf-8") == "dirty\n"
    assert (lease.path / "untracked.txt").read_text(encoding="utf-8") == "untracked\n"
    (lease.path / "child.txt").write_text("result\n", encoding="utf-8")

    metadata = await manager.finalize(lease.task_id)
    assert metadata.change_state == "pending_integration"
    assert "child.txt" in metadata.changed_files
    resolution = await manager.integrate(lease.task_id)
    assert resolution.change_state == "integrated"
    assert (root / "child.txt").read_text(encoding="utf-8") == "result\n"
    assert _git(root, "write-tree") == index_before
    await manager.close()


@pytest.mark.asyncio
async def test_conflict_is_retained_for_inspection_and_can_be_discarded(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manager = WorktreeManager(root)
    assert await manager.initialize()
    lease = await manager.create(str(uuid4()), Path("."))
    (lease.path / "tracked.txt").write_text("child\n", encoding="utf-8")
    await manager.finalize(lease.task_id)
    (root / "tracked.txt").write_text("main\n", encoding="utf-8")

    resolution = await manager.integrate(lease.task_id)
    assert resolution.change_state == "conflict"
    inspected = await manager.inspect(lease.task_id, "merge_inputs", ("tracked.txt",))
    assert inspected["inspection_id"]
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "main\n"
    discarded = await manager.discard(lease.task_id)
    assert discarded.change_state == "discarded"
    await manager.close()


@pytest.mark.asyncio
async def test_agent_merge_applies_non_conflicts_and_complete_conflict_decision(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manager = WorktreeManager(root)
    assert await manager.initialize()
    lease = await manager.create(str(uuid4()), Path("."))
    (lease.path / "tracked.txt").write_text("child\n", encoding="utf-8")
    (lease.path / "new-child.txt").write_text("non-conflicting\n", encoding="utf-8")
    await manager.finalize(lease.task_id)
    (root / "tracked.txt").write_text("main\n", encoding="utf-8")
    assert (await manager.integrate(lease.task_id)).change_state == "conflict"
    inputs = await manager.inspect(lease.task_id, "merge_inputs", ("tracked.txt",))
    files = inputs["files"]
    assert isinstance(files, dict)
    tracked = files["tracked.txt"]
    assert isinstance(tracked, dict)
    resolution = await manager.agent_merge(
        lease.task_id,
        str(inputs["inspection_id"]),
        (
            {
                "path": "tracked.txt",
                "expected_current_hash": tracked["current_hash"],
                "action": "write",
                "content": "merged\n",
            },
        ),
    )
    assert resolution.change_state == "integrated_by_primary"
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "merged\n"
    assert (root / "new-child.txt").read_text(encoding="utf-8") == "non-conflicting\n"
    await manager.close()


@pytest.mark.asyncio
async def test_child_worktree_integrates_only_into_its_parent_then_primary(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manager = WorktreeManager(root)
    assert await manager.initialize()
    parent = await manager.create(str(uuid4()), Path("."))
    child = await manager.create_child(str(uuid4()), parent.task_id, Path("."))
    (child.path / "nested.txt").write_text("child result\n", encoding="utf-8")

    child_metadata = await manager.finalize(child.task_id)
    assert child_metadata.change_state == "pending_integration"
    child_resolution = await manager.integrate(child.task_id)
    assert child_resolution.change_state == "integrated"
    assert (parent.path / "nested.txt").read_text(encoding="utf-8") == "child result\n"
    assert not (root / "nested.txt").exists()

    parent_metadata = await manager.finalize(parent.task_id)
    assert parent_metadata.change_state == "pending_integration"
    assert (await manager.integrate(parent.task_id)).change_state == "integrated"
    assert (root / "nested.txt").read_text(encoding="utf-8") == "child result\n"
    await manager.close()

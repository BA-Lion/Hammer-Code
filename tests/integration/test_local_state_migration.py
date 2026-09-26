from __future__ import annotations

from pathlib import Path

from hammer_code.local_state_migration import CURRENT_LOCAL_STATE_SCHEMA, migrate_local_state


def test_migration_only_clears_sessions_and_skill_state(tmp_path: Path) -> None:
    harness = tmp_path / ".hammer-code"
    for path in (
        harness / "sessions" / "old.json",
        harness / "skill" / "state" / "usage.json",
        harness / "skill" / "project" / "keep" / "SKILL.md",
        harness / "skill" / "history" / "keep.txt",
        harness / "skill" / "prune" / "keep.txt",
        harness / "memory" / "sentinel.txt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")
    assert migrate_local_state(tmp_path)
    assert not (harness / "sessions").exists()
    assert not (harness / "skill" / "state").exists()
    assert (harness / "skill" / "project" / "keep" / "SKILL.md").read_text() == "keep"
    assert (harness / "skill" / "history" / "keep.txt").read_text() == "keep"
    assert (harness / "skill" / "prune" / "keep.txt").read_text() == "keep"
    assert (harness / "memory" / "sentinel.txt").read_text() == "keep"
    assert not migrate_local_state(tmp_path)
    assert str(CURRENT_LOCAL_STATE_SCHEMA) in (harness / "local-state-schema.json").read_text()

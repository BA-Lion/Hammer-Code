from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hammer_code.skill.models import SkillDefinition, SkillScope, SkillSource
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite


@pytest.mark.asyncio
async def test_repository_publishes_written_skill_and_usage(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    initial, report = await repository.initialize()
    assert not initial.effective
    assert report.recovered == 0
    now = datetime.now(UTC).replace(microsecond=0)
    definition = SkillDefinition(
        "generated-guide",
        "A generated guide",
        "0.1.0",
        now,
        now,
        SkillSource.GENERATED,
        "Use this guide for {{arguments}}.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "generated-guide",
    )

    await repository.commit(
        PreparedSkillWrite("add", definition, None, "unit test", "test-operation")
    )
    snapshot = await repository.snapshot_for_turn()
    loaded = snapshot.effective["generated-guide"]
    await repository.record_retrieved((loaded.ref,))
    await repository.record_used(loaded.ref)

    usage = repository.store.load_usage()["project:generated-guide"]
    assert usage.retrieve == usage.used == 1
    assert loaded.source is SkillSource.GENERATED
    assert (loaded.skill_dir / "SKILL.md").is_file()
    summary = json.loads(repository.store.provenance_summary_path.read_text(encoding="utf-8"))
    assert summary["recent"][0]["frontmatter"]["name"] == "generated-guide"
    assert "prompt_template" not in summary["recent"][0]["frontmatter"]

    preimage = hashlib.sha256((loaded.skill_dir / "SKILL.md").read_bytes()).hexdigest()
    merged = SkillDefinition(
        loaded.name,
        loaded.description,
        "0.1.1",
        loaded.created_at,
        datetime.now(UTC).replace(microsecond=0),
        loaded.source,
        loaded.prompt_template + " Updated.",
        loaded.scope,
        loaded.skill_dir,
    )
    await repository.commit(
        PreparedSkillWrite("merge", merged, preimage, "update", "merge-operation")
    )
    updated_usage = repository.store.load_usage()["project:generated-guide"]
    assert updated_usage.retrieve == updated_usage.used == 1
    assert updated_usage.version == "0.1.1"
    assert updated_usage.version_retrieve == updated_usage.version_used == 0


@pytest.mark.asyncio
async def test_generated_skill_prune_is_recoverable(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    definition = SkillDefinition(
        "prune-me",
        "A generated Skill",
        "0.1.0",
        now,
        now,
        SkillSource.GENERATED,
        "Body.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "prune-me",
    )
    await repository.commit(PreparedSkillWrite("add", definition, None, "test", "add-prune"))
    snapshot = await repository.snapshot_for_turn()
    loaded = snapshot.effective["prune-me"]
    preimage = hashlib.sha256((loaded.skill_dir / "SKILL.md").read_bytes()).hexdigest()

    await repository.prune(loaded, preimage)

    assert "prune-me" not in (await repository.snapshot_for_turn()).effective
    pruned_at = next((repository.store.root / "prune" / "project" / "prune-me").iterdir()).name
    await repository.restore_pruned(loaded, pruned_at)
    assert "prune-me" in (await repository.snapshot_for_turn()).effective

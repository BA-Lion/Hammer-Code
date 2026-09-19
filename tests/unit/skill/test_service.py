from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hammer_code.config import ContextConfig, SkillConfig
from hammer_code.skill.models import (
    SkillContext,
    SkillDefinition,
    SkillObservation,
    SkillScope,
    SkillSource,
)
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.service import SkillInvocationService
from hammer_code.skill.store import PreparedSkillWrite


@pytest.mark.asyncio
async def test_model_invocation_returns_complete_rendered_skill_and_records_usage(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    definition = SkillDefinition(
        "review",
        "Review a change",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review {{arguments}} at {{skill_dir}}.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
    )
    await repository.commit(PreparedSkillWrite("add", definition, None, "test", "service-add"))
    snapshot = await repository.snapshot_for_turn()
    service = SkillInvocationService(repository, SkillConfig(), ContextConfig())
    service.begin_turn(snapshot, SkillObservation(snapshot.generation))

    result = await service.invoke("review", "the current diff")

    assert not result.is_error
    assert "the current diff" in result.content
    assert "{{arguments}}" not in result.content
    assert repository.store.load_usage()["project:review"].used == 1
    assert service.end_turn(completed=True).used == (definition.ref,)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_unavailable_fork_does_not_claim_skill_was_used(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    definition = SkillDefinition(
        "fork-review",
        "Review independently",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review {{arguments}}.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "fork-review",
        context=SkillContext.FORK,
    )
    await repository.commit(PreparedSkillWrite("add", definition, None, "test", "fork-add"))
    snapshot = await repository.snapshot_for_turn()
    service = SkillInvocationService(repository, SkillConfig(), ContextConfig())
    service.begin_turn(snapshot, SkillObservation(snapshot.generation))

    result = await service.invoke("fork-review", "the diff")

    assert result.is_error
    assert repository.store.load_usage()["project:fork-review"].used == 0

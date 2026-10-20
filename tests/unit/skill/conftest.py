from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hammer_code.skill.models import SkillDefinition, SkillScope, SkillSource
from hammer_code.skill.parser import serialize_skill


@pytest.fixture
def skill_factory(tmp_path: Path):
    def make(
        name: str,
        description: str = "workflow review",
        *,
        scope: SkillScope = SkillScope.PROJECT,
        body: str = "Complete current instructions.",
        invocable: bool = True,
        when: str | None = None,
    ) -> SkillDefinition:
        now = datetime(2026, 10, 1, tzinfo=UTC)
        directory = tmp_path / ".hammer-code/skill" / scope.value / name
        directory.mkdir(parents=True, exist_ok=True)
        definition = SkillDefinition(
            name,
            description,
            "0.1.0",
            now,
            now,
            SkillSource.MANUAL,
            body,
            scope,
            directory,
            model_invocable=invocable,
            when_to_use=when,
        )
        (directory / "SKILL.md").write_text(serialize_skill(definition), encoding="utf-8")
        return definition

    return make

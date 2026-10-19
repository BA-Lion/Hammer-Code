import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from hammer_code.skill.models import (
    CatalogSnapshot,
    ExtractedCandidate,
    SkillDefinition,
    SkillScope,
    SkillSource,
    empty_snapshot,
)
from hammer_code.skill.prompts import (
    build_correction_prompt,
    build_extractor_prompt,
    build_maintenance_prompt,
    build_skill_catalog_prompt,
)


def _catalog_skill(name: str, scope: SkillScope = SkillScope.PROJECT) -> SkillDefinition:
    stamp = datetime(2026, 10, 19, tzinfo=UTC)
    return SkillDefinition(
        name,
        f"Description for {scope.value}:{name}，支持中文。",
        "0.1.0",
        stamp,
        stamp,
        SkillSource.MANUAL,
        f"Private body for {scope.value}:{name}.",
        scope,
        Path("skills") / scope.value / name,
    )


def test_catalog_exposes_only_effective_callable_names_and_descriptions() -> None:
    project = _catalog_skill("review")
    shadowed = _catalog_skill("review", SkillScope.USER)
    another = _catalog_skill("analyze")
    hidden = replace(_catalog_skill("private"), model_invocable=False)
    items = (project, shadowed, another, hidden)
    snapshot = CatalogSnapshot(
        1,
        "fingerprint",
        {item.name: item for item in (project, another, hidden)},
        {(item.scope, item.name): item for item in items},
    )

    prompt = build_skill_catalog_prompt(snapshot)
    payload = prompt.split("<skill-catalog-json>\n", 1)[1].split("\n</skill-catalog-json>", 1)[0]
    assert json.loads(payload) == [
        {"name": "analyze", "description": another.description},
        {"name": "review", "description": project.description},
    ]
    assert "支持中文" in prompt
    assert shadowed.description not in prompt
    assert hidden.name not in payload
    assert all(item.prompt_template not in prompt for item in items)
    assert all(str(item.skill_dir) not in prompt for item in items)


def test_catalog_omits_empty_and_model_disabled_catalogs() -> None:
    assert build_skill_catalog_prompt(empty_snapshot()) == ""
    hidden = replace(_catalog_skill("private"), model_invocable=False)
    snapshot = CatalogSnapshot(
        1, "fingerprint", {hidden.name: hidden}, {(hidden.scope, hidden.name): hidden}
    )
    assert build_skill_catalog_prompt(snapshot) == ""


def test_extractor_prompt_supplies_its_strict_json_schema() -> None:
    prompt = build_extractor_prompt("one-off input", (), None)

    assert "Required output JSON Schema:" in prompt
    assert '"candidate_action"' in prompt
    assert '"additionalProperties": false' in prompt
    assert '"evaluations"' not in prompt


def test_maintenance_prompt_supplies_its_strict_json_schema() -> None:
    candidate = ExtractedCandidate(
        name="validation",
        description="Validate a synthetic input",
        body="Validate {{arguments}}.",
        evolution_note="Add deterministic validation",
    )
    schema_prompt = build_maintenance_prompt(candidate, (), None)

    assert "Required output JSON Schema:" in schema_prompt
    assert '"action"' in schema_prompt
    assert '"evolution_note"' in schema_prompt
    assert '"additionalProperties": false' in schema_prompt


def test_maintenance_prompt_treats_selected_target_as_update_or_discard() -> None:
    candidate = ExtractedCandidate(
        name="validation",
        description="Validate a synthetic input",
        body="Use the corrected validation rule.",
        evolution_note="Correct an obsolete rule",
    )

    prompt = build_maintenance_prompt(candidate, (), None)

    assert "historical states to improve, not authorities" in prompt
    assert "Conflict alone is not a reason to discard" in prompt
    assert "merge that exact target or discard" in prompt
    assert "never add or merge another target" in prompt
    assert "All target sources use the same policy" in prompt
    assert "discard duplicates, narrow, unstable, contradictory" not in prompt


def test_extractor_correction_repeats_schema_and_original_json() -> None:
    prompt = build_correction_prompt("Extractor", '{"candidate_action":"discard"}', "invalid")

    assert '"candidate_action"' in prompt
    assert "Original JSON text follows:" in prompt
    assert '{"candidate_action":"discard"}' in prompt

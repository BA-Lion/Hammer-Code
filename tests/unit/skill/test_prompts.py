from hammer_code.skill.models import ExtractedCandidate
from hammer_code.skill.prompts import (
    build_correction_prompt,
    build_extractor_prompt,
    build_maintenance_prompt,
)


def test_extractor_prompt_supplies_its_strict_json_schema() -> None:
    prompt = build_extractor_prompt("one-off input", (), None)

    assert "Required output JSON Schema:" in prompt
    assert '"candidate_action"' in prompt
    assert '"additionalProperties": false' in prompt
    assert '"evaluations"' in prompt


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

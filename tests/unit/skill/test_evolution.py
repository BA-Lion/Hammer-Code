from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hammer_code.config import SkillConfig, SkillEvolutionConfig
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    StopReason,
)
from hammer_code.domain.messages import (
    ContentBlock,
    Message,
    ProviderStateBlock,
    ReasoningBlock,
    ReasoningVisibility,
    RefusalBlock,
    Role,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from hammer_code.domain.usage import TokenUsage
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.skill.evolution import SkillEvolutionError, SkillEvolutionService
from hammer_code.skill.models import (
    MaintenanceOperation,
    SkillDefinition,
    SkillScope,
    SkillSource,
)
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite


class _Client:
    def __init__(
        self,
        responses: list[str | tuple[ContentBlock, ...] | Exception],
        stop_reasons: list[StopReason] | None = None,
    ) -> None:
        self.responses = deque(responses)
        self.stop_reasons = deque(stop_reasons or [])
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        blocks = (TextBlock(response),) if isinstance(response, str) else response
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, blocks),
                self.stop_reasons.popleft() if self.stop_reasons else StopReason.END_TURN,
                TokenUsage(1, 1),
            ),
        )


def _config() -> SkillConfig:
    return SkillConfig(evolution=SkillEvolutionConfig(enabled=True))


def _candidate(name: str = "release-check") -> str:
    return (
        '{"evaluations":[],"candidate_action":"candidate","candidate":'
        '{"name":"'
        + name
        + '","description":"Reusable release checklist","body":"Check {{arguments}}.",'
        '"when_to_use":"before a release","user_invocable":true,"model_invocable":true,'
        '"context":"inline","arguments_required":false,"argument_hint":null,'
        '"allowed_tools":null,"evolution_note":"Captured reusable release checks"},'
        '"discard_reason":null}'
    )


def _add() -> str:
    return (
        '{"action":"add","target":null,"skill":'
        '{"name":"release-check","description":"Reusable release checklist",'
        '"when_to_use":"before a release","user_invocable":true,"model_invocable":true,'
        '"context":"inline","arguments_required":false,"argument_hint":null,'
        '"allowed_tools":null,"body":"Check {{arguments}}."},'
        '"evolution_note":"Captured reusable release checks","reason":"No equivalent Skill exists"}'
    )


def _discard() -> str:
    return (
        '{"evaluations":[],"candidate_action":"discard","candidate":null,'
        '"discard_reason":"one-off"}'
    )


def _maintenance_discard(reason: str = "Candidate is too narrow") -> str:
    return (
        '{"action":"discard","target":null,"skill":null,"evolution_note":null,'
        f'"reason":"{reason}"}}'
    )


def _with_reasoning(text: str) -> tuple[ContentBlock, ...]:
    midpoint = len(text) // 2
    return (
        ReasoningBlock("internal analysis", ReasoningVisibility.SUMMARY),
        TextBlock(text[:midpoint]),
        ReasoningBlock("more internal analysis", ReasoningVisibility.VISIBLE),
        TextBlock(text[midpoint:]),
    )


@pytest.mark.asyncio
async def test_evolution_adds_one_generated_skill_from_a_separate_queue_item(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    client = _Client([_candidate(), _add()])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=lambda _: None,
    )

    service.schedule((), "please remember the release steps", None)
    await service.wait_idle()

    definition = (await repository.snapshot_for_turn()).effective["release-check"]
    assert definition.source is SkillSource.GENERATED
    assert len(client.requests) == 2
    assert "current-user-input-json" in client.requests[0].system_prompt


@pytest.mark.asyncio
async def test_evolution_uses_bounded_json_correction_without_committing_bad_output(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    client = _Client(
        [
            "not json",
            '{"evaluations":[],"candidate_action":"discard","candidate":null,"discard_reason":"one-off"}',
        ]
    )
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=lambda _: None,
    )

    service.schedule((), "temporary detail", None)
    await service.wait_idle()

    assert not (await repository.snapshot_for_turn()).effective
    assert len(client.requests) == 2
    assert "invalid" in client.requests[1].system_prompt


@pytest.mark.asyncio
async def test_explicit_feedback_can_update_the_exact_selected_target(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    original = SkillDefinition(
        "review",
        "Review changes",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Never perform a final safety check.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
    )
    await repository.commit(PreparedSkillWrite("add", original, None, "seed", "seed-review"))
    merge = (
        '{"action":"merge","target":{"scope":"project","name":"review"},"skill":'
        '{"name":"review","description":"Review changes","when_to_use":null,'
        '"user_invocable":true,"model_invocable":true,"context":"inline",'
        '"arguments_required":false,"argument_hint":null,"allowed_tools":null,'
        '"body":"Review {{arguments}} carefully and always perform a final safety check."},'
        '"evolution_note":"Replaced the obsolete safety rule","reason":"New evidence corrects it"}'
    )
    client = _Client([_candidate("review"), merge])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=lambda _: None,
    )

    await service.feedback("project:review", "add a final safety check")
    await service.wait_idle()

    updated = (await repository.snapshot_for_turn()).by_identity[(SkillScope.PROJECT, "review")]
    assert updated.version == "0.1.1"
    assert updated.source is SkillSource.MANUAL
    assert "always perform" in updated.prompt_template
    assert "Never perform" not in updated.prompt_template
    assert "Selected update target" in client.requests[1].system_prompt
    assert "merge that exact target or discard" in client.requests[1].system_prompt


@pytest.mark.asyncio
async def test_explicit_feedback_may_discard_the_selected_target_without_modifying_it(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    original = SkillDefinition(
        "review",
        "Review changes",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review carefully.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
    )
    await repository.commit(PreparedSkillWrite("add", original, None, "seed", "seed-review"))
    warnings: list[str] = []
    client = _Client([_candidate("review"), _maintenance_discard("Feedback is ambiguous")])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    await service.feedback("project:review", "maybe make this shorter")
    await service.wait_idle()

    unchanged = (await repository.snapshot_for_turn()).by_identity[(SkillScope.PROJECT, "review")]
    provenance = [
        json.loads(line)
        for line in repository.store.provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    assert unchanged.version == "0.1.0"
    assert unchanged.prompt_template == "Review carefully."
    assert provenance[-1]["action"] == "discard"
    assert provenance[-1]["status"] == "discarded"
    assert warnings == []


@pytest.mark.asyncio
async def test_explicit_feedback_cannot_add_when_an_update_target_is_selected(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    original = SkillDefinition(
        "review",
        "Review changes",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review carefully.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
    )
    await repository.commit(PreparedSkillWrite("add", original, None, "seed", "seed-review"))
    provenance_before = repository.store.provenance_path.read_text(encoding="utf-8")
    warnings: list[str] = []
    client = _Client([_candidate("review"), _add()])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    await service.feedback("project:review", "create a separate copy")
    await service.wait_idle()

    snapshot = await repository.snapshot_for_turn()
    assert snapshot.by_identity[(SkillScope.PROJECT, "review")].version == "0.1.0"
    assert "release-check" not in snapshot.effective
    assert repository.store.provenance_path.read_text(encoding="utf-8") == provenance_before
    assert warnings == [
        "Skill evolution failed during Maintenance; this maintenance item was skipped."
    ]


@pytest.mark.asyncio
async def test_automatic_target_selection_may_also_discard_without_modifying_skill(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    now = datetime.now(UTC).replace(microsecond=0)
    original = SkillDefinition(
        "review",
        "Reusable release checklist",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review carefully.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
        when_to_use="before a release",
    )
    await repository.commit(PreparedSkillWrite("add", original, None, "seed", "seed-review"))
    warnings: list[str] = []
    client = _Client([_candidate("review"), _maintenance_discard("No reliable improvement")])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "consider changing the review skill", None)
    await service.wait_idle()

    unchanged = (await repository.snapshot_for_turn()).by_identity[(SkillScope.PROJECT, "review")]
    assert unchanged.version == "0.1.0"
    assert '"name": "review"' in client.requests[1].system_prompt
    assert warnings == []


def test_selected_update_target_cannot_merge_another_offered_candidate(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    selected = SkillDefinition(
        "review",
        "Review changes",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Review.",
        SkillScope.PROJECT,
        tmp_path / "review",
    )
    other = SkillDefinition(
        "release-check",
        "Check releases",
        "0.1.0",
        now,
        now,
        SkillSource.MANUAL,
        "Check releases.",
        SkillScope.PROJECT,
        tmp_path / "release-check",
    )
    operation = MaintenanceOperation.model_validate_json(
        '{"action":"merge","target":{"scope":"project","name":"release-check"},'
        '"skill":{"name":"release-check","description":"Check releases",'
        '"when_to_use":null,"user_invocable":true,"model_invocable":true,'
        '"context":"inline","arguments_required":false,"argument_hint":null,'
        '"allowed_tools":null,"body":"Check releases carefully."},'
        '"evolution_note":"Updated checks","reason":"Selected another candidate"}'
    )
    service = SkillEvolutionService(
        cast(ModelClient, _Client([])),
        SkillRepository(tmp_path),
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=lambda _: None,
    )

    with pytest.raises(SkillEvolutionError, match="selected update target"):
        service._validate_operation(operation, (selected, other), selected.ref)


@pytest.mark.asyncio
async def test_evolution_ignores_reasoning_and_joins_text_in_both_stages(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    warnings: list[str] = []
    client = _Client([_with_reasoning(_candidate()), _with_reasoning(_add())])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "please remember the release steps", None)
    await service.wait_idle()

    definition = (await repository.snapshot_for_turn()).effective["release-check"]
    assert definition.source is SkillSource.GENERATED
    assert len(client.requests) == 2
    assert warnings == []


@pytest.mark.asyncio
async def test_evolution_accepts_other_as_a_complete_text_stop_reason(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    warnings: list[str] = []
    client = _Client([_discard()], [StopReason.OTHER])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "temporary detail", None)
    await service.wait_idle()

    assert warnings == []
    assert repository.store.provenance_path.is_file()


@pytest.mark.parametrize(
    "unsupported",
    [
        ProviderStateBlock("test", "opaque", {"secret": "sensitive-provider-state"}),
        RefusalBlock("sensitive refusal"),
        ToolCallBlock("call-1", "read_file", {"path": "secret"}, '{"path":"secret"}'),
        ToolResultBlock("call-1", (TextBlock("sensitive tool result"),), False),
    ],
    ids=("provider-state", "refusal", "tool-call", "tool-result"),
)
@pytest.mark.asyncio
async def test_evolution_rejects_unsupported_blocks_with_a_sanitized_stage_warning(
    tmp_path: Path, unsupported: ContentBlock
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    warnings: list[str] = []
    client = _Client([(TextBlock(_discard()), unsupported)])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "temporary detail", None)
    await service.wait_idle()

    assert warnings == [
        "Skill evolution failed during Extractor; this maintenance item was skipped."
    ]
    assert "sensitive" not in warnings[0]
    assert not repository.store.provenance_path.exists()


@pytest.mark.asyncio
async def test_evolution_reports_maintenance_failure_without_leaking_response(
    tmp_path: Path,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    warnings: list[str] = []
    client = _Client(
        [
            _candidate(),
            (TextBlock(_add()), RefusalBlock("sensitive maintenance refusal")),
        ]
    )
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "please remember the release steps", None)
    await service.wait_idle()

    assert warnings == [
        "Skill evolution failed during Maintenance; this maintenance item was skipped."
    ]
    assert "sensitive" not in warnings[0]
    assert not (await repository.snapshot_for_turn()).effective


@pytest.mark.asyncio
async def test_evolution_continues_with_the_next_item_after_a_failure(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    warnings: list[str] = []
    client = _Client(
        [
            RuntimeError("sensitive transport failure"),
            _discard(),
        ]
    )
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
        warning=warnings.append,
    )

    service.schedule((), "first item", None)
    service.schedule((), "second item", None)
    await service.wait_idle()

    assert len(client.requests) == 2
    assert warnings == [
        "Skill evolution failed during Extractor; this maintenance item was skipped."
    ]
    assert "sensitive" not in warnings[0]
    assert repository.store.provenance_path.is_file()

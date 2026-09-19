from __future__ import annotations

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
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.skill.evolution import SkillEvolutionService
from hammer_code.skill.models import SkillDefinition, SkillScope, SkillSource
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite


class _Client:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque(responses)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, (TextBlock(self.responses.popleft()),)),
                StopReason.END_TURN,
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
    )

    service.schedule((), "temporary detail", None)
    await service.wait_idle()

    assert not (await repository.snapshot_for_turn()).effective
    assert len(client.requests) == 2
    assert "invalid" in client.requests[1].system_prompt


@pytest.mark.asyncio
async def test_explicit_feedback_can_merge_only_the_exact_forced_target(tmp_path: Path) -> None:
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
        "Review.",
        SkillScope.PROJECT,
        tmp_path / ".hammer-code" / "skill" / "project" / "review",
    )
    await repository.commit(PreparedSkillWrite("add", original, None, "seed", "seed-review"))
    merge = (
        '{"action":"merge","target":{"scope":"project","name":"review"},"skill":'
        '{"name":"review","description":"Review changes","when_to_use":null,'
        '"user_invocable":true,"model_invocable":true,"context":"inline",'
        '"arguments_required":false,"argument_hint":null,"allowed_tools":null,'
        '"body":"Review {{arguments}} carefully."},'
        '"evolution_note":"Added feedback checks","reason":"Explicit feedback"}'
    )
    client = _Client([_candidate("review"), merge])
    service = SkillEvolutionService(
        cast(ModelClient, client),
        repository,
        cast(PermissionService, SimpleNamespace(mode=PermissionMode.UNATTENDED)),
        cast(ConversationManager, SimpleNamespace()),
        _config(),
        100,
    )

    await service.feedback("project:review", "add a final safety check")
    await service.wait_idle()

    updated = (await repository.snapshot_for_turn()).by_identity[(SkillScope.PROJECT, "review")]
    assert updated.version == "0.1.1"
    assert updated.source is SkillSource.MANUAL
    assert "carefully" in updated.prompt_template

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    StopReason,
    ToolDefinition,
)
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.llm.client import ClientCapabilities, ModelClient
from hammer_code.memory.service import MemoryService
from hammer_code.memory.store import MemoryStore
from hammer_code.session.manager import SessionManager
from hammer_code.session.session import SessionCoordinator


class _MemoryClient(ModelClient):
    def __init__(
        self, *, extraction: str | None = None, merge: str | tuple[str, ...] | None = None
    ) -> None:
        self.extraction = extraction or (
            '{"user-preferences":[],"user-experience":[],"project-knowledge":[],"references":[]}'
        )
        self.merge = (merge,) if isinstance(merge, str) else merge or ('{"operations":[]}',)
        self.merge_responses = 0
        self.requests: list[ModelRequest] = []

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    async def aclose(self) -> None:
        pass

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if "candidate extraction" in request.system_prompt:
            text = self.extraction
        else:
            text = self.merge[min(self.merge_responses, len(self.merge) - 1)]
            self.merge_responses += 1
        yield ResponseCompleted(
            request.request_id,
            ModelResponse(
                "provider",
                "model",
                Message(Role.ASSISTANT, (TextBlock(text),)),
                StopReason.END_TURN,
                TokenUsage(1, 1, status=UsageStatus.FINAL),
            ),
        )


class _UI:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def memory_warning(self, message: str) -> None:
        self.warnings.append(message)


class _Registry:
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (ToolDefinition("read_file", "read", {"type": "object"}),)


def _manager() -> ConversationManager:
    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_chat_completions",
                    "model": "model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "KEY",
                    "max_output_tokens": 100,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    manager = ConversationManager()
    manager.create(resolve_profile(config, None, {"KEY": "secret"}))
    return manager


async def _append_turn(
    manager: ConversationManager, coordinator: SessionCoordinator, index: int
) -> None:
    turn = manager.begin_turn(f"user turn {index}")
    committed = manager.commit(
        turn, Message(Role.ASSISTANT, (TextBlock(f"assistant turn {index}"),))
    )
    await coordinator.append_turn(committed, completed=True)


def _service(
    tmp_path: Path, client: _MemoryClient
) -> tuple[MemoryService, SessionCoordinator, _UI]:
    manager = _manager()
    session = SessionManager(tmp_path).create("openai_chat_completions")
    coordinator = SessionCoordinator(session, manager)
    ui = _UI()
    executor = SimpleNamespace(registry=_Registry())
    return (
        MemoryService(
            client,
            MemoryStore(tmp_path),
            coordinator,
            ui,  # type: ignore[arg-type]
            100,
            executor,  # type: ignore[arg-type]
        ),
        coordinator,
        ui,
    )


@pytest.mark.asyncio
async def test_memory_service_calls_model_only_after_five_completed_turns(tmp_path: Path) -> None:
    client = _MemoryClient()
    service, coordinator, ui = _service(tmp_path, client)

    for index in range(1, 5):
        await _append_turn(coordinator.manager, coordinator, index)
        service.maybe_schedule()
        assert service._task is not None
        await service._task
        assert client.requests == []

    await _append_turn(coordinator.manager, coordinator, 5)
    service.maybe_schedule()
    assert service._task is not None
    await service._task

    assert len(client.requests) == 2
    assert "0 < turn_index <= 5" in client.requests[0].system_prompt
    source = "\n".join(
        block.text
        for message in client.requests[0].messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert "turn_index=1" in source
    assert "turn_index=5" in source
    assert coordinator.session.meta.memory_cursor == 5
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_memory_service_reports_sanitized_failure_stage(tmp_path: Path) -> None:
    client = _MemoryClient(extraction="not-json")
    service, coordinator, ui = _service(tmp_path, client)
    for index in range(1, 6):
        await _append_turn(coordinator.manager, coordinator, index)

    service.maybe_schedule()
    assert service._task is not None
    assert not await service._task

    assert coordinator.session.meta.memory_cursor == 0
    assert ui.warnings == [
        "Memory maintenance failed during candidate extraction; it will retry later."
    ]

    for index in range(6, 10):
        await _append_turn(coordinator.manager, coordinator, index)
        service.maybe_schedule()
        assert service._task is not None
        await service._task
    assert len(client.requests) == 1

    await _append_turn(coordinator.manager, coordinator, 10)
    service.maybe_schedule()
    assert service._task is not None
    await service._task
    assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_memory_service_commits_valid_add_operation(tmp_path: Path) -> None:
    client = _MemoryClient(
        extraction=(
            '{"user-preferences":[{"content":"Prefer direct answers",'
            '"evidence_turns":[2,5]}],"user-experience":[],"project-knowledge":[],'
            '"references":[]}'
        ),
        merge=(
            '{"operations":[{"action":"add","category":"user-preferences",'
            '"memory_id":"direct-answers","path":"response-style.md",'
            '"content":"Prefer direct answers.",'
            '"index_description":"Preferred response style"}]}'
        ),
    )
    service, coordinator, ui = _service(tmp_path, client)
    for index in range(1, 6):
        await _append_turn(coordinator.manager, coordinator, index)

    service.maybe_schedule()
    assert service._task is not None
    assert await service._task

    category = tmp_path / ".hammer-code" / "memory" / "user-preferences"
    assert (category / "response-style.md").read_text(
        encoding="utf-8"
    ) == "Prefer direct answers.\n"
    assert (category / "index.md").read_text(encoding="utf-8") == (
        "- [direct-answers](response-style.md): Preferred response style\n"
    )
    assert coordinator.session.meta.memory_cursor == 5
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_memory_service_corrects_invalid_incremental_merge(tmp_path: Path) -> None:
    invalid_full_document = (
        '{"operations":[{"action":"merge","category":"project-knowledge",'
        '"memory_id":"synthetic-widget","path":"synthetic-widget.md",'
        '"content":["# Synthetic widget","",'
        '"- The synthetic project uses red widgets.",'
        '"- The synthetic project also uses blue widgets."],'
        '"index_description":null}]}'
    )
    corrected_increment = (
        '{"operations":[{"action":"merge","category":"project-knowledge",'
        '"memory_id":"synthetic-widget","path":"synthetic-widget.md",'
        '"content":["- The synthetic project also uses blue widgets."],'
        '"index_description":null}]}'
    )
    client = _MemoryClient(
        extraction=(
            '{"user-preferences":[],"user-experience":[],"project-knowledge":['
            '{"content":"The synthetic project also uses blue widgets.",'
            '"evidence_turns":[5]}],"references":[]}'
        ),
        merge=(invalid_full_document, corrected_increment),
    )
    service, coordinator, ui = _service(tmp_path, client)
    category = tmp_path / ".hammer-code" / "memory" / "project-knowledge"
    category.mkdir(parents=True)
    (category / "index.md").write_text(
        "- [synthetic-widget](synthetic-widget.md): Synthetic widget facts\n",
        encoding="utf-8",
    )
    (category / "synthetic-widget.md").write_text(
        "# Synthetic widget\n\n- The synthetic project uses red widgets.\n",
        encoding="utf-8",
    )
    for index in range(1, 6):
        await _append_turn(coordinator.manager, coordinator, index)

    service.maybe_schedule()
    assert service._task is not None
    assert await service._task

    assert len(client.requests) == 3
    correction = client.requests[2].messages[-1]
    assert isinstance(correction.content[0], TextBlock)
    assert "merge.content must" in correction.content[0].text
    assert (category / "synthetic-widget.md").read_text(encoding="utf-8") == (
        "# Synthetic widget\n\n"
        "- The synthetic project uses red widgets.\n"
        "- The synthetic project also uses blue widgets.\n"
    )
    assert coordinator.session.meta.memory_cursor == 5
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_memory_service_stops_after_bounded_merge_corrections(tmp_path: Path) -> None:
    client = _MemoryClient(merge=("not-json", "still-not-json", "invalid-again"))
    service, coordinator, ui = _service(tmp_path, client)
    for index in range(1, 6):
        await _append_turn(coordinator.manager, coordinator, index)

    service.maybe_schedule()
    assert service._task is not None
    assert not await service._task

    assert len(client.requests) == 4
    assert coordinator.session.meta.memory_cursor == 0
    assert ui.warnings == ["Memory maintenance failed during candidate merge; it will retry later."]


def test_merge_catalog_exposes_only_explicit_memory_read_paths(tmp_path: Path) -> None:
    category = tmp_path / ".hammer-code" / "memory" / "user-preferences"
    category.mkdir(parents=True)
    (category / "index.md").write_text(
        "- [response-style](style.md): Preferred response style\n", encoding="utf-8"
    )
    (category / "style.md").write_text("Prefer direct answers.\n", encoding="utf-8")
    catalog = MemoryStore(tmp_path).load_catalog()

    rendered = MemoryService._merge_catalog(catalog)

    assert "path=style.md" in rendered
    assert "read_file_path=.hammer-code/memory/user-preferences/style.md" in rendered

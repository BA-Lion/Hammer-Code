"""Single-task, two-stage Memory maintenance service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hammer_code.domain.events import (
    ModelRequest,
    ResponseCompleted,
    StopReason,
    ToolDefinition,
    UsageUpdated,
)
from hammer_code.domain.messages import (
    Message,
    RefusalBlock,
    Role,
    TextBlock,
    ToolCallBlock,
)
from hammer_code.llm.client import ModelClient
from hammer_code.memory.models import MemoryBatch
from hammer_code.memory.prompts import EXTRACTION_PROMPT, MERGE_PROMPT
from hammer_code.memory.store import MemoryStore, MemoryStoreError
from hammer_code.permissions.paths import PathPolicy
from hammer_code.prompts import build_memory_index_prompt
from hammer_code.session.session import SessionCoordinator
from hammer_code.tools.executor import ToolExecutor
from hammer_code.ui.console import ConsolePort

_CATEGORY_KEYS = (
    "user-preferences",
    "user-experience",
    "project-knowledge",
    "references",
)


class MemoryResponseError(ValueError):
    pass


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content: str = Field(min_length=1)
    evidence_turns: list[int] = Field(min_length=1)

    @field_validator("evidence_turns")
    @classmethod
    def _sorted_unique_positive(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)) or any(item < 1 for item in value):
            raise ValueError("Candidate evidence turns must be sorted unique positive integers")
        return value


class CandidateSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_preferences: list[Candidate] = Field(alias="user-preferences")
    user_experience: list[Candidate] = Field(alias="user-experience")
    project_knowledge: list[Candidate] = Field(alias="project-knowledge")
    references: list[Candidate]


@dataclass
class MemoryService:
    client: ModelClient
    store: MemoryStore
    coordinator: SessionCoordinator
    ui: ConsolePort
    max_output_tokens: int
    executor: ToolExecutor | None = None

    def __post_init__(self) -> None:
        self._task: asyncio.Task[bool] | None = None
        self._pending = False

    async def read_index_prompt(self) -> str:
        try:
            catalog = await asyncio.to_thread(self.store.load_catalog)
            return build_memory_index_prompt(
                {
                    category.value: tuple(
                        f"- [{entry.memory_id}]({entry.path}): {entry.description}"
                        for entry in value.index.entries
                    )
                    for category, value in catalog.categories.items()
                }
            )
        except MemoryStoreError:
            self.ui.memory_warning("Memory Index is unavailable; it was omitted from this request.")
            return ""

    def maybe_schedule(self) -> None:
        if self._task is not None and not self._task.done():
            self._pending = True
            return
        self._task = asyncio.create_task(self._run(force=False))

    async def flush_before_compaction(self) -> bool:
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                return False
        return await self._run(force=True)

    async def cancel(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self, *, force: bool) -> bool:
        try:
            restore = await asyncio.to_thread(self.coordinator.session.load)
            cursor = min(self.coordinator.session.meta.memory_cursor, restore.latest_durable_turn)
            completed_after = sum(turn > cursor for turn in restore.completed_turns)
            if not force and completed_after < 5:
                return True
            target = restore.latest_durable_turn
            if target <= cursor:
                return True
            messages = self._maintenance_messages(restore.records, cursor, target)
            candidates = await self._extract_candidates(messages, cursor, target)
            batch = await self._merge_candidates(candidates)
            prepared = await asyncio.to_thread(self.store.prepare_batch, batch)
            await asyncio.to_thread(self.store.commit_batch, prepared)
            latest = await asyncio.to_thread(self.coordinator.session.load)
            if latest.latest_durable_turn < target:
                raise MemoryResponseError("Session is no longer durable through the Memory target")
            if not self.coordinator.session.compare_replace_meta(cursor, target):
                raise MemoryResponseError("Memory cursor changed before commit")
            return True
        except asyncio.CancelledError:
            raise
        except (MemoryResponseError, MemoryStoreError, OSError, ValueError):
            self.ui.memory_warning("Memory maintenance did not complete; it will retry later.")
            return False
        finally:
            task = asyncio.current_task()
            if self._pending and task is not None and not task.cancelled():
                self._pending = False
                self._task = asyncio.create_task(self._run(force=False))

    async def _extract_candidates(
        self, messages: tuple[Message, ...], cursor: int, target: int
    ) -> CandidateSet:
        text = await self._request_text(EXTRACTION_PROMPT, messages, tools=())
        try:
            result = CandidateSet.model_validate_json(text)
        except ValidationError as exc:
            raise MemoryResponseError("Memory candidate response is not valid JSON") from exc
        for category in _CATEGORY_KEYS:
            for candidate in getattr(result, category.replace("-", "_")):
                if any(turn <= cursor or turn > target for turn in candidate.evidence_turns):
                    raise MemoryResponseError(
                        "Memory candidate cites turns outside its durable range"
                    )
        return result

    async def _merge_candidates(self, candidates: CandidateSet) -> MemoryBatch:
        catalog = await asyncio.to_thread(self.store.load_catalog)
        context = Message(
            Role.USER,
            (
                TextBlock(
                    "Candidates:\n"
                    + candidates.model_dump_json(by_alias=True)
                    + "\n\nIndexes:\n"
                    + build_memory_index_prompt(
                        {
                            category.value: tuple(
                                f"- [{entry.memory_id}]({entry.path}): {entry.description}"
                                for entry in value.index.entries
                            )
                            for category, value in catalog.categories.items()
                        }
                    )
                ),
            ),
        )
        definition = self._read_file_definition()
        messages: list[Message] = [context]
        reads = 0
        for _ in range(20):
            completed = await self._request_response(MERGE_PROMPT, tuple(messages), (definition,))
            response = completed.response
            calls = tuple(
                block for block in response.message.content if isinstance(block, ToolCallBlock)
            )
            if response.stop_reason is StopReason.TOOL_CALL or calls:
                if not calls or response.stop_reason is not StopReason.TOOL_CALL:
                    raise MemoryResponseError("Memory merge tool-call response is invalid")
                reads += len(calls)
                if reads > 50 or any(not self._is_memory_read(call) for call in calls):
                    raise MemoryResponseError("Memory merge attempted an unauthorized tool call")
                assert self.executor is not None
                results = await self.executor.execute_batch(calls)
                messages.extend((response.message, Message(Role.USER, tuple(results))))
                continue
            if response.stop_reason is StopReason.MAX_TOKENS or any(
                isinstance(block, RefusalBlock) for block in response.message.content
            ):
                raise MemoryResponseError("Memory merge did not return a complete JSON response")
            text = "".join(
                block.text for block in response.message.content if isinstance(block, TextBlock)
            )
            try:
                return MemoryBatch.model_validate_json(text)
            except ValidationError as exc:
                raise MemoryResponseError("Memory merge response is not valid JSON") from exc
        raise MemoryResponseError("Memory merge exceeded its request limit")

    def _read_file_definition(self) -> ToolDefinition:
        if self.executor is None:
            raise MemoryResponseError(
                "Memory merge cannot read indexed topics without a tool executor"
            )
        definition = next(
            (item for item in self.executor.registry.definitions() if item.name == "read_file"),
            None,
        )
        if definition is None:
            raise MemoryResponseError("Memory merge read_file tool is unavailable")
        return definition

    def _is_memory_read(self, call: ToolCallBlock) -> bool:
        if call.name != "read_file" or not isinstance(call.arguments.get("path"), str):
            return False
        try:
            project_root = self.store.root.parent.parent
            candidate = PathPolicy(project_root).canonicalize(str(call.arguments["path"]))
            return candidate.is_relative_to(self.store.root.resolve())
        except (OSError, ValueError):
            return False

    async def _request_text(
        self, prompt: str, messages: tuple[Message, ...], *, tools: tuple
    ) -> str:
        completed = await self._request_response(prompt, messages, tools)
        if completed.response.stop_reason in {StopReason.MAX_TOKENS, StopReason.TOOL_CALL}:
            raise MemoryResponseError("Memory model did not return a complete text response")
        if any(
            isinstance(block, (ToolCallBlock, RefusalBlock))
            for block in completed.response.message.content
        ):
            raise MemoryResponseError("Memory response contained a tool call or refusal")
        return "".join(
            block.text
            for block in completed.response.message.content
            if isinstance(block, TextBlock)
        )

    async def _request_response(
        self, prompt: str, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> ResponseCompleted:
        operation_id = f"memory-{uuid4()}"
        request_id = str(uuid4())
        completed: ResponseCompleted | None = None
        try:
            async for event in self.client.stream(
                ModelRequest(
                    request_id,
                    operation_id,
                    prompt,
                    messages,
                    tools,
                    self.max_output_tokens,
                )
            ):
                if event.request_id != request_id:
                    raise MemoryResponseError("Memory stream returned an unexpected request id")
                if isinstance(event, UsageUpdated):
                    self.coordinator.manager.record_maintenance_usage(
                        operation_id, request_id, event.usage
                    )
                elif isinstance(event, ResponseCompleted):
                    if completed is not None:
                        raise MemoryResponseError("Memory stream returned duplicate completion")
                    completed = event
                    self.coordinator.manager.record_maintenance_usage(
                        operation_id, request_id, event.response.usage
                    )
        except MemoryResponseError:
            raise
        except Exception as exc:
            raise MemoryResponseError("Memory model request failed") from exc
        if completed is None:
            raise MemoryResponseError("Memory stream ended without a completion")
        return completed

    @staticmethod
    def _maintenance_messages(records: tuple, cursor: int, target: int) -> tuple[Message, ...]:
        messages: list[Message] = []
        for record in records:
            if not cursor < record.turn_index <= target:
                continue
            blocks: list = []
            for raw in record.content:
                # Persisted records are validated before becoming a maintenance input; JSON reparse
                # avoids giving maintenance requests opaque provider state or hidden reasoning.
                block_type = str(raw.get("type", ""))
                if block_type == "text":
                    blocks.append(TextBlock(str(raw["text"])))
                elif block_type == "refusal":
                    blocks.append(RefusalBlock(str(raw["reason"])))
                elif block_type == "tool_call":
                    blocks.append(TextBlock(f"Tool called: {raw.get('name', '')}"))
                elif block_type == "tool_result":
                    content = raw.get("content", [])
                    blocks.extend(
                        TextBlock(str(item["text"]))
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
            if blocks:
                messages.append(
                    Message(
                        Role.ASSISTANT if record.type.value == "assistant" else Role.USER,
                        tuple(blocks),
                    )
                )
        return tuple(messages)

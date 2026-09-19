"""Per-session, permission-gated two-stage Skill maintenance worker."""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import SkillConfig
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import ModelRequest, ResponseCompleted, UsageUpdated
from hammer_code.domain.messages import Message, RefusalBlock, TextBlock
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.skill.catalog import SkillCatalog
from hammer_code.skill.models import (
    CatalogSnapshot,
    EvolutionQueueItem,
    ExtractedCandidate,
    ExtractorResponse,
    MaintenanceOperation,
    MaintenanceSkill,
    SkillDefinition,
    SkillObservation,
    SkillRef,
    SkillScope,
    SkillSource,
)
from hammer_code.skill.prompts import (
    build_correction_prompt,
    build_extractor_prompt,
    build_maintenance_prompt,
)
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite


class SkillEvolutionError(ValueError):
    pass


def _can_write(config: SkillConfig, permissions: PermissionService) -> bool:
    return (
        config.enabled
        and config.evolution.enabled
        and permissions.mode
        in {
            PermissionMode.ACCEPT_EDITS,
            PermissionMode.UNATTENDED,
        }
    )


class SkillEvolutionService:
    """A single Session worker; each user input is retained as its own queue item."""

    def __init__(
        self,
        client: ModelClient,
        repository: SkillRepository,
        permissions: PermissionService,
        manager: ConversationManager,
        config: SkillConfig,
        max_output_tokens: int,
        available_tool_names: tuple[str, ...] = (),
    ) -> None:
        self.client = client
        self.repository = repository
        self.permissions = permissions
        self.manager = manager
        self.config = config
        self.max_output_tokens = max_output_tokens
        self.available_tool_names = frozenset(available_tool_names)
        self._queue: deque[EvolutionQueueItem] = deque()
        self._task: asyncio.Task[None] | None = None
        self._accepting = True
        self._estimator = TokenEstimator()

    def schedule(
        self, messages: tuple[Message, ...], current_input: str, previous: SkillObservation | None
    ) -> None:
        if not self._accepting or not _can_write(self.config, self.permissions):
            return
        self._queue.append(EvolutionQueueItem(tuple(messages), current_input, previous))
        self._start_worker()

    async def feedback(self, name: str, feedback: str) -> None:
        if not feedback.strip():
            raise SkillEvolutionError("Skill feedback must not be empty")
        if not _can_write(self.config, self.permissions):
            raise SkillEvolutionError(
                "Skill feedback requires enabled evolution and accept_edits or unattended mode"
            )
        snapshot = await self.repository.snapshot_for_turn()
        definition = SkillCatalog.resolve(snapshot, name, user=False, model=False)
        self._queue.append(EvolutionQueueItem((), "", None, definition.ref, feedback.strip()))
        self._start_worker()

    def _start_worker(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker())

    async def wait_idle(self) -> bool:
        if self._task is not None:
            await asyncio.shield(self._task)
        return True

    async def drain(self) -> bool:
        self._accepting = False
        return await self.wait_idle()

    async def cancel(self) -> None:
        self._accepting = False
        self._queue.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _worker(self) -> None:
        while self._queue:
            item = self._queue.popleft()
            try:
                await self._process(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Main conversation progress never depends on a maintenance result.
                continue

    def _visible(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        visible: list[Message] = []
        for message in reversed(messages):
            blocks = tuple(
                block for block in message.content if isinstance(block, (TextBlock, RefusalBlock))
            )
            if blocks:
                visible.append(Message(message.role, blocks))
            if len(visible) >= self.config.evolution.max_history_messages:
                break
        visible.reverse()
        while (
            visible
            and self._estimator.estimate_messages(tuple(visible))
            > self.config.evolution.max_input_tokens
        ):
            visible.pop(0)
        return tuple(visible)

    def _truncate_current_input(self, value: str) -> str:
        """Keep the independent current-input tag inside its configured budget."""
        if self._estimator.estimate_text(value) <= self.config.evolution.max_input_tokens:
            return value
        encoded = value.encode("utf-8")
        low, high = 0, len(encoded)
        while low < high:
            midpoint = (low + high + 1) // 2
            trial = encoded[:midpoint].decode("utf-8", errors="ignore")
            if (
                self._estimator.estimate_text(trial + "\n[truncated]")
                <= self.config.evolution.max_input_tokens
            ):
                low = midpoint
            else:
                high = midpoint - 1
        return encoded[:low].decode("utf-8", errors="ignore") + "\n[truncated]"

    async def _request_text(self, prompt: str, messages: tuple[Message, ...]) -> str:
        request_id = str(uuid4())
        request = ModelRequest(
            request_id,
            f"skill-evolution-{request_id}",
            prompt,
            messages,
            (),
            self.max_output_tokens,
        )
        completed: ResponseCompleted | None = None
        async for event in self.client.stream(request):
            if event.request_id != request_id:
                raise SkillEvolutionError("maintenance client emitted a mismatched request")
            if isinstance(event, UsageUpdated):
                self.manager.record_maintenance_usage(
                    f"skill-evolution-{request_id}", request_id, event.usage
                )
            elif isinstance(event, ResponseCompleted):
                if completed is not None:
                    raise SkillEvolutionError("maintenance client emitted duplicate completion")
                completed = event
        if completed is None or completed.response.stop_reason.value != "end_turn":
            raise SkillEvolutionError(
                "maintenance request did not produce one completed text response"
            )
        if any(not isinstance(block, TextBlock) for block in completed.response.message.content):
            raise SkillEvolutionError("maintenance response included non-text content")
        text_blocks = tuple(
            block for block in completed.response.message.content if isinstance(block, TextBlock)
        )
        text = "".join(block.text for block in text_blocks)
        if not text.strip():
            raise SkillEvolutionError("maintenance response was empty")
        return text

    async def _validated(
        self,
        stage: str,
        model: type[ExtractorResponse] | type[MaintenanceOperation],
        prompt: str,
        messages: tuple[Message, ...],
    ) -> ExtractorResponse | MaintenanceOperation:
        response = await self._request_text(prompt, messages)
        for attempt in range(self.config.evolution.max_corrections + 1):
            try:
                return model.model_validate_json(response)
            except ValidationError as exc:
                if attempt >= self.config.evolution.max_corrections:
                    raise SkillEvolutionError(f"{stage} returned invalid JSON") from exc
                errors = "; ".join(item["msg"] for item in exc.errors()[:5])
                response = await self._request_text(
                    build_correction_prompt(stage, response, errors), ()
                )
        raise AssertionError("unreachable")

    @staticmethod
    def _expected_evaluations(previous: SkillObservation | None) -> set[SkillRef]:
        return (
            set()
            if previous is None or not previous.completed
            else {item.ref for item in previous.retrieved}
        )

    def _validate_extractor(
        self, response: ExtractorResponse, previous: SkillObservation | None
    ) -> tuple[tuple[SkillRef, str], ...]:
        expected = self._expected_evaluations(previous)
        received = [SkillRef(item.scope, item.name, item.version) for item in response.evaluations]
        if len(received) != len(set(received)) or set(received) != expected:
            raise SkillEvolutionError(
                "Extractor evaluations do not exactly match previous retrieval"
            )
        if response.candidate and response.candidate.allowed_tools is not None:
            if set(response.candidate.allowed_tools).difference(self.available_tool_names):
                raise SkillEvolutionError("Extractor candidate contains unknown allowed tools")
        if (
            response.candidate is not None
            and len(response.candidate.body) > self.config.evolution.max_body_chars
        ):
            raise SkillEvolutionError("Extractor candidate body exceeds configured maximum")
        return tuple(
            (SkillRef(item.scope, item.name, item.version), item.reason)
            for item in response.evaluations
            if item.relevant
        )

    def _candidate_definitions(
        self, snapshot: CatalogSnapshot, candidate: ExtractedCandidate, forced_ref: SkillRef | None
    ) -> tuple[SkillDefinition, ...]:
        query = self._candidate_query(candidate)
        matches = (
            snapshot.index.normalized_merge_candidates(
                query,
                self._proposed_definition(candidate),
                snapshot.by_identity.values(),
                self.config.evolution.merge_candidate_top_k,
            )
            if snapshot.index
            else ()
        )
        refs = [item.ref for item in matches]
        if forced_ref is not None and forced_ref not in refs:
            refs.insert(0, forced_ref)
        selected: list[SkillDefinition] = []
        forced = (
            snapshot.by_identity.get((forced_ref.scope, forced_ref.name)) if forced_ref else None
        )
        for ref in refs:
            definition = snapshot.by_identity.get((ref.scope, ref.name))
            if definition is None or definition.version != ref.version:
                continue
            prompt = build_maintenance_prompt(candidate, tuple(selected + [definition]), forced)
            if (
                self._estimator.estimate_text(prompt)
                > self.config.evolution.max_maintenance_input_tokens
            ):
                if ref == forced_ref:
                    raise SkillEvolutionError("forced target exceeds maintenance input budget")
                continue
            selected.append(definition)
        if forced_ref is not None and not any(item.ref == forced_ref for item in selected):
            raise SkillEvolutionError("forced target is unavailable")
        return tuple(selected)

    @staticmethod
    def _candidate_query(candidate: ExtractedCandidate) -> str:
        return candidate.name + " " + candidate.description + " " + (candidate.when_to_use or "")

    def _proposed_definition(self, candidate: ExtractedCandidate) -> SkillDefinition:
        now = datetime.now(UTC)
        return SkillDefinition(
            candidate.name,
            candidate.description,
            "0.1.0",
            now,
            now,
            SkillSource.GENERATED,
            candidate.body,
            SkillScope.PROJECT,
            self.repository.catalog.root / "project" / candidate.name,
            candidate.user_invocable,
            candidate.model_invocable,
            candidate.context,
            candidate.when_to_use,
            candidate.arguments_required,
            candidate.argument_hint,
            candidate.allowed_tools,
        )

    def _automatic_forced_ref(
        self, snapshot: CatalogSnapshot, candidate: ExtractedCandidate
    ) -> SkillRef | None:
        if snapshot.index is None:
            return None
        matches = snapshot.index.normalized_merge_candidates(
            self._candidate_query(candidate),
            self._proposed_definition(candidate),
            snapshot.by_identity.values(),
            2,
        )
        first = next(iter(matches), None)
        if first is None or first.relative_score < self.config.evolution.forced_merge_score:
            return None
        if (
            len(matches) > 1
            and first.relative_score - matches[1].relative_score
            < self.config.evolution.forced_merge_margin
        ):
            return None
        return first.ref

    def _validate_operation(
        self,
        operation: MaintenanceOperation,
        candidates: tuple[SkillDefinition, ...],
        forced_ref: SkillRef | None,
    ) -> None:
        if operation.action == "discard":
            if forced_ref is not None:
                raise SkillEvolutionError("explicit feedback must merge its forced target")
            return
        assert operation.skill is not None
        if len(operation.skill.body) > self.config.evolution.max_body_chars:
            raise SkillEvolutionError("Maintenance Skill body exceeds configured maximum")
        if operation.skill.allowed_tools is not None and set(
            operation.skill.allowed_tools
        ).difference(self.available_tool_names):
            raise SkillEvolutionError("Maintenance output contains unknown allowed tools")
        if operation.action == "add":
            if forced_ref is not None:
                raise SkillEvolutionError("explicit feedback cannot add a Skill")
            return
        assert operation.target is not None
        target = next(
            (
                item
                for item in candidates
                if item.scope == operation.target.scope and item.name == operation.target.name
            ),
            None,
        )
        if target is None or operation.skill.name != target.name:
            raise SkillEvolutionError("Maintenance merge target is not an offered candidate")
        if forced_ref is not None and target.ref != forced_ref:
            raise SkillEvolutionError("Maintenance merge does not match forced target")

    async def _process(self, item: EvolutionQueueItem) -> None:
        if not _can_write(self.config, self.permissions):
            return
        snapshot = await self.repository.snapshot_for_turn()
        previous = item.previous_observation
        extractor = await self._validated(
            "Extractor",
            ExtractorResponse,
            build_extractor_prompt(
                self._truncate_current_input(item.current_input),
                previous.retrieved if previous and previous.completed else (),
                item.feedback,
            ),
            self._visible(item.messages),
        )
        assert isinstance(extractor, ExtractorResponse)
        evaluations = self._validate_extractor(extractor, previous)
        if evaluations:
            await self.repository.record_evaluations(evaluations)
        if extractor.candidate is None:
            await self.repository.record_discard(extractor.discard_reason or "Extractor discarded")
            return
        forced_ref = item.forced_ref or self._automatic_forced_ref(snapshot, extractor.candidate)
        candidates = self._candidate_definitions(snapshot, extractor.candidate, forced_ref)
        forced = (
            snapshot.by_identity.get((forced_ref.scope, forced_ref.name)) if forced_ref else None
        )
        operation = await self._validated(
            "Maintenance",
            MaintenanceOperation,
            build_maintenance_prompt(extractor.candidate, candidates, forced),
            (),
        )
        assert isinstance(operation, MaintenanceOperation)
        self._validate_operation(operation, candidates, forced_ref)
        if operation.action == "discard":
            await self.repository.record_discard(operation.reason)
            return
        await self._commit(snapshot, operation)

    @staticmethod
    def _definition(
        skill: MaintenanceSkill,
        *,
        scope: SkillScope,
        source: SkillSource,
        created_at: datetime,
        updated_at: datetime,
        skill_dir: Path,
        version: str,
    ) -> SkillDefinition:
        return SkillDefinition(
            skill.name,
            skill.description,
            version,
            created_at,
            updated_at,
            source,
            skill.body,
            scope,
            skill_dir,
            skill.user_invocable,
            skill.model_invocable,
            skill.context,
            skill.when_to_use,
            skill.arguments_required,
            skill.argument_hint,
            skill.allowed_tools,
        )

    async def _commit(self, snapshot: CatalogSnapshot, operation: MaintenanceOperation) -> None:
        if not _can_write(self.config, self.permissions) or operation.action == "discard":
            return
        assert operation.skill is not None and operation.evolution_note is not None
        now = datetime.now(UTC).replace(microsecond=0)
        if operation.action == "add":
            definition = self._definition(
                operation.skill,
                scope=SkillScope.PROJECT,
                source=SkillSource.GENERATED,
                created_at=now,
                updated_at=now,
                skill_dir=self.repository.catalog.root / "project" / operation.skill.name,
                version="0.1.0",
            )
            prepared = PreparedSkillWrite(
                "add", definition, None, operation.evolution_note, str(uuid4())
            )
        else:
            assert operation.target is not None
            previous = snapshot.by_identity.get((operation.target.scope, operation.target.name))
            if previous is None:
                raise SkillEvolutionError("merge target disappeared")
            major, minor, patch = (int(part) for part in previous.version.split("."))
            definition = self._definition(
                operation.skill,
                scope=previous.scope,
                source=previous.source,
                created_at=previous.created_at,
                updated_at=now,
                skill_dir=previous.skill_dir,
                version=f"{major}.{minor}.{patch + 1}",
            )
            preimage = hashlib.sha256((previous.skill_dir / "SKILL.md").read_bytes()).hexdigest()
            prepared = PreparedSkillWrite(
                "merge", definition, preimage, operation.evolution_note, str(uuid4())
            )
        await self.repository.commit(prepared, expected_snapshot=snapshot)
        # Pruning is evaluated only after a successful, fully published write;
        # its own repository lock rechecks source, version, and preimage.
        await self.repository.prune_eligible(self.config)

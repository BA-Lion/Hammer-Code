"""Per-session, permission-gated two-stage Skill maintenance worker."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import SkillConfig
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import ModelRequest, ResponseCompleted, StopReason, UsageUpdated
from hammer_code.domain.messages import Message, ReasoningBlock, TextBlock
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.service import PermissionService
from hammer_code.skill.catalog import SkillCatalog
from hammer_code.skill.embedding import EmbeddingQuery, SkillMetadata
from hammer_code.skill.models import (
    CatalogSnapshot,
    EvaluatorResponse,
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
    SkillTurnEvidence,
)
from hammer_code.skill.prompts import (
    build_correction_prompt,
    build_evaluator_prompt,
    build_extractor_prompt,
    build_maintenance_prompt,
)
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite


class SkillEvolutionError(ValueError):
    pass


class SkillEvolutionStageError(SkillEvolutionError):
    """Keep an internal cause while exposing only a safe processing stage."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"Skill evolution failed during {stage}")
        self.stage = stage


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
        *,
        warning: Callable[[str], None],
    ) -> None:
        self.client = client
        self.repository = repository
        self.permissions = permissions
        self.manager = manager
        self.config = config
        self.max_output_tokens = max_output_tokens
        self.available_tool_names = frozenset(available_tool_names)
        self._warning = warning
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

    def schedule_evidence(self, evidence: SkillTurnEvidence, current_input: str) -> None:
        """Queue the immutable completed-turn snapshot without blocking the main turn."""
        if not self._accepting or not _can_write(self.config, self.permissions):
            return
        self._queue.append(
            EvolutionQueueItem(evidence.live_messages, current_input, None, evidence=evidence)
        )
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
        self._queue.append(
            EvolutionQueueItem(
                (),
                "",
                None,
                update_target_ref=definition.ref,
                feedback=feedback.strip(),
            )
        )
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
            except SkillEvolutionStageError as exc:
                self._warn_failure(exc.stage)
            except Exception:
                self._warn_failure("processing")

    def _warn_failure(self, stage: str) -> None:
        """Report once per failed item without leaking model or exception content."""
        try:
            self._warning(
                f"Skill evolution failed during {stage}; this maintenance item was skipped."
            )
        except Exception:
            # A presentation failure must not terminate the queue or affect the main conversation.
            pass

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
        if completed is None or completed.response.stop_reason not in {
            StopReason.END_TURN,
            StopReason.OTHER,
        }:
            raise SkillEvolutionError(
                "Skill Evolution request did not produce one completed text response"
            )
        if any(
            not isinstance(block, (TextBlock, ReasoningBlock))
            for block in completed.response.message.content
        ):
            raise SkillEvolutionError("Skill Evolution response included unsupported content")
        text_blocks = tuple(
            block for block in completed.response.message.content if isinstance(block, TextBlock)
        )
        text = "".join(block.text for block in text_blocks)
        if not text.strip():
            raise SkillEvolutionError("Skill Evolution response was empty")
        return text

    async def _validated(
        self,
        stage: str,
        model: type[ExtractorResponse] | type[MaintenanceOperation] | type[EvaluatorResponse],
        prompt: str,
        messages: tuple[Message, ...],
    ) -> ExtractorResponse | MaintenanceOperation | EvaluatorResponse:
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

    def _validate_extractor(self, response: ExtractorResponse) -> None:
        if response.candidate and response.candidate.allowed_tools is not None:
            if set(response.candidate.allowed_tools).difference(self.available_tool_names):
                raise SkillEvolutionError("Extractor candidate contains unknown allowed tools")
        if (
            response.candidate is not None
            and len(response.candidate.body) > self.config.evolution.max_body_chars
        ):
            raise SkillEvolutionError("Extractor candidate body exceeds configured maximum")
        return None

    @staticmethod
    def _validate_evaluator(response: EvaluatorResponse, evidence: SkillTurnEvidence) -> None:
        expected_retrieved = {item.ref for item in evidence.retrieved}
        expected_loaded = {item.ref for item in evidence.loaded}
        relevance = [
            SkillRef(item.scope, item.name, item.version) for item in response.relevance_evaluations
        ]
        application = [
            SkillRef(item.scope, item.name, item.version)
            for item in response.application_evaluations
        ]
        if len(relevance) != len(set(relevance)) or set(relevance) != expected_retrieved:
            raise SkillEvolutionError(
                "Evaluator relevance identities do not exactly match retrieval"
            )
        if len(application) != len(set(application)) or set(application) != expected_loaded:
            raise SkillEvolutionError(
                "Evaluator application identities do not exactly match loaded Skills"
            )

    async def _candidate_definitions(
        self,
        snapshot: CatalogSnapshot,
        candidate: ExtractedCandidate,
        update_target_ref: SkillRef | None,
    ) -> tuple[CatalogSnapshot, tuple[SkillDefinition, ...], SkillRef | None]:
        matches = await self.repository.search.search(
            EmbeddingQuery.from_metadata(
                SkillMetadata(candidate.name, candidate.description, candidate.when_to_use)
            ),
            self.config.evolution.merge_candidate_top_k,
            snapshot=snapshot,
            purpose="maintenance",
        )
        snapshot = await self.repository.snapshot_for_turn()
        if update_target_ref is not None:
            latest = snapshot.by_identity.get((update_target_ref.scope, update_target_ref.name))
            if latest is None:
                raise SkillEvolutionError("selected update target is unavailable")
            update_target_ref = latest.ref
        refs = [item.ref for item in matches]
        if update_target_ref is not None and update_target_ref not in refs:
            refs.insert(0, update_target_ref)
        selected: list[SkillDefinition] = []
        seen: set[tuple[SkillScope, str]] = set()
        update_target = (
            snapshot.by_identity.get((update_target_ref.scope, update_target_ref.name))
            if update_target_ref
            else None
        )
        for ref in refs:
            identity = (ref.scope, ref.name)
            if identity in seen:
                continue
            seen.add(identity)
            definition = snapshot.by_identity.get((ref.scope, ref.name))
            if definition is None:
                continue
            prompt = build_maintenance_prompt(
                candidate, tuple(selected + [definition]), update_target
            )
            if (
                self._estimator.estimate_text(prompt)
                > self.config.evolution.max_maintenance_input_tokens
            ):
                if ref == update_target_ref:
                    raise SkillEvolutionError(
                        "selected update target exceeds maintenance input budget"
                    )
                continue
            selected.append(definition)
        if update_target_ref is not None and not any(
            item.ref == update_target_ref for item in selected
        ):
            raise SkillEvolutionError("selected update target is unavailable")
        return snapshot, tuple(selected), update_target_ref

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

    def _validate_operation(
        self,
        operation: MaintenanceOperation,
        candidates: tuple[SkillDefinition, ...],
        update_target_ref: SkillRef | None,
    ) -> None:
        if operation.action == "discard":
            return
        assert operation.skill is not None
        if len(operation.skill.body) > self.config.evolution.max_body_chars:
            raise SkillEvolutionError("Maintenance Skill body exceeds configured maximum")
        if operation.skill.allowed_tools is not None and set(
            operation.skill.allowed_tools
        ).difference(self.available_tool_names):
            raise SkillEvolutionError("Maintenance output contains unknown allowed tools")
        if operation.action == "add":
            if update_target_ref is not None:
                raise SkillEvolutionError(
                    "Maintenance cannot add when an update target is selected"
                )
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
        if update_target_ref is not None and target.ref != update_target_ref:
            raise SkillEvolutionError("Maintenance merge does not match selected update target")

    async def _process(self, item: EvolutionQueueItem) -> None:
        stage = "permission check"
        try:
            if not _can_write(self.config, self.permissions):
                return
            stage = "catalog snapshot"
            snapshot = await self.repository.snapshot_for_turn()
            previous = item.previous_observation
            evidence = item.evidence
            if evidence is not None:
                stage = "Evaluator"
                try:
                    evaluator = await self._validated(
                        "Evaluator",
                        EvaluatorResponse,
                        build_evaluator_prompt(
                            evidence.retrieved, evidence.loaded, evidence.retrieved_prompt
                        ),
                        evidence.live_messages,
                    )
                    assert isinstance(evaluator, EvaluatorResponse)
                    self._validate_evaluator(evaluator, evidence)
                    await self.repository.record_evaluation(
                        evaluator.relevance_evaluations,
                        evaluator.application_evaluations,
                        retrieve_event_ids=evidence.retrieve_event_ids,
                        loaded_event_ids=tuple(item.event_id for item in evidence.loaded),
                    )
                except Exception:
                    try:
                        await self.repository.record_evaluation((), (), failure_stage="evaluator")
                    except Exception:
                        # An auxiliary write failure must not suppress Extractor.
                        pass
                    self._warn_failure("Evaluator")
                else:
                    try:
                        await self.repository.prune_eligible(
                            self.config, can_write=lambda: _can_write(self.config, self.permissions)
                        )
                    except Exception:
                        self._warn_failure("prune")
            stage = "Extractor"
            extractor = await self._validated(
                "Extractor",
                ExtractorResponse,
                build_extractor_prompt(
                    item.current_input,
                    evidence.retrieved
                    if evidence is not None
                    else previous.retrieved
                    if previous and previous.completed
                    else (),
                    item.feedback,
                    evidence.retrieved_prompt if evidence is not None else "",
                ),
                item.messages,
            )
            assert isinstance(extractor, ExtractorResponse)
            self._validate_extractor(extractor)
            if extractor.candidate is None:
                stage = "commit"
                await self.repository.record_discard(
                    extractor.discard_reason or "Extractor discarded"
                )
                return
            stage = "Maintenance"
            update_target_ref = item.update_target_ref
            snapshot = await self.repository.snapshot_for_turn()
            snapshot, candidates, update_target_ref = await self._candidate_definitions(
                snapshot, extractor.candidate, update_target_ref
            )
            update_target = (
                snapshot.by_identity.get((update_target_ref.scope, update_target_ref.name))
                if update_target_ref
                else None
            )
            operation = await self._validated(
                "Maintenance",
                MaintenanceOperation,
                build_maintenance_prompt(extractor.candidate, candidates, update_target),
                (),
            )
            assert isinstance(operation, MaintenanceOperation)
            self._validate_operation(operation, candidates, update_target_ref)
            stage = "commit"
            if operation.action == "discard":
                await self.repository.record_discard(operation.reason)
                return
            await self._commit(snapshot, operation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise SkillEvolutionStageError(stage) from exc

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
            preimage = snapshot.content_hashes.get((previous.scope, previous.name))
            if preimage is None:
                raise SkillEvolutionError("merge snapshot has no content preimage")
            prepared = PreparedSkillWrite(
                "merge", definition, preimage, operation.evolution_note, str(uuid4())
            )
        await self.repository.commit(prepared, expected_snapshot=snapshot)
        # Pruning is evaluated only after a successful, fully published write;
        # its own repository lock rechecks source, version, and preimage.
        await self.repository.prune_eligible(
            self.config, can_write=lambda: _can_write(self.config, self.permissions)
        )

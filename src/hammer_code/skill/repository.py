"""The single process-scoped owner of Skill catalog and mutable state."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hammer_code.config import SkillConfig
from hammer_code.skill.catalog import SkillCatalog, SkillCatalogError
from hammer_code.skill.models import (
    ApplicationEvaluation,
    CatalogSnapshot,
    RelevanceEvaluation,
    SkillDefinition,
    SkillRef,
    empty_snapshot,
)
from hammer_code.skill.store import (
    CommitResult,
    PreparedSkillWrite,
    RecoveryReport,
    SkillStore,
    SkillStoreError,
)


class SkillRepositoryError(ValueError):
    pass


class SkillRepository:
    def __init__(self, project_root: Path) -> None:
        self.catalog = SkillCatalog(project_root)
        self.store = SkillStore(project_root)
        self._current = empty_snapshot()
        self._lock = asyncio.Lock()
        self._state_degraded = False
        self._catalog_unavailable = False
        self._pending_evaluations: set[SkillRef] = set()

    @property
    def state_degraded(self) -> bool:
        return self._state_degraded

    async def initialize(self) -> tuple[CatalogSnapshot, RecoveryReport]:
        async with self._lock:
            report = await asyncio.to_thread(self.store.recover_pending)
            self._current = await asyncio.to_thread(self.catalog.build_snapshot, None)
            self._catalog_unavailable = False
            try:
                await asyncio.to_thread(
                    self.store.rebuild_usage, self._current.by_identity.values()
                )
                self._state_degraded = False
            except SkillStoreError:
                self._state_degraded = True
            return self._current, report

    async def snapshot_for_turn(self) -> CatalogSnapshot:
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                self._catalog_unavailable = False
                return self._current
            except SkillCatalogError as exc:
                self._catalog_unavailable = True
                raise SkillRepositoryError("Skill catalog is unavailable") from exc

    def _assert_writable_state(self) -> None:
        if self._state_degraded:
            raise SkillRepositoryError("Skill state is unavailable; Skill usage is disabled")

    async def _validate_refs(self, refs: tuple[SkillRef, ...]) -> None:
        try:
            self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
        except SkillCatalogError as exc:
            self._catalog_unavailable = True
            raise SkillRepositoryError("Skill catalog is unavailable") from exc
        for ref in refs:
            definition = self._current.by_identity.get((ref.scope, ref.name))
            if definition is None or definition.version != ref.version:
                raise SkillRepositoryError("Skill identity changed before state update")

    async def resolve_latest(self, name: str, *, user: bool, model: bool) -> SkillDefinition:
        """Resolve at invocation time, never from the turn-start retrieval snapshot."""
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                return self.catalog.resolve(self._current, name, user=user, model=model)
            except SkillCatalogError as exc:
                self._catalog_unavailable = True
                raise SkillRepositoryError("Skill catalog is unavailable") from exc

    async def record_retrieved(
        self,
        refs: tuple[SkillRef, ...],
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        catalog_generation: int = 0,
    ) -> tuple[str, ...]:
        async with self._lock:
            self._assert_writable_state()
            try:
                event_ids = await asyncio.to_thread(
                    self.store.record_retrieved,
                    refs,
                    session_id=session_id,
                    turn_id=turn_id,
                    catalog_generation=catalog_generation,
                )
                await asyncio.to_thread(
                    self.store.rebuild_usage, self._current.by_identity.values()
                )
                return event_ids
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

    async def record_loaded(
        self,
        ref: SkillRef,
        *,
        invocation_kind: str,
        session_id: str | None = None,
        turn_id: str | None = None,
        catalog_generation: int = 0,
    ) -> str:
        async with self._lock:
            self._assert_writable_state()
            try:
                event_id = await asyncio.to_thread(
                    self.store.record_loaded,
                    ref,
                    invocation_kind=invocation_kind,
                    session_id=session_id,
                    turn_id=turn_id,
                    catalog_generation=catalog_generation,
                )
                await asyncio.to_thread(
                    self.store.rebuild_usage, self._current.by_identity.values()
                )
                return event_id
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

    async def record_evaluation(
        self,
        relevance: tuple[RelevanceEvaluation, ...],
        application: tuple[ApplicationEvaluation, ...],
        *,
        retrieve_event_ids: tuple[str, ...] = (),
        loaded_event_ids: tuple[str, ...] = (),
        session_id: str | None = None,
        turn_id: str | None = None,
        failure_stage: str | None = None,
    ) -> str:
        async with self._lock:
            self._assert_writable_state()
            try:
                event_id = await asyncio.to_thread(
                    self.store.record_evaluation,
                    relevance,
                    application,
                    retrieve_event_ids=retrieve_event_ids,
                    loaded_event_ids=loaded_event_ids,
                    session_id=session_id,
                    turn_id=turn_id,
                    failure_stage=failure_stage,
                )
                await asyncio.to_thread(
                    self.store.rebuild_usage, self._current.by_identity.values()
                )
                return event_id
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

    async def record_used(self, ref: SkillRef) -> None:
        """Compatibility shim for callers predating the unambiguous loaded signal."""
        await self.record_loaded(ref, invocation_kind="legacy")

    async def reserve_evaluation(self, refs: tuple[SkillRef, ...]) -> None:
        async with self._lock:
            self._pending_evaluations.update(refs)

    async def release_evaluation(self, refs: tuple[SkillRef, ...]) -> None:
        async with self._lock:
            self._pending_evaluations.difference_update(refs)

    async def record_discard(self, reason: str) -> None:
        async with self._lock:
            self._assert_writable_state()
            try:
                await asyncio.to_thread(self.store.record_discard, reason)
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill provenance could not be saved") from exc

    async def commit(
        self, prepared: PreparedSkillWrite, *, expected_snapshot: CatalogSnapshot | None = None
    ) -> CommitResult:
        async with self._lock:
            self._assert_writable_state()
            try:
                # Re-scan inside the write lock.  A worker may only commit the
                # exact catalog generation it analysed; Store then performs the
                # target-specific preimage CAS as a second guard.
                latest = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                if (
                    expected_snapshot is not None
                    and latest.fingerprint != expected_snapshot.fingerprint
                ):
                    raise SkillRepositoryError("Skill catalog changed before commit")
                self._current = latest
                result = await asyncio.to_thread(self.store.commit, prepared)
                self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                return result
            except (SkillStoreError, SkillCatalogError) as exc:
                raise SkillRepositoryError("Skill update could not be committed") from exc

    async def prune(self, definition: SkillDefinition, expected_preimage: str) -> CommitResult:
        async with self._lock:
            self._assert_writable_state()
            try:
                result = await asyncio.to_thread(
                    self.store.commit_prune, definition, expected_preimage
                )
                self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                return result
            except (SkillStoreError, SkillCatalogError) as exc:
                raise SkillRepositoryError("Skill prune could not be committed") from exc

    async def restore_pruned(self, definition: SkillDefinition, pruned_at: str) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self.store.restore_pruned, definition, pruned_at)
                self._current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
            except (SkillStoreError, SkillCatalogError) as exc:
                raise SkillRepositoryError("Pruned Skill could not be restored") from exc

    async def prune_eligible(self, config: SkillConfig) -> tuple[SkillRef, ...]:
        """Prune only current generated entries satisfying a configured safe threshold."""
        async with self._lock:
            self._assert_writable_state()
            try:
                current = await asyncio.to_thread(self.catalog.build_snapshot, self._current)
                usage = await asyncio.to_thread(self.store.load_usage)
                pruned: list[SkillRef] = []
                for definition in tuple(current.by_identity.values()):
                    if definition.source.value != "generated":
                        continue
                    entry = usage.get(f"{definition.scope.value}:{definition.name}")
                    if entry is None or entry.version != definition.version:
                        continue
                    window = entry.adjudicated_retrievals
                    if len(window) < 5 or definition.ref in self._pending_evaluations:
                        continue
                    baseline = entry.version_last_applied_at or definition.updated_at.astimezone(
                        UTC
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    oldest = datetime.strptime(
                        window[0].retrieved_at, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=UTC)
                    baseline_time = datetime.strptime(baseline, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=UTC
                    )
                    if oldest < baseline_time + timedelta(
                        days=config.evolution.prune_unapplied_days
                    ):
                        continue
                    preimage = await asyncio.to_thread(
                        lambda path=definition.skill_dir / "SKILL.md": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                    )
                    await asyncio.to_thread(self.store.commit_prune, definition, preimage)
                    pruned.append(definition.ref)
                if pruned:
                    self._current = await asyncio.to_thread(self.catalog.build_snapshot, current)
                return tuple(pruned)
            except (OSError, SkillStoreError, SkillCatalogError, ValueError) as exc:
                raise SkillRepositoryError("Eligible Skill prune could not be committed") from exc

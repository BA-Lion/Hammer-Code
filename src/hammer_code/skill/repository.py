"""The single process-scoped owner of Skill catalog and mutable state."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from hammer_code.config import SkillConfig
from hammer_code.skill.catalog import SkillCatalog, SkillCatalogError
from hammer_code.skill.models import CatalogSnapshot, SkillDefinition, SkillRef, empty_snapshot
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

    @property
    def state_degraded(self) -> bool:
        return self._state_degraded

    async def initialize(self) -> tuple[CatalogSnapshot, RecoveryReport]:
        async with self._lock:
            report = await asyncio.to_thread(self.store.recover_pending)
            self._current = await asyncio.to_thread(self.catalog.build_snapshot, None)
            self._catalog_unavailable = False
            try:
                await asyncio.to_thread(self.store.load_usage)
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

    async def record_retrieved(self, refs: tuple[SkillRef, ...]) -> None:
        async with self._lock:
            self._assert_writable_state()
            try:
                await self._validate_refs(refs)
                await asyncio.to_thread(self.store.record_retrieved, refs)
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

    async def record_evaluations(self, items: tuple[tuple[SkillRef, str], ...]) -> None:
        async with self._lock:
            self._assert_writable_state()
            try:
                await self._validate_refs(tuple(ref for ref, _ in items))
                await asyncio.to_thread(self.store.record_evaluations, items)
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

    async def record_used(self, ref: SkillRef) -> None:
        async with self._lock:
            self._assert_writable_state()
            try:
                await self._validate_refs((ref,))
                await asyncio.to_thread(self.store.record_used, ref)
            except SkillStoreError as exc:
                self._state_degraded = True
                raise SkillRepositoryError("Skill usage state could not be saved") from exc

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
                now = datetime.now(UTC)
                pruned: list[SkillRef] = []
                for definition in tuple(current.by_identity.values()):
                    if definition.source.value != "generated":
                        continue
                    entry = usage.get(f"{definition.scope.value}:{definition.name}")
                    if entry is None or entry.version != definition.version:
                        continue
                    old = (now - definition.updated_at).days >= config.evolution.prune_unused_days
                    last_used = (
                        datetime.strptime(entry.last_used_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                            tzinfo=UTC
                        )
                        if entry.last_used_at
                        else None
                    )
                    unused = (
                        last_used is None
                        or (now - last_used).days >= config.evolution.prune_unused_days
                    )
                    dormant = (
                        old
                        and unused
                        and (
                            entry.version_used > 0
                            or entry.version_retrieve >= config.evolution.prune_min_retrieve
                        )
                    )
                    ineffective = (
                        entry.version_retrieve >= config.evolution.prune_min_retrieve
                        and entry.version_relevant >= config.evolution.prune_min_relevant
                        and entry.version_relevant / max(entry.version_used, 1)
                        > config.evolution.prune_relevant_used_ratio
                    )
                    if not (dormant or ineffective):
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

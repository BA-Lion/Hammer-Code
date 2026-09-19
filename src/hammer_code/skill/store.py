"""Atomic local Skill state, history, and recoverable current-file writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from hammer_code.session.session import atomic_write_text
from hammer_code.skill.models import SkillDefinition, SkillRef, SkillScope, SkillSource, UsageEntry
from hammer_code.skill.parser import SkillParseError, parse_skill, serialize_skill


class SkillStoreError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedSkillWrite:
    action: str
    definition: SkillDefinition
    expected_preimage: str | None
    evolution_note: str
    operation_id: str


@dataclass(frozen=True)
class CommitResult:
    operation_id: str
    definition: SkillDefinition
    changed: bool


@dataclass(frozen=True)
class RecoveryReport:
    recovered: int = 0
    quarantined: int = 0


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        return path.is_symlink()
    except OSError as exc:
        raise SkillStoreError("Skill path could not be inspected safely") from exc
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class SkillStore:
    """Synchronous store. Call only through :class:`SkillRepository`'s lock."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve() / ".hammer-code" / "skill"
        self.state_root = self.root / "state"

    @property
    def usage_path(self) -> Path:
        return self.state_root / "usage.json"

    @property
    def provenance_path(self) -> Path:
        return self.state_root / "provenance.jsonl"

    @property
    def provenance_summary_path(self) -> Path:
        return self.state_root / "skill_provenance.json"

    def _read_usage(self) -> dict[str, object]:
        if not self.usage_path.exists():
            return {"schema_version": 1, "skills": {}}
        try:
            payload = json.loads(self.usage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillStoreError("Skill usage state is invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise SkillStoreError("Skill usage state has an unsupported schema")
        if not isinstance(payload.get("skills"), dict):
            raise SkillStoreError("Skill usage state entries are invalid")
        return payload

    def load_usage(self) -> dict[str, UsageEntry]:
        result: dict[str, UsageEntry] = {}
        entries = cast(dict[object, object], self._read_usage()["skills"])
        for key, value in entries.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise SkillStoreError("Skill usage state entries are invalid")
            try:
                result[key] = UsageEntry(**value)
            except TypeError as exc:
                raise SkillStoreError("Skill usage state entries are invalid") from exc
        return result

    def _write_usage(self, entries: dict[str, UsageEntry]) -> None:
        payload = {
            "schema_version": 1,
            "skills": {key: entry.__dict__ for key, entry in sorted(entries.items())},
        }
        atomic_write_text(
            self.usage_path, json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        )

    def _record(self, refs: tuple[SkillRef, ...], event: str, reason: str | None = None) -> None:
        entries = self.load_usage()
        for ref in refs:
            key = f"{ref.scope.value}:{ref.name}"
            current = entries.get(key, UsageEntry())
            values = dict(current.__dict__)
            if current.version != ref.version:
                values.update(
                    version=ref.version,
                    version_retrieve=0,
                    version_relevant=0,
                    version_used=0,
                )
            values[event] += 1
            version_key = f"version_{event}"
            values[version_key] += 1
            timestamp_key = f"last_{event}d_at" if event == "retrieve" else f"last_{event}_at"
            values[timestamp_key] = _utc_stamp()
            if reason is not None:
                values["latest_reason"] = reason
            entries[key] = UsageEntry(**values)
        self._write_usage(entries)

    def record_retrieved(self, refs: tuple[SkillRef, ...]) -> None:
        self._record(refs, "retrieve")

    def record_evaluations(self, items: tuple[tuple[SkillRef, str], ...]) -> None:
        self._record(tuple(ref for ref, _ in items), "relevant")

    def record_used(self, ref: SkillRef) -> None:
        self._record((ref,), "used")

    def prepare_add(self, definition: SkillDefinition, note: str) -> PreparedSkillWrite:
        return PreparedSkillWrite("add", definition, None, note, str(uuid4()))

    def prepare_merge(
        self, definition: SkillDefinition, expected_preimage: str, note: str
    ) -> PreparedSkillWrite:
        return PreparedSkillWrite("merge", definition, expected_preimage, note, str(uuid4()))

    def _target(self, definition: SkillDefinition) -> Path:
        if self.root.exists() and _is_reparse(self.root):
            raise SkillStoreError("Skill root is a reparse point")
        root = self.root / definition.scope.value
        target = root / definition.skill_dir.name
        try:
            if root.exists() and _is_reparse(root):
                raise SkillStoreError("Skill scope root is a reparse point")
            if target.exists() and _is_reparse(target):
                raise SkillStoreError("Skill target is a reparse point")
            if target.resolve().parent != root.resolve():
                raise SkillStoreError("Skill target escapes its scope")
        except OSError as exc:
            raise SkillStoreError("Skill target cannot be resolved") from exc
        return target

    def _journal_path(self, operation_id: str) -> Path:
        return self.state_root / "transactions" / f"{operation_id}.json"

    def _write_journal(self, prepared: PreparedSkillWrite, phase: str, target_hash: str) -> None:
        value = {
            "operation_id": prepared.operation_id,
            "action": prepared.action,
            "phase": phase,
            "scope": prepared.definition.scope.value,
            "name": prepared.definition.name,
            "preimage": prepared.expected_preimage,
            "target_hash": target_hash,
            "evolution_note": prepared.evolution_note,
        }
        atomic_write_text(
            self._journal_path(prepared.operation_id), json.dumps(value, sort_keys=True) + "\n"
        )

    def _update_provenance_summary(self, definition: SkillDefinition, note: str) -> None:
        """Keep a bounded metadata-only view; bodies remain in current/history files."""
        try:
            current = (
                json.loads(self.provenance_summary_path.read_text(encoding="utf-8"))
                if self.provenance_summary_path.exists()
                else {"recent": [], "total_evolution_count": 0}
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillStoreError("Skill provenance summary is invalid") from exc
        recent = current.get("recent")
        total = current.get("total_evolution_count")
        if not isinstance(recent, list) or not isinstance(total, int):
            raise SkillStoreError("Skill provenance summary is invalid")
        frontmatter = {
            "name": definition.name,
            "description": definition.description,
            "version": definition.version,
            "created-at": definition.created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated-at": definition.updated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": definition.source.value,
            "context": definition.context.value,
        }
        summary = {
            "latest_evolve_time": _utc_stamp(),
            "total_evolution_count": total + 1,
            "recent": ([{"frontmatter": frontmatter, "evolution_note": note}] + recent)[:20],
        }
        atomic_write_text(
            self.provenance_summary_path,
            json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
        )

    def append_provenance(self, value: Mapping[str, object]) -> None:
        """Append a metadata-only audit event and make it durable before return."""
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_discard(self, reason: str) -> None:
        self.append_provenance(
            {
                "operation_id": str(uuid4()),
                "time": _utc_stamp(),
                "action": "discard",
                "status": "discarded",
                "reason": reason,
            }
        )

    def commit(self, prepared: PreparedSkillWrite) -> CommitResult:
        definition = prepared.definition
        target = self._target(definition)
        skill_path = target / "SKILL.md"
        if prepared.action == "add" and target.exists():
            raise SkillStoreError("Skill add target already exists")
        if prepared.action == "merge":
            if not skill_path.is_file() or _content_hash(skill_path) != prepared.expected_preimage:
                raise SkillStoreError("Skill changed before merge commit")
        content = serialize_skill(definition)
        target_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._write_journal(prepared, "prepared", target_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if prepared.action == "merge":
            history = self.root / "history" / definition.scope.value / definition.name
            try:
                old_version = parse_skill(skill_path, definition.scope).version
            except SkillParseError as exc:
                raise SkillStoreError("Skill current file cannot be snapshotted") from exc
            snapshot = history / f"{old_version}-{_utc_stamp().replace(':', '')}"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, snapshot)
            atomic_write_text(skill_path, content)
        else:
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
            try:
                atomic_write_text(staging / "SKILL.md", content)
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        if _content_hash(skill_path) != target_hash:
            raise SkillStoreError("Skill current file verification failed")
        self._write_journal(prepared, "current_committed", target_hash)
        # The current revision changes here.  Preserve lifetime counters while
        # resetting per-version counters before the catalog is republished.
        entries = self.load_usage()
        key = f"{definition.scope.value}:{definition.name}"
        previous = entries.get(key, UsageEntry())
        entries[key] = UsageEntry(
            source=definition.source,
            version=definition.version,
            retrieve=previous.retrieve,
            relevant=previous.relevant,
            used=previous.used,
            last_retrieved_at=previous.last_retrieved_at,
            last_relevant_at=previous.last_relevant_at,
            last_used_at=previous.last_used_at,
            latest_reason=previous.latest_reason,
        )
        self._write_usage(entries)
        provenance = {
            "operation_id": prepared.operation_id,
            "time": _utc_stamp(),
            "action": prepared.action,
            "scope": definition.scope.value,
            "name": definition.name,
            "version": definition.version,
            "evolution_note": prepared.evolution_note,
        }
        self.append_provenance(provenance)
        self._update_provenance_summary(definition, prepared.evolution_note)
        self._journal_path(prepared.operation_id).unlink(missing_ok=True)
        return CommitResult(prepared.operation_id, definition, True)

    def commit_prune(self, definition: SkillDefinition, expected_preimage: str) -> CommitResult:
        """Move only generated Skills into a recoverable prune directory."""
        if definition.source is not SkillSource.GENERATED:
            raise SkillStoreError("Only generated Skills may be automatically pruned")
        target = self._target(definition)
        skill_path = target / "SKILL.md"
        if not skill_path.is_file() or _content_hash(skill_path) != expected_preimage:
            raise SkillStoreError("Skill changed before prune commit")
        operation_id = str(uuid4())
        prepared = PreparedSkillWrite(
            "prune", definition, expected_preimage, "unused", operation_id
        )
        self._write_journal(prepared, "prepared", expected_preimage)
        destination = (
            self.root
            / "prune"
            / definition.scope.value
            / definition.name
            / _utc_stamp().replace(":", "")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise SkillStoreError("Prune destination already exists")
        os.replace(target, destination)
        self._write_journal(prepared, "current_committed", expected_preimage)
        self.append_provenance(
            {
                "operation_id": operation_id,
                "time": _utc_stamp(),
                "action": "prune",
                "scope": definition.scope.value,
                "name": definition.name,
                "version": definition.version,
                "status": "committed",
            }
        )
        self._journal_path(operation_id).unlink(missing_ok=True)
        return CommitResult(operation_id, definition, True)

    def restore_pruned(self, definition: SkillDefinition, pruned_at: str) -> None:
        """Restore one exact pruned directory only when it cannot overwrite a current Skill."""
        target = self._target(definition)
        source = self.root / "prune" / definition.scope.value / definition.name / pruned_at
        try:
            if source.resolve().parent.parent.parent != (self.root / "prune").resolve():
                raise SkillStoreError("Prune restore source escapes Skill root")
        except OSError as exc:
            raise SkillStoreError("Prune restore source cannot be resolved") from exc
        if target.exists() or not (source / "SKILL.md").is_file():
            raise SkillStoreError("Pruned Skill cannot be restored safely")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)

    def recover_pending(self) -> RecoveryReport:
        root = self.state_root / "transactions"
        if not root.exists():
            return RecoveryReport()
        recovered = 0
        quarantined = 0
        for path in root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("phase") == "current_committed":
                    scope = SkillScope(data["scope"])
                    skill_path = self.root / scope.value / str(data["name"]) / "SKILL.md"
                    definition = parse_skill(skill_path, scope)
                    entries = self.load_usage()
                    key = f"{scope.value}:{definition.name}"
                    entry = entries.get(key, UsageEntry())
                    if entry.version != definition.version or entry.source != definition.source:
                        entries[key] = UsageEntry(
                            source=definition.source,
                            version=definition.version,
                            retrieve=entry.retrieve,
                            relevant=entry.relevant,
                            used=entry.used,
                            last_retrieved_at=entry.last_retrieved_at,
                            last_relevant_at=entry.last_relevant_at,
                            last_used_at=entry.last_used_at,
                            latest_reason=entry.latest_reason,
                        )
                        self._write_usage(entries)
                    path.unlink()
                    recovered += 1
                else:
                    quarantine = root / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(path, quarantine / path.name)
                    quarantined += 1
            except (KeyError, OSError, SkillParseError, ValueError, json.JSONDecodeError):
                if path.exists():
                    quarantine = root / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(path, quarantine / path.name)
                quarantined += 1
        return RecoveryReport(recovered, quarantined)

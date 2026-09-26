"""Durable Skill facts and recoverable local Skill file operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hammer_code.session.session import atomic_write_text
from hammer_code.skill.models import (
    AdjudicatedRetrieval,
    ApplicationEvaluation,
    RelevanceEvaluation,
    SkillDefinition,
    SkillRef,
    SkillScope,
    SkillSource,
    UsageEntry,
)
from hammer_code.skill.parser import parse_skill, serialize_skill


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
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _entry_payload(entry: UsageEntry) -> dict[str, object]:
    value = asdict(entry)
    if entry.source is not None:
        value["source"] = entry.source.value
    value["adjudicated_retrievals"] = [asdict(item) for item in entry.adjudicated_retrievals]
    return value


def _entry(value: object) -> UsageEntry:
    if not isinstance(value, dict):
        raise SkillStoreError("Skill usage state entries are invalid")
    item = dict(value)
    try:
        if item.get("source") is not None:
            item["source"] = SkillSource(item["source"])
        item["adjudicated_retrievals"] = tuple(
            AdjudicatedRetrieval(**record) for record in item.get("adjudicated_retrievals", ())
        )
        return UsageEntry(**item)
    except (TypeError, ValueError) as exc:
        raise SkillStoreError("Skill usage state entries are invalid") from exc


class SkillStore:
    """Synchronous store; :class:`SkillRepository` owns its process lock."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve() / ".hammer-code" / "skill"
        self.state_root = self.root / "state"

    @property
    def usage_path(self) -> Path:
        return self.state_root / "usage.json"

    @property
    def record_path(self) -> Path:
        return self.state_root / "record.jsonl"

    @property
    def provenance_path(self) -> Path:
        return self.state_root / "provenance.jsonl"

    @property
    def provenance_summary_path(self) -> Path:
        return self.state_root / "skill_provenance.json"

    def _read_usage(self) -> dict[str, object]:
        if not self.usage_path.exists():
            return {"schema_version": 2, "skills": {}}
        try:
            value = json.loads(self.usage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillStoreError("Skill usage state is invalid") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 2:
            raise SkillStoreError("Skill usage state has an unsupported schema")
        if not isinstance(value.get("skills"), dict):
            raise SkillStoreError("Skill usage state entries are invalid")
        return value

    def load_usage(self) -> dict[str, UsageEntry]:
        raw = self._read_usage()["skills"]
        assert isinstance(raw, dict)
        return {str(key): _entry(value) for key, value in raw.items()}

    def _write_usage(self, entries: Mapping[str, UsageEntry]) -> None:
        atomic_write_text(
            self.usage_path,
            json.dumps(
                {
                    "schema_version": 2,
                    "skills": {
                        key: _entry_payload(entry) for key, entry in sorted(entries.items())
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )

    def _event(
        self,
        event_type: str,
        *,
        ref: SkillRef | None = None,
        source: SkillSource | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        operation_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if event_type not in {"retrieve", "loaded", "evaluation", "evolve", "prune", "recover"}:
            raise SkillStoreError("Unknown Skill record event")
        value: dict[str, object] = {
            "schema_version": 2,
            "event_id": str(uuid4()),
            "occurred_at": _utc_stamp(),
            "event_type": event_type,
            "payload": dict(payload or {}),
        }
        for key, item in (
            ("session_id", session_id),
            ("turn_id", turn_id),
            ("operation_id", operation_id),
        ):
            if item is not None:
                value[key] = item
        if ref is not None:
            value.update(scope=ref.scope.value, name=ref.name, version=ref.version)
        if source is not None:
            value["source"] = source.value
        return value

    def append_record(self, event: Mapping[str, object]) -> str:
        required = {"schema_version", "event_id", "occurred_at", "event_type", "payload"}
        allowed = required | {
            "session_id",
            "turn_id",
            "operation_id",
            "scope",
            "name",
            "version",
            "source",
        }
        if (
            set(event).difference(allowed)
            or not required.issubset(event)
            or event.get("schema_version") != 2
        ):
            raise SkillStoreError("Skill record event has an invalid envelope")
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.record_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SkillStoreError("Skill record could not be appended") from exc
        return str(event["event_id"])

    def _scan_records(self) -> tuple[dict[str, object], ...]:
        if not self.record_path.exists():
            return ()
        raw = self.record_path.read_bytes()
        records: list[dict[str, object]] = []
        position = 0
        for line in raw.splitlines(keepends=True):
            position += len(line)
            if not line.endswith((b"\n", b"\r")):
                with self.record_path.open("r+b") as handle:
                    handle.truncate(position - len(line))
                break
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SkillStoreError("Skill record contains an invalid line") from exc
            if not isinstance(value, dict) or value.get("schema_version") != 2:
                raise SkillStoreError("Skill record contains an unsupported event")
            records.append(value)
        return tuple(records)

    @staticmethod
    def _ref(value: Mapping[str, object]) -> SkillRef | None:
        try:
            return SkillRef(
                SkillScope(str(value["scope"])), str(value["name"]), str(value["version"])
            )
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _key(ref: SkillRef) -> str:
        return f"{ref.scope.value}:{ref.name}"

    @staticmethod
    def _increment(entry: UsageEntry, name: str, stamp: str) -> UsageEntry:
        values = asdict(entry)
        values[name] += 1
        values[f"version_{name}"] += 1
        values["last_retrieved_at" if name == "retrieve" else f"last_{name}_at"] = stamp
        return UsageEntry(**values)

    def rebuild_usage(self, definitions: Iterable[SkillDefinition] = ()) -> dict[str, UsageEntry]:
        entries: dict[str, UsageEntry] = {}
        retrieves: dict[str, tuple[SkillRef, str, str | None, str]] = {}
        records = sorted(
            self._scan_records(),
            key=lambda item: (str(item.get("occurred_at")), str(item.get("event_id"))),
        )
        for event in records:
            event_type, ref, stamp = (
                event.get("event_type"),
                self._ref(event),
                str(event.get("occurred_at")),
            )
            if ref is not None and event_type in {"retrieve", "loaded"}:
                key = self._key(ref)
                source = SkillSource(str(event.get("source", "imported")))
                current = entries.get(
                    key, UsageEntry(source=source, version=ref.version, version_updated_at=stamp)
                )
                if current.version != ref.version:
                    current = UsageEntry(
                        source=current.source,
                        version=ref.version,
                        version_updated_at=stamp,
                        retrieve=current.retrieve,
                        loaded=current.loaded,
                        relevant=current.relevant,
                        applied=current.applied,
                        last_retrieved_at=current.last_retrieved_at,
                        last_loaded_at=current.last_loaded_at,
                        last_relevant_at=current.last_relevant_at,
                        last_applied_at=current.last_applied_at,
                    )
                entries[key] = self._increment(current, str(event_type), stamp)
                if event_type == "retrieve":
                    turn_id = event.get("turn_id")
                    retrieves[str(event["event_id"])] = (
                        ref,
                        stamp,
                        turn_id if isinstance(turn_id, str) else None,
                        key,
                    )
                continue
            if event_type != "evaluation":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("status") != "completed":
                continue
            for field, signal in (
                ("relevance_evaluations", "relevant"),
                ("application_evaluations", "applied"),
            ):
                for item in payload.get(field, []):
                    if not isinstance(item, dict) or not item.get(signal):
                        continue
                    item_ref = self._ref(item)
                    if item_ref is None:
                        continue
                    current = entries.get(self._key(item_ref))
                    if current is None or current.version != item_ref.version:
                        continue
                    current = self._increment(current, signal, stamp)
                    if signal == "applied":
                        current = UsageEntry(
                            **{**asdict(current), "version_last_applied_at": stamp}
                        )
                    entries[self._key(item_ref)] = current
            for event_id in payload.get("retrieve_event_ids", []):
                found = retrieves.get(str(event_id))
                if found is None:
                    continue
                ref, retrieved_at, turn_id, key = found
                current = entries.get(key)
                if current is None or current.version != ref.version:
                    continue
                observation = AdjudicatedRetrieval(
                    str(event_id), turn_id, retrieved_at, str(event["event_id"])
                )
                entries[key] = UsageEntry(
                    **{
                        **asdict(current),
                        "adjudicated_retrievals": (current.adjudicated_retrievals + (observation,))[
                            -5:
                        ],
                    }
                )
        for definition in definitions:
            key = f"{definition.scope.value}:{definition.name}"
            current = entries.get(key)
            stamp = definition.updated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            if current is None or current.version != definition.version:
                entries[key] = UsageEntry(
                    source=definition.source,
                    version=definition.version,
                    version_updated_at=stamp,
                    retrieve=0 if current is None else current.retrieve,
                    loaded=0 if current is None else current.loaded,
                    relevant=0 if current is None else current.relevant,
                    applied=0 if current is None else current.applied,
                    last_retrieved_at=None if current is None else current.last_retrieved_at,
                    last_loaded_at=None if current is None else current.last_loaded_at,
                    last_relevant_at=None if current is None else current.last_relevant_at,
                    last_applied_at=None if current is None else current.last_applied_at,
                )
            else:
                entries[key] = UsageEntry(
                    **{**asdict(current), "source": definition.source, "version_updated_at": stamp}
                )
        self._write_usage(entries)
        return entries

    def record_retrieved(
        self,
        refs: tuple[SkillRef, ...],
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        catalog_generation: int = 0,
    ) -> tuple[str, ...]:
        return tuple(
            self.append_record(
                self._event(
                    "retrieve",
                    ref=ref,
                    session_id=session_id,
                    turn_id=turn_id,
                    payload={"catalog_generation": catalog_generation},
                )
            )
            for ref in refs
        )

    def record_loaded(
        self,
        ref: SkillRef,
        *,
        invocation_kind: str,
        session_id: str | None = None,
        turn_id: str | None = None,
        catalog_generation: int = 0,
    ) -> str:
        return self.append_record(
            self._event(
                "loaded",
                ref=ref,
                session_id=session_id,
                turn_id=turn_id,
                payload={
                    "catalog_generation": catalog_generation,
                    "invocation_kind": invocation_kind,
                },
            )
        )

    def record_evaluation(
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
        payload: dict[str, object] = {
            "retrieve_event_ids": list(retrieve_event_ids),
            "loaded_event_ids": list(loaded_event_ids),
        }
        if failure_stage:
            payload.update(status="failed", stage_code=failure_stage)
        else:
            payload.update(
                status="completed",
                relevance_evaluations=[item.model_dump(mode="json") for item in relevance],
                application_evaluations=[item.model_dump(mode="json") for item in application],
            )
        return self.append_record(
            self._event("evaluation", session_id=session_id, turn_id=turn_id, payload=payload)
        )

    def append_provenance(self, value: Mapping[str, object]) -> None:
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _update_provenance_summary(self, definition: SkillDefinition, note: str) -> None:
        try:
            previous = (
                json.loads(self.provenance_summary_path.read_text(encoding="utf-8"))
                if self.provenance_summary_path.exists()
                else {"recent": [], "total_evolution_count": 0}
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillStoreError("Skill provenance summary is invalid") from exc
        recent = previous.get("recent", [])
        total = previous.get("total_evolution_count", 0)
        if not isinstance(recent, list) or not isinstance(total, int):
            raise SkillStoreError("Skill provenance summary is invalid")
        frontmatter = {
            "name": definition.name,
            "description": definition.description,
            "version": definition.version,
            "source": definition.source.value,
            "context": definition.context.value,
        }
        atomic_write_text(
            self.provenance_summary_path,
            json.dumps(
                {
                    "latest_evolve_time": _utc_stamp(),
                    "total_evolution_count": total + 1,
                    "recent": ([{"frontmatter": frontmatter, "evolution_note": note}] + recent)[
                        :20
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )

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

    def _target(self, definition: SkillDefinition) -> Path:
        if self.root.exists() and _is_reparse(self.root):
            raise SkillStoreError("Skill root is a reparse point")
        root, target = (
            self.root / definition.scope.value,
            self.root / definition.scope.value / definition.skill_dir.name,
        )
        if (root.exists() and _is_reparse(root)) or (target.exists() and _is_reparse(target)):
            raise SkillStoreError("Skill target is a reparse point")
        if target.resolve().parent != root.resolve():
            raise SkillStoreError("Skill target escapes its scope")
        return target

    def _journal_path(self, operation_id: str) -> Path:
        return self.state_root / "transactions" / f"{operation_id}.json"

    def _write_journal(self, prepared: PreparedSkillWrite, phase: str, target_hash: str) -> None:
        atomic_write_text(
            self._journal_path(prepared.operation_id),
            json.dumps(
                {
                    "operation_id": prepared.operation_id,
                    "action": prepared.action,
                    "phase": phase,
                    "scope": prepared.definition.scope.value,
                    "name": prepared.definition.name,
                    "version": prepared.definition.version,
                    "source": prepared.definition.source.value,
                    "preimage": prepared.expected_preimage,
                    "target_hash": target_hash,
                    "evolution_note": prepared.evolution_note,
                },
                sort_keys=True,
            )
            + "\n",
        )

    def prepare_add(self, definition: SkillDefinition, note: str) -> PreparedSkillWrite:
        return PreparedSkillWrite("add", definition, None, note, str(uuid4()))

    def prepare_merge(
        self, definition: SkillDefinition, expected_preimage: str, note: str
    ) -> PreparedSkillWrite:
        return PreparedSkillWrite("merge", definition, expected_preimage, note, str(uuid4()))

    def commit(self, prepared: PreparedSkillWrite) -> CommitResult:
        definition, target = prepared.definition, self._target(prepared.definition)
        skill_path = target / "SKILL.md"
        if prepared.action == "add" and target.exists():
            raise SkillStoreError("Skill add target already exists")
        if prepared.action == "merge" and (
            not skill_path.is_file() or _content_hash(skill_path) != prepared.expected_preimage
        ):
            raise SkillStoreError("Skill changed before merge commit")
        content = serialize_skill(definition)
        target_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._write_journal(prepared, "prepared", target_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if prepared.action == "merge":
            old_version = parse_skill(skill_path, definition.scope).version
            history = (
                self.root
                / "history"
                / definition.scope.value
                / definition.name
                / f"{old_version}-{_utc_stamp().replace(':', '')}"
            )
            history.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, history)
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
        self.append_provenance(
            {
                "operation_id": prepared.operation_id,
                "time": _utc_stamp(),
                "action": prepared.action,
                "scope": definition.scope.value,
                "name": definition.name,
                "version": definition.version,
                "evolution_note": prepared.evolution_note,
            }
        )
        self._update_provenance_summary(definition, prepared.evolution_note)
        self._write_journal(prepared, "provenance_committed", target_hash)
        self.append_record(
            self._event(
                "evolve",
                ref=definition.ref,
                source=definition.source,
                operation_id=prepared.operation_id,
                payload={
                    "action": prepared.action,
                    "provenance_operation_id": prepared.operation_id,
                },
            )
        )
        self._write_journal(prepared, "record_committed", target_hash)
        self.rebuild_usage((definition,))
        self._journal_path(prepared.operation_id).unlink(missing_ok=True)
        return CommitResult(prepared.operation_id, definition, True)

    def commit_prune(self, definition: SkillDefinition, expected_preimage: str) -> CommitResult:
        if definition.source is not SkillSource.GENERATED:
            raise SkillStoreError("Only generated Skills may be automatically pruned")
        target, skill_path = self._target(definition), self._target(definition) / "SKILL.md"
        if not skill_path.is_file() or _content_hash(skill_path) != expected_preimage:
            raise SkillStoreError("Skill changed before prune commit")
        operation_id = str(uuid4())
        prepared = PreparedSkillWrite(
            "prune", definition, expected_preimage, "continuous-unapplied", operation_id
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
        os.replace(target, destination)
        self._write_journal(prepared, "current_committed", expected_preimage)
        self.append_record(
            self._event(
                "prune",
                ref=definition.ref,
                source=definition.source,
                operation_id=operation_id,
                payload={"prune_identity": destination.name},
            )
        )
        self._write_journal(prepared, "record_committed", expected_preimage)
        self.rebuild_usage()
        self._journal_path(operation_id).unlink(missing_ok=True)
        return CommitResult(operation_id, definition, True)

    def restore_pruned(self, definition: SkillDefinition, pruned_at: str) -> None:
        target, source = (
            self._target(definition),
            self.root / "prune" / definition.scope.value / definition.name / pruned_at,
        )
        if target.exists() or not (source / "SKILL.md").is_file():
            raise SkillStoreError("Pruned Skill cannot be restored safely")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        self.append_record(
            self._event(
                "recover",
                ref=definition.ref,
                source=definition.source,
                payload={"prune_identity": pruned_at},
            )
        )

    def recover_pending(self) -> RecoveryReport:
        root = self.state_root / "transactions"
        if not root.exists():
            return RecoveryReport()
        recovered = quarantined = 0
        for path in root.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                phase = value["phase"]
                if phase in {"record_committed", "provenance_committed", "current_committed"}:
                    path.unlink()
                    recovered += 1
                else:
                    quarantine = root / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(path, quarantine / path.name)
                    quarantined += 1
            except (OSError, KeyError, json.JSONDecodeError):
                quarantine = root / "quarantine"
                quarantine.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    os.replace(path, quarantine / path.name)
                quarantined += 1
        return RecoveryReport(recovered, quarantined)

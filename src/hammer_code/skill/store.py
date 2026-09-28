"""Durable Skill facts and recoverable local Skill file operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
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
    prune_identity: str | None = None


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
    def _increment_for_ref(
        entry: UsageEntry, name: str, stamp: str, version: str | None
    ) -> UsageEntry:
        values = asdict(entry)
        values[name] += 1
        if entry.version == version:
            values[f"version_{name}"] += 1
            if name == "applied":
                values["version_last_applied_at"] = stamp
        values["last_retrieved_at" if name == "retrieve" else f"last_{name}_at"] = stamp
        values["adjudicated_retrievals"] = entry.adjudicated_retrievals
        return UsageEntry(**values)

    @staticmethod
    def _version_entry(current: UsageEntry | None, definition: SkillDefinition) -> UsageEntry:
        stamp = definition.updated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if current is not None and current.version == definition.version:
            return replace(current, source=definition.source, version_updated_at=stamp)
        return UsageEntry(
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

    def sync_usage(self, definitions: Iterable[SkillDefinition]) -> dict[str, UsageEntry]:
        """Reconcile current versions without reconstructing lifetime from records."""
        entries = self.load_usage()
        for definition in definitions:
            key = self._key(definition.ref)
            entries[key] = self._version_entry(entries.get(key), definition)
        self._write_usage(entries)
        return entries

    def _apply_usage_events(self, events: Iterable[Mapping[str, object]]) -> None:
        """Reduce only newly appended facts against the persisted version boundary.

        A record/usage write gap remains a degraded auxiliary-state condition;
        this reducer does not replay history or claim to repair unrelated gaps.
        """
        entries = self.load_usage()
        for event in events:
            event_type, ref, stamp = (
                event.get("event_type"),
                self._ref(event),
                str(event.get("occurred_at")),
            )
            if ref is not None and event_type in {"retrieve", "loaded"}:
                key = self._key(ref)
                current = entries.get(key)
                if current is None:
                    continue
                entries[key] = self._increment_for_ref(current, str(event_type), stamp, ref.version)
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
                    if current is None:
                        continue
                    entries[self._key(item_ref)] = self._increment_for_ref(
                        current, signal, stamp, item_ref.version
                    )
            wanted = set(payload.get("retrieve_event_ids", []))
            # Exact linked retrieval facts only; no record index or usage reconstruction.
            for retrieval in self._scan_records() if wanted else ():
                if (
                    retrieval.get("event_type") != "retrieve"
                    or retrieval.get("event_id") not in wanted
                ):
                    continue
                ref = self._ref(retrieval)
                if ref is None:
                    continue
                key = self._key(ref)
                current = entries.get(key)
                if current is None or current.version != ref.version:
                    continue
                turn_id = retrieval.get("turn_id")
                observation = AdjudicatedRetrieval(
                    str(retrieval["event_id"]),
                    turn_id if isinstance(turn_id, str) else None,
                    str(retrieval["occurred_at"]),
                    str(event["event_id"]),
                )
                window = {item.retrieve_event_id: item for item in current.adjudicated_retrievals}
                window[observation.retrieve_event_id] = observation
                entries[key] = replace(
                    current,
                    adjudicated_retrievals=tuple(
                        sorted(
                            window.values(),
                            key=lambda item: (item.retrieved_at, item.retrieve_event_id),
                        )
                    )[-5:],
                )
        self._write_usage(entries)

    def _record_usage_event(self, event: Mapping[str, object]) -> str:
        event_id = self.append_record(event)
        self._apply_usage_events((event,))
        return event_id

    def record_retrieved(
        self,
        refs: tuple[SkillRef, ...],
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        catalog_generation: int = 0,
    ) -> tuple[str, ...]:
        return tuple(
            self._record_usage_event(
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
        return self._record_usage_event(
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
        return self._record_usage_event(
            self._event("evaluation", session_id=session_id, turn_id=turn_id, payload=payload)
        )

    def append_provenance(self, value: Mapping[str, object]) -> None:
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _update_provenance_summary(
        self, definition: SkillDefinition, note: str, operation_id: str
    ) -> None:
        try:
            previous = (
                json.loads(self.provenance_summary_path.read_text(encoding="utf-8"))
                if self.provenance_summary_path.exists()
                else {"recent": [], "total_evolution_count": 0}
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillStoreError("Skill provenance summary is invalid") from exc
        if not isinstance(previous, dict):
            raise SkillStoreError("Skill provenance summary is invalid")
        recent = previous.get("recent", [])
        total = previous.get("total_evolution_count", 0)
        if (
            not isinstance(recent, list)
            or not all(isinstance(item, dict) for item in recent)
            or not isinstance(total, int)
        ):
            raise SkillStoreError("Skill provenance summary is invalid")
        frontmatter = {
            "name": definition.name,
            "description": definition.description,
            "version": definition.version,
            "source": definition.source.value,
            "context": definition.context.value,
        }
        journal_path = self._journal_path(operation_id)
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        current_hash = (
            _content_hash(self.provenance_summary_path)
            if self.provenance_summary_path.exists()
            else None
        )
        if "summary_content" in journal:
            content = journal["summary_content"]
            if not isinstance(content, str) or hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest() != journal.get("summary_hash"):
                raise SkillStoreError("Skill summary journal is invalid")
            planned = json.loads(content)
            if (
                not isinstance(planned, dict)
                or not isinstance(planned.get("recent"), list)
                or not planned["recent"]
                or planned["recent"][0]
                != {
                    "operation_id": operation_id,
                    "frontmatter": frontmatter,
                    "evolution_note": note,
                }
            ):
                raise SkillStoreError("Skill summary journal identity is invalid")
            if current_hash == journal["summary_hash"]:
                return
            if current_hash != journal.get("summary_preimage"):
                if any(item.get("operation_id") == operation_id for item in recent):
                    return
                raise SkillStoreError("Skill summary no longer proves this operation")
        else:
            if any(item.get("operation_id") == operation_id for item in recent):
                return
            if "skill_dir" not in journal and self.provenance_summary_path.exists():
                raise SkillStoreError("Legacy Skill summary cannot prove this operation")
            content = (
                json.dumps(
                    {
                        "latest_evolve_time": _utc_stamp(),
                        "total_evolution_count": total + 1,
                        "recent": (
                            [
                                {
                                    "operation_id": operation_id,
                                    "frontmatter": frontmatter,
                                    "evolution_note": note,
                                }
                            ]
                            + recent
                        )[:20],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            journal.update(
                summary_preimage=current_hash,
                summary_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                summary_content=content,
            )
            # Bind the bounded auxiliary summary before its atomic replacement.
            atomic_write_text(journal_path, json.dumps(journal, sort_keys=True) + "\n")
        atomic_write_text(self.provenance_summary_path, content)

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
        if any(part.exists() and _is_reparse(part) for part in (self.root.parent, self.root)):
            raise SkillStoreError("Skill root is a reparse point")
        root, target = (
            self.root / definition.scope.value,
            self.root / definition.scope.value / definition.skill_dir.name,
        )
        if (root.exists() and _is_reparse(root)) or (target.exists() and _is_reparse(target)):
            raise SkillStoreError("Skill target is a reparse point")
        if target.resolve().parent != root.resolve():
            raise SkillStoreError("Skill target escapes its scope")
        if definition.skill_dir.resolve() != target.resolve():
            raise SkillStoreError("Skill target does not match its definition")
        skill_path = target / "SKILL.md"
        if skill_path.exists() and _is_reparse(skill_path):
            raise SkillStoreError("Skill file is a reparse point")
        return target

    def _journal_path(self, operation_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_-]+", operation_id) is None:
            raise SkillStoreError("Skill operation identity is invalid")
        if any(
            part.exists() and _is_reparse(part)
            for part in (
                self.root.parent,
                self.root,
                self.state_root,
                self.state_root / "transactions",
            )
        ):
            raise SkillStoreError("Skill journal path is a reparse point")
        return self.state_root / "transactions" / f"{operation_id}.json"

    def _write_journal(self, prepared: PreparedSkillWrite, phase: str, target_hash: str) -> None:
        summary: dict[str, object] = {}
        path = self._journal_path(prepared.operation_id)
        if phase != "prepared" and path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            summary = {
                key: previous[key]
                for key in ("summary_content", "summary_preimage", "summary_hash")
                if key in previous
            }
        atomic_write_text(
            path,
            json.dumps(
                {
                    "operation_id": prepared.operation_id,
                    "action": prepared.action,
                    "phase": phase,
                    "scope": prepared.definition.scope.value,
                    "name": prepared.definition.name,
                    "skill_dir": prepared.definition.skill_dir.name,
                    "version": prepared.definition.version,
                    "source": prepared.definition.source.value,
                    "preimage": prepared.expected_preimage,
                    "target_hash": target_hash,
                    "evolution_note": prepared.evolution_note,
                    "prune_identity": prepared.prune_identity,
                    **summary,
                },
                sort_keys=True,
            )
            + "\n",
        )

    def _operation_exists(
        self, path: Path, prepared: PreparedSkillWrite, event_type: str | None = None
    ) -> bool:
        """Look up only this operation; matching IDs with conflicting facts fail closed."""
        if not path.exists():
            return False
        found = False
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SkillStoreError("Skill operation evidence is invalid")
                if value.get("operation_id") != prepared.operation_id:
                    continue
                if (
                    self._ref(value) != prepared.definition.ref
                    or (event_type is not None and value.get("event_type") != event_type)
                    or (event_type is None and value.get("action") != prepared.action)
                ):
                    raise SkillStoreError("Skill operation evidence conflicts with journal")
                if event_type is not None:
                    payload = value.get("payload")
                    if (
                        value.get("source") != prepared.definition.source.value
                        or not isinstance(payload, dict)
                        or (
                            event_type == "evolve"
                            and (
                                payload.get("action") != prepared.action
                                or payload.get("provenance_operation_id") != prepared.operation_id
                            )
                        )
                        or (
                            event_type != "evolve"
                            and payload.get("prune_identity") != prepared.prune_identity
                        )
                    ):
                        raise SkillStoreError(
                            "Skill operation record payload conflicts with journal"
                        )
                found = True
        return found

    def _finish_operation(self, prepared: PreparedSkillWrite, target_hash: str) -> None:
        definition = prepared.definition
        if prepared.action in {"add", "merge"}:
            if not self._operation_exists(self.provenance_path, prepared):
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
            self._update_provenance_summary(
                definition, prepared.evolution_note, prepared.operation_id
            )
            self._write_journal(prepared, "provenance_committed", target_hash)
        event_type = "evolve" if prepared.action in {"add", "merge"} else prepared.action
        if not self._operation_exists(self.record_path, prepared, event_type):
            self.append_record(
                self._event(
                    event_type,
                    ref=definition.ref,
                    source=definition.source,
                    operation_id=prepared.operation_id,
                    payload={
                        "action": prepared.action,
                        "provenance_operation_id": prepared.operation_id,
                    }
                    if event_type == "evolve"
                    else {"prune_identity": prepared.prune_identity},
                )
            )
        self._write_journal(prepared, "record_committed", target_hash)
        if event_type == "evolve":
            self.sync_usage((definition,))
        # Prune/recover keep the same usage entry, including late evaluations.
        self._journal_path(prepared.operation_id).unlink(missing_ok=True)

    def _pruned_target(self, prepared: PreparedSkillWrite) -> Path:
        identity = prepared.prune_identity
        if identity is None or re.fullmatch(r"[A-Za-z0-9_-]+", identity) is None:
            raise SkillStoreError("Prune identity is invalid")
        path = (
            self.root
            / "prune"
            / prepared.definition.scope.value
            / prepared.definition.name
            / identity
        )
        for part in (self.root / "prune", path.parent.parent, path.parent, path, path / "SKILL.md"):
            if part.exists() and _is_reparse(part):
                raise SkillStoreError("Prune path is a reparse point")
        return path

    def _discard_unpublished_staging(self, prepared: PreparedSkillWrite, target_hash: str) -> None:
        root = self.state_root / "staging"
        if not root.exists():
            return
        if _is_reparse(root):
            raise SkillStoreError("Skill staging path is a reparse point")
        for path in root.glob(f"{prepared.operation_id}.*"):
            if (
                _is_reparse(path)
                or path.resolve().parent != root.resolve()
                or {item.name for item in path.iterdir()} != {"SKILL.md"}
            ):
                raise SkillStoreError("Unpublished Skill staging identity is ambiguous")
            self._verify_definition(path / "SKILL.md", prepared, target_hash)
            shutil.rmtree(path)

    @staticmethod
    def _verify_definition(
        path: Path, prepared: PreparedSkillWrite, target_hash: str
    ) -> SkillDefinition:
        if not path.is_file() or _is_reparse(path) or _content_hash(path) != target_hash:
            raise SkillStoreError("Skill operation target hash does not match")
        definition = parse_skill(path, prepared.definition.scope)
        if (
            definition.ref != prepared.definition.ref
            or definition.source != prepared.definition.source
        ):
            raise SkillStoreError("Skill operation target identity does not match")
        return definition

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
            staging_root = self.state_root / "staging"
            if staging_root.exists() and _is_reparse(staging_root):
                raise SkillStoreError("Skill staging path is a reparse point")
            staging_root.mkdir(parents=True, exist_ok=True)
            # Unpublished content must not be discoverable as a catalog child
            # if the process exits before the directory replacement.
            staging = Path(tempfile.mkdtemp(prefix=f"{prepared.operation_id}.", dir=staging_root))
            try:
                atomic_write_text(staging / "SKILL.md", content)
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        if _content_hash(skill_path) != target_hash:
            raise SkillStoreError("Skill current file verification failed")
        self._write_journal(prepared, "current_committed", target_hash)
        verified = self._verify_definition(skill_path, prepared, target_hash)
        self._finish_operation(replace(prepared, definition=verified), target_hash)
        return CommitResult(prepared.operation_id, definition, True)

    def commit_prune(
        self,
        definition: SkillDefinition,
        expected_preimage: str,
        *,
        guard: Callable[[], bool] | None = None,
    ) -> CommitResult:
        if definition.source is not SkillSource.GENERATED:
            raise SkillStoreError("Only generated Skills may be automatically pruned")
        target, skill_path = self._target(definition), self._target(definition) / "SKILL.md"
        if not skill_path.is_file() or _content_hash(skill_path) != expected_preimage:
            raise SkillStoreError("Skill changed before prune commit")
        operation_id = str(uuid4())
        prepared = PreparedSkillWrite(
            "prune",
            definition,
            expected_preimage,
            "continuous-unapplied",
            operation_id,
            _utc_stamp().replace(":", "") + "-" + operation_id,
        )
        self._verify_definition(skill_path, prepared, expected_preimage)
        destination = self._pruned_target(prepared)
        self._write_journal(prepared, "prepared", expected_preimage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (guard is not None and not guard()) or _content_hash(skill_path) != expected_preimage:
            self._journal_path(operation_id).unlink(missing_ok=True)
            raise SkillStoreError("Skill prune conditions changed before move")
        os.replace(target, destination)
        self._write_journal(prepared, "current_committed", expected_preimage)
        self._finish_operation(prepared, expected_preimage)
        return CommitResult(operation_id, definition, True)

    def restore_pruned(self, definition: SkillDefinition, pruned_at: str) -> None:
        prepared = PreparedSkillWrite(
            "recover",
            definition,
            None,
            "manual-recover",
            str(uuid4()),
            pruned_at,
        )
        target, source = self._target(definition), self._pruned_target(prepared)
        if target.exists() or not (source / "SKILL.md").is_file():
            raise SkillStoreError("Pruned Skill cannot be restored safely")
        target_hash = _content_hash(source / "SKILL.md")
        self._verify_definition(source / "SKILL.md", prepared, target_hash)
        prepared = replace(prepared, expected_preimage=target_hash)
        self._write_journal(prepared, "prepared", target_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        self._write_journal(prepared, "current_committed", target_hash)
        self._finish_operation(prepared, target_hash)

    def _recover_operation(self, path: Path) -> None:
        if _is_reparse(path):
            raise SkillStoreError("Skill journal is a reparse point")
        value = json.loads(path.read_text(encoding="utf-8"))
        operation_id = value["operation_id"]
        if self._journal_path(operation_id) != path:
            raise SkillStoreError("Skill journal filename does not match its identity")
        action, phase = value["action"], value["phase"]
        if action not in {"add", "merge", "prune", "recover"} or phase not in {
            "prepared",
            "current_committed",
            "provenance_committed",
            "record_committed",
        }:
            raise SkillStoreError("Skill journal action or phase is invalid")
        name, directory = value["name"], value.get("skill_dir", value["name"])
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or not isinstance(directory, str)
            or directory in {"", ".", ".."}
            or any(char in directory for char in "/\\:\x00")
        ):
            raise SkillStoreError("Skill journal target identity is invalid")
        if (
            not isinstance(value["version"], str)
            or re.fullmatch(
                r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value["version"]
            )
            is None
        ):
            raise SkillStoreError("Skill journal version is invalid")
        target_hash, preimage = value["target_hash"], value["preimage"]
        if not isinstance(target_hash, str) or re.fullmatch(r"[a-f0-9]{64}", target_hash) is None:
            raise SkillStoreError("Skill journal hash is invalid")
        if action == "add":
            if preimage is not None:
                raise SkillStoreError("Skill add journal has a preimage")
        elif not isinstance(preimage, str) or re.fullmatch(r"[a-f0-9]{64}", preimage) is None:
            raise SkillStoreError("Skill journal preimage is invalid")
        scope, source = SkillScope(value["scope"]), SkillSource(value["source"])
        now = datetime.now(UTC)
        definition = SkillDefinition(
            name,
            "Journal identity",
            value["version"],
            now,
            now,
            source,
            "Journal identity",
            scope,
            self.root / scope.value / directory,
        )
        prepared = PreparedSkillWrite(
            action,
            definition,
            preimage,
            value["evolution_note"],
            operation_id,
            value.get("prune_identity"),
        )
        if not isinstance(prepared.evolution_note, str) or not prepared.evolution_note.strip():
            raise SkillStoreError("Skill journal evolution note is invalid")
        target = self._target(definition)
        skill_path = target / "SKILL.md"
        if action in {"add", "merge"}:
            if phase == "prepared" and (
                (action == "add" and not target.exists())
                or (
                    action == "merge"
                    and skill_path.is_file()
                    and _content_hash(skill_path) == preimage
                )
            ):
                if action == "add":
                    self._discard_unpublished_staging(prepared, target_hash)
                path.unlink()
                return
            verified = self._verify_definition(skill_path, prepared, target_hash)
            self._finish_operation(replace(prepared, definition=verified), target_hash)
            return
        pruned = self._pruned_target(prepared)
        if preimage != target_hash or (action == "prune" and source is not SkillSource.GENERATED):
            raise SkillStoreError("Skill move journal facts conflict")
        original, moved = (target, pruned) if action == "prune" else (pruned, target)
        if phase == "prepared" and original.exists() and not moved.exists():
            self._verify_definition(original / "SKILL.md", prepared, target_hash)
            path.unlink()
            return
        if original.exists() or not moved.exists():
            raise SkillStoreError("Skill move endpoints are ambiguous")
        self._verify_definition(moved / "SKILL.md", prepared, target_hash)
        self._finish_operation(prepared, target_hash)

    def recover_pending(self) -> RecoveryReport:
        root = self.state_root / "transactions"
        if not root.exists():
            return RecoveryReport()
        if any(
            part.exists() and _is_reparse(part)
            for part in (self.root.parent, self.root, self.state_root, root, root / "quarantine")
        ):
            return RecoveryReport(quarantined=1)
        recovered = quarantined = 0
        for path in root.glob("*.json"):
            try:
                self._recover_operation(path)
                recovered += 1
            except (OSError, KeyError, TypeError, ValueError):
                try:
                    quarantine = root / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        os.replace(path, quarantine / path.name)
                except OSError:
                    # A read-only auxiliary directory cannot disable current Skills.
                    pass
                quarantined += 1
        return RecoveryReport(recovered, quarantined)

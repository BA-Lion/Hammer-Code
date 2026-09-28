from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hammer_code.config import ContextConfig, SkillConfig, SkillEvolutionConfig
from hammer_code.skill import store as store_module
from hammer_code.skill.models import (
    AdjudicatedRetrieval,
    ApplicationEvaluation,
    RelevanceEvaluation,
    SkillContext,
    SkillDefinition,
    SkillObservation,
    SkillScope,
    SkillSource,
    UsageEntry,
)
from hammer_code.skill.repository import SkillRepository, SkillRepositoryError
from hammer_code.skill.service import SkillInvocationService
from hammer_code.skill.store import PreparedSkillWrite, SkillStore


class Crash(BaseException):
    """A simulated process exit, outside ordinary exception cleanup."""


def definition(root: Path, version: str = "0.1.0") -> SkillDefinition:
    stamp = datetime(2026, 8, 1, tzinfo=UTC)
    return SkillDefinition(
        "review",
        "Review a change",
        version,
        stamp,
        stamp,
        SkillSource.GENERATED,
        "Unique review body for {{arguments}}.",
        SkillScope.PROJECT,
        root / ".hammer-code" / "skill" / "project" / "review",
    )


def preimage(item: SkillDefinition) -> str:
    return hashlib.sha256((item.skill_dir / "SKILL.md").read_bytes()).hexdigest()


def seed_usage(store: SkillStore, item: SkillDefinition) -> UsageEntry:
    entry = UsageEntry(
        source=item.source,
        version=item.version,
        version_updated_at="2026-08-01T00:00:00Z",
        retrieve=11,
        loaded=7,
        relevant=5,
        applied=3,
        version_retrieve=4,
        version_loaded=3,
        version_relevant=2,
        version_applied=1,
        last_retrieved_at="2026-08-10T00:00:00Z",
        last_loaded_at="2026-08-11T00:00:00Z",
        last_relevant_at="2026-08-12T00:00:00Z",
        last_applied_at="2026-08-13T00:00:00Z",
        version_last_applied_at="2026-08-13T00:00:00Z",
        adjudicated_retrievals=(
            AdjudicatedRetrieval("old-retrieve", "old-turn", "2026-08-10T00:00:00Z", "old-eval"),
        ),
    )
    store._write_usage({"project:review": entry})
    return entry


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "merge"])
@pytest.mark.parametrize(
    "step",
    [
        "prepared",
        "file-replaced",
        "current_committed",
        "provenance-appended",
        "summary-planned",
        "summary-written",
        "provenance_committed",
        "record-appended",
        "record_committed",
        "usage-written",
    ],
)
async def test_write_crash_recovers_only_its_operation_and_preserves_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    step: str,
) -> None:
    store = SkillStore(tmp_path)
    old = definition(tmp_path)
    before = None
    if action == "merge":
        store.commit(PreparedSkillWrite("add", old, None, "initial", "initial-op"))
        before = seed_usage(store, old)
    item = (
        replace(old, version="0.1.1", updated_at=datetime(2026, 9, 1, tzinfo=UTC))
        if action == "merge"
        else old
    )
    operation = PreparedSkillWrite(
        action, item, preimage(old) if action == "merge" else None, "change", "repair-op"
    )

    with monkeypatch.context() as fault:
        if step in {"prepared", "current_committed", "provenance_committed", "record_committed"}:
            original = store._write_journal

            def journal(prepared, phase, target_hash):
                original(prepared, phase, target_hash)
                if phase == step:
                    raise Crash()

            fault.setattr(store, "_write_journal", journal)
        elif step == "file-replaced":
            original_journal = store._write_journal

            def before_journal(prepared, phase, target_hash):
                if phase == "current_committed":
                    raise Crash()
                original_journal(prepared, phase, target_hash)

            fault.setattr(store, "_write_journal", before_journal)
        elif step == "summary-planned":
            original_atomic = store_module.atomic_write_text

            def after_plan(path, text):
                original_atomic(path, text)
                if path == store._journal_path("repair-op") and "summary_content" in text:
                    raise Crash()

            fault.setattr(store_module, "atomic_write_text", after_plan)
        else:
            method = {
                "provenance-appended": "append_provenance",
                "summary-written": "_update_provenance_summary",
                "record-appended": "append_record",
                "usage-written": "sync_usage",
            }[step]
            original_method = getattr(store, method)

            def after_step(*args, **kwargs):
                original_method(*args, **kwargs)
                raise Crash()

            fault.setattr(store, method, after_step)
        with pytest.raises(Crash):
            store.commit(operation)

    repository = SkillRepository(tmp_path)
    # Startup must not replace journal-restored lifetime with a history replay.
    monkeypatch.setattr(
        repository.store, "_scan_records", lambda: pytest.fail("full record replay")
    )
    snapshot, report = await repository.initialize()
    assert report.recovered == 1 and report.quarantined == 0
    if step == "prepared":
        if action == "add":
            assert not snapshot.effective
        else:
            assert snapshot.effective["review"].version == old.version
            assert repository.store.load_usage()["project:review"] == before
        return
    assert snapshot.effective["review"].version == item.version
    entry = repository.store.load_usage()["project:review"]
    assert entry.version == item.version
    assert entry.version_updated_at == item.updated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert (entry.retrieve, entry.loaded, entry.relevant, entry.applied) == (
        (11, 7, 5, 3) if before else (0, 0, 0, 0)
    )
    assert (
        entry.version_retrieve,
        entry.version_loaded,
        entry.version_relevant,
        entry.version_applied,
    ) == (0, 0, 0, 0)
    assert entry.version_last_applied_at is None and entry.adjudicated_retrievals == ()
    if before:
        assert entry.last_applied_at == before.last_applied_at
        assert entry.last_retrieved_at == before.last_retrieved_at
        assert entry.last_loaded_at == before.last_loaded_at
        assert entry.last_relevant_at == before.last_relevant_at
    records = [
        json.loads(line) for line in store.record_path.read_text(encoding="utf-8").splitlines()
    ]
    provenance = [
        json.loads(line) for line in store.provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event.get("operation_id") == "repair-op" for event in records) == 1
    assert sum(event.get("operation_id") == "repair-op" for event in provenance) == 1
    summary = json.loads(store.provenance_summary_path.read_text(encoding="utf-8"))
    assert summary["total_evolution_count"] == (2 if action == "merge" else 1)
    await repository.record_loaded(item.ref, invocation_kind="direct")
    expected = repository.store.load_usage()["project:review"]
    _, again = await repository.initialize()
    assert again.recovered == again.quarantined == 0
    assert repository.store.load_usage()["project:review"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "operation_id",
        "target_hash",
        "name",
        "version",
        "source",
        "skill_dir",
        "usage-missing",
        "usage-corrupt",
    ],
)
@pytest.mark.parametrize("action", ["add", "merge"])
async def test_unprovable_journal_never_disables_current_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    action: str,
) -> None:
    store = SkillStore(tmp_path)
    item = definition(tmp_path)
    old_hash = None
    if action == "merge":
        store.commit(PreparedSkillWrite("add", item, None, "initial", "initial-op"))
        seed_usage(store, item)
        old_hash = preimage(item)
        item = replace(item, version="0.1.1")
    with monkeypatch.context() as fault:
        original = store.sync_usage

        def before_usage(*args):
            raise Crash()

        fault.setattr(store, "sync_usage", before_usage)
        with pytest.raises(Crash):
            store.commit(PreparedSkillWrite(action, item, old_hash, "test", "pending-op"))
        fault.setattr(store, "sync_usage", original)
    journal_path = store._journal_path("pending-op")
    value = json.loads(journal_path.read_text(encoding="utf-8"))
    if damage == "usage-corrupt":
        store.usage_path.write_text("invalid", encoding="utf-8")
    elif damage == "usage-missing":
        store.usage_path.unlink(missing_ok=True)
    else:
        value[damage] = {
            "operation_id": "other-op",
            "target_hash": "0" * 64,
            "name": "other",
            "version": "8.0.0",
            "source": "manual",
            "skill_dir": "../outside",
        }[damage]
        journal_path.write_text(json.dumps(value), encoding="utf-8")
    original_file = (item.skill_dir / "SKILL.md").read_bytes()
    repository = SkillRepository(tmp_path)
    snapshot, report = await repository.initialize()
    assert report.quarantined == (0 if damage == "usage-missing" else 1)
    service = SkillInvocationService(repository, SkillConfig(), ContextConfig())
    service.begin_turn(snapshot, SkillObservation(snapshot.generation))
    result = await service.invoke("review", "a change")
    assert not result.is_error and "Unique review body for a change" in result.content
    assert (item.skill_dir / "SKILL.md").read_bytes() == original_file
    if damage == "usage-missing":
        assert repository.store.load_usage()["project:review"].loaded == 1


@pytest.mark.parametrize("action", ["prune", "recover"])
@pytest.mark.parametrize(
    "phase", ["prepared", "move-before-journal", "current_committed", "record_committed"]
)
def test_move_journal_recovery_does_not_repeat_move_or_drop_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    phase: str,
) -> None:
    store = SkillStore(tmp_path)
    item = definition(tmp_path)
    store.commit(PreparedSkillWrite("add", item, None, "test", "initial-op"))
    before = seed_usage(store, item)
    identity = ""
    if action == "recover":
        store.commit_prune(item, preimage(item))
        identity = next((store.root / "prune" / "project" / "review").iterdir()).name
    with monkeypatch.context() as fault:
        original = store._write_journal

        def journal(prepared, stage, target_hash):
            if phase == "move-before-journal" and stage == "current_committed":
                raise Crash()
            original(prepared, stage, target_hash)
            if stage == phase:
                raise Crash()

        fault.setattr(store, "_write_journal", journal)
        with pytest.raises(Crash):
            if action == "prune":
                store.commit_prune(item, preimage(item))
            else:
                store.restore_pruned(item, identity)
    report = store.recover_pending()
    assert report.recovered == 1 and report.quarantined == 0
    assert store.load_usage()["project:review"] == before
    moved = phase != "prepared"
    assert item.skill_dir.exists() == (not moved if action == "prune" else moved)
    assert store.recover_pending().recovered == 0
    records = [
        json.loads(line) for line in store.record_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event_type"] == action for event in records) == int(moved)


@pytest.mark.asyncio
async def test_recovery_isolates_summary_when_later_operations_removed_its_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SkillStore(tmp_path)
    item = definition(tmp_path)
    with monkeypatch.context() as fault:
        original = store.sync_usage

        def crash(*args):
            original(*args)
            raise Crash()

        fault.setattr(store, "sync_usage", crash)
        with pytest.raises(Crash):
            store.commit(PreparedSkillWrite("add", item, None, "test", "pending-op"))
    for index in range(21):
        other = replace(
            item, name=f"other-{index}", skill_dir=item.skill_dir.parent / f"other-{index}"
        )
        store.commit(PreparedSkillWrite("add", other, None, "other operation", f"other-op-{index}"))
    before = store.provenance_summary_path.read_bytes()
    repository = SkillRepository(tmp_path)
    snapshot, report = await repository.initialize()
    assert report.quarantined == 1
    assert snapshot.effective["review"].ref == item.ref
    assert store.provenance_summary_path.read_bytes() == before
    assert json.loads(before)["total_evolution_count"] == 22


@pytest.mark.asyncio
async def test_recovery_preserves_a_direct_child_folder_distinct_from_skill_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SkillStore(tmp_path)
    item = definition(tmp_path)
    item = replace(item, skill_dir=item.skill_dir.parent / "custom folder")
    with monkeypatch.context() as fault:

        def crash(*args):
            raise Crash()

        fault.setattr(store, "sync_usage", crash)
        with pytest.raises(Crash):
            store.commit(PreparedSkillWrite("add", item, None, "test", "folder-op"))
    repository = SkillRepository(tmp_path)
    snapshot, report = await repository.initialize()
    assert report.recovered == 1 and report.quarantined == 0
    assert snapshot.effective["review"].skill_dir == item.skill_dir
    assert not (item.skill_dir.parent / "review").exists()


@pytest.mark.asyncio
async def test_unpublished_add_staging_is_not_discovered_after_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SkillStore(tmp_path)
    item = definition(tmp_path)
    with monkeypatch.context() as fault:
        original = store_module.os.replace

        def before_move(source, destination):
            if Path(destination) == item.skill_dir:
                raise Crash()
            original(source, destination)

        fault.setattr(store_module.os, "replace", before_move)
        with pytest.raises(Crash):
            store.commit(PreparedSkillWrite("add", item, None, "test", "stage-op"))
    snapshot, report = await SkillRepository(tmp_path).initialize()
    assert report.recovered == 1 and report.quarantined == 0
    assert snapshot.effective == {}
    assert not item.skill_dir.exists()
    assert list((store.state_root / "staging").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("new_loaded_first", [False, True])
async def test_old_evaluation_updates_lifetime_without_new_version_pollution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_loaded_first: bool,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    old = definition(tmp_path)
    await repository.commit(PreparedSkillWrite("add", old, None, "test", "old-op"))
    ids = await repository.record_retrieved((old.ref,))
    await repository.record_loaded(old.ref, invocation_kind="model")
    new = replace(old, version="0.1.1", updated_at=datetime(2026, 9, 1, tzinfo=UTC))
    await repository.commit(
        PreparedSkillWrite("merge", new, preimage(old), "new version", "new-op")
    )
    await repository.record_retrieved((old.ref,))
    await repository.record_loaded(old.ref, invocation_kind="model")
    if new_loaded_first:
        await repository.record_loaded(new.ref, invocation_kind="model")
    before = repository.store.load_usage()["project:review"]
    monkeypatch.setattr(store_module, "_utc_stamp", lambda: "2026-09-28T00:00:00Z")
    await repository.record_evaluation(
        (
            RelevanceEvaluation(
                scope=old.scope,
                name=old.name,
                version=old.version,
                relevant=True,
                reason="old input",
            ),
        ),
        (
            ApplicationEvaluation(
                scope=old.scope, name=old.name, version=old.version, applied=True, reason="old body"
            ),
        ),
        retrieve_event_ids=ids,
    )
    after = repository.store.load_usage()["project:review"]
    assert after.relevant == before.relevant + 1 and after.applied == before.applied + 1
    assert after.version_relevant == after.version_applied == 0
    assert after.version_retrieve == 0
    assert after.version_last_applied_at is None and after.adjudicated_retrievals == ()
    assert after.last_applied_at == "2026-09-28T00:00:00Z"
    assert after.version_loaded == before.version_loaded


@pytest.mark.asyncio
async def test_five_adjudications_prune_with_sixth_pending_and_account_after_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    item = definition(tmp_path)
    await repository.commit(PreparedSkillWrite("add", item, None, "test", "add-op"))
    config = SkillConfig(evolution=SkillEvolutionConfig(enabled=True, prune_unapplied_days=7))
    ids = []
    for day in range(1, 7):
        monkeypatch.setattr(
            store_module, "_utc_stamp", lambda day=day: f"2026-09-{day:02d}T00:00:00Z"
        )
        ids.append(await repository.record_retrieved((item.ref,)))
    for index in range(4):
        await repository.record_evaluation((), (), retrieve_event_ids=ids[index])
    await repository.record_evaluation((), (), retrieve_event_ids=ids[4], failure_stage="evaluator")
    assert await repository.prune_eligible(config, can_write=lambda: True) == ()
    await repository.record_evaluation((), (), retrieve_event_ids=ids[4])
    assert await repository.prune_eligible(config, can_write=lambda: True) == (item.ref,)
    assert not item.skill_dir.exists()
    await repository.record_evaluation(
        (),
        (
            ApplicationEvaluation(
                scope=item.scope,
                name=item.name,
                version=item.version,
                applied=True,
                reason="late use",
            ),
        ),
        retrieve_event_ids=ids[5],
    )
    after = repository.store.load_usage()["project:review"]
    assert after.retrieve == 6 and after.version_retrieve == 6
    assert after.applied == after.version_applied == 1
    assert after.version_last_applied_at == "2026-09-06T00:00:00Z"
    assert len(after.adjudicated_retrievals) == 5
    assert not item.skill_dir.exists()
    await repository.initialize()
    assert repository.store.load_usage()["project:review"] == after
    new = replace(item, version="0.1.1")
    await repository.commit(PreparedSkillWrite("add", new, None, "republish", "republish-op"))
    await repository.record_evaluation(
        (),
        (
            ApplicationEvaluation(
                scope=item.scope,
                name=item.name,
                version=item.version,
                applied=True,
                reason="older use",
            ),
        ),
    )
    newest = repository.store.load_usage()["project:review"]
    assert newest.applied == 2 and newest.version_applied == 0
    assert newest.version_last_applied_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["permission", "config", "window", "preimage", "version"])
async def test_prune_rechecks_before_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    item = definition(tmp_path)
    await repository.commit(PreparedSkillWrite("add", item, None, "test", "add-op"))
    window = tuple(
        AdjudicatedRetrieval(f"r{i}", None, f"2026-09-0{i + 1}T00:00:00Z", f"e{i}")
        for i in range(5)
    )
    entry = replace(repository.store.load_usage()["project:review"], adjudicated_retrievals=window)
    repository.store._write_usage({"project:review": entry})
    config = SkillConfig(evolution=SkillEvolutionConfig(enabled=True, prune_unapplied_days=7))
    writable = True
    original = repository.store._write_journal

    def journal(prepared, phase, target_hash):
        nonlocal writable
        original(prepared, phase, target_hash)
        if phase == "prepared":
            if changed == "permission":
                writable = False
            elif changed == "config":
                object.__setattr__(config.evolution, "enabled", False)
            elif changed == "window":
                repository.store._write_usage(
                    {"project:review": replace(entry, adjudicated_retrievals=())}
                )
            else:
                path = item.skill_dir / "SKILL.md"
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    text.replace("0.1.0", "0.1.1") if changed == "version" else text + "changed",
                    encoding="utf-8",
                )

    monkeypatch.setattr(repository.store, "_write_journal", journal)
    with pytest.raises(SkillRepositoryError):
        await repository.prune_eligible(config, can_write=lambda: writable)
    assert item.skill_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["model", "direct"])
@pytest.mark.parametrize("context", [SkillContext.INLINE, SkillContext.FORK])
async def test_cross_entry_aliases_keep_first_definition_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: str,
    context: SkillContext,
) -> None:
    repository = SkillRepository(tmp_path)
    await repository.initialize()
    item = replace(definition(tmp_path), context=context)
    await repository.commit(PreparedSkillWrite("add", item, None, "test", "add-op"))
    snapshot = await repository.snapshot_for_turn()
    service = SkillInvocationService(repository, SkillConfig(), ContextConfig())
    service.begin_turn(snapshot, SkillObservation(snapshot.generation))
    renders = 0
    import hammer_code.skill.service as service_module

    original_render = service_module.render_skill

    def render(*args):
        nonlocal renders
        renders += 1
        return original_render(*args)

    monkeypatch.setattr(service_module, "render_skill", render)
    if context is SkillContext.FORK:
        service._fork_runtime = object()  # type: ignore[assignment]

        async def fork(*args):
            from hammer_code.tools.base import ToolExecutionResult

            return ToolExecutionResult("fork done")

        monkeypatch.setattr(service, "_invoke_fork", fork)
    if first == "model":
        assert not (await service.invoke("review", "original")).is_error
    else:
        await service.invoke_user("project:review", "original")
    new = replace(item, version="0.1.1", context=SkillContext.INLINE, prompt_template="New body.")
    await repository.commit(PreparedSkillWrite("merge", new, preimage(item), "publish", "merge-op"))
    assert "@0.1.0" in (await service.invoke("review", "changed")).content
    repeated = await service.invoke_user("project:review", "changed")
    assert repeated.already_loaded and repeated.definition.ref == item.ref
    assert repeated.is_fork == (context is SkillContext.FORK)
    assert "@0.1.0" in repeated.prompt and "New body" not in repeated.prompt

    async def fail_lookup(*args, **kwargs):
        pytest.fail("duplicate alias re-read catalog")

    monkeypatch.setattr(repository, "resolve_latest", fail_lookup)
    assert not (await service.invoke("review", "third")).is_error
    assert (await service.invoke_user("project:review", "third")).already_loaded
    evidence = service.finish_turn_evidence(completed=True, live_messages=())
    assert evidence is not None and len(evidence.loaded) == 1
    assert evidence.loaded[0].ref == item.ref and "original" in evidence.loaded[0].body
    assert renders == 1
    records = [
        json.loads(line)
        for line in repository.store.record_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event_type"] == "loaded" for event in records) == 1

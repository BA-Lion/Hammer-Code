from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hammer_code.config import SkillVectorConfig
from hammer_code.skill.embedding import EmbeddingQuery
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.vector_runtime import SkillVectorRuntime


class Provider:
    dimensions = 384

    def __init__(self):
        self.documents = []
        self.queries = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.started = asyncio.Event()
        self.active = 0
        self.maximum = 0
        self.closed = False

    async def embed_query(self, query):
        self.queries.append(query)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.started.set()
        try:
            await self.gate.wait()
            return (1.0,) + (0.0,) * 383
        finally:
            self.active -= 1

    async def embed_documents(self, documents):
        self.documents.extend(documents)
        return tuple((1.0,) + (0.0,) * 383 for _ in documents)

    async def aclose(self):
        self.closed = True


async def test_initial_refresh_query_and_change_without_version(
    tmp_path: Path, skill_factory
) -> None:
    pytest.importorskip("numpy")
    definition = skill_factory("review")
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    provider = Provider()
    runtime = SkillVectorRuntime(repo, SkillVectorConfig(), lambda _: None, provider=provider)
    runtime.start()
    try:
        await asyncio.wait_for(runtime.wait_idle(), 3)
        assert provider.documents
        hits = await repo.search.search(
            EmbeddingQuery("unrelated"), 3, snapshot=await repo.snapshot_for_turn()
        )
        assert hits[0].ref == definition.ref
        path = definition.skill_dir / "SKILL.md"
        path.write_text(
            path.read_text("utf-8").replace("workflow review", "Changed metadata"), encoding="utf-8"
        )
        await repo.snapshot_for_turn()
        await asyncio.wait_for(runtime.wait_idle(), 3)
        assert provider.documents[-1].description == "Changed metadata"
        previous = len(provider.documents)
        runtime.request_refresh()
        await asyncio.wait_for(runtime.wait_idle(), 3)
        assert len(provider.documents) > previous
        assert hits[0].ref.version == "0.1.0"
    finally:
        await runtime.aclose()
    assert provider.closed and runtime.state == "closed"


async def test_periodic_1800_seconds_without_real_wait(tmp_path: Path, skill_factory) -> None:
    skill_factory("review")
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    tick = asyncio.Event()
    entered = asyncio.Event()
    durations = []

    async def sleep(seconds):
        durations.append(seconds)
        entered.set()
        await tick.wait()
        tick.clear()

    provider = Provider()
    runtime = SkillVectorRuntime(
        repo, SkillVectorConfig(), lambda _: None, provider=provider, sleep=sleep
    )
    runtime.start()
    try:
        await runtime.wait_idle()
        await entered.wait()
        assert durations == [1800]
        initial = len(provider.documents)
        tick.set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(provider.documents) > initial:
                break
        await runtime.wait_idle()
        assert len(provider.documents) > initial
    finally:
        await runtime.aclose()


async def test_timeout_expired_jobs_and_close_wait_for_inflight(
    tmp_path: Path, skill_factory
) -> None:
    definition = skill_factory("review")
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    provider = Provider()
    runtime = SkillVectorRuntime(
        repo,
        SkillVectorConfig(query_timeout_seconds=0.03),
        lambda _: None,
        provider=provider,
        capacity=1,
    )
    runtime.start()
    await runtime.wait_idle()
    provider.gate.clear()
    identities = frozenset({(definition.scope, definition.name)})
    first = asyncio.create_task(runtime.rank(EmbeddingQuery("first"), identities, 6))
    await provider.started.wait()
    assert await first == ()
    assert provider.active == 1
    second = asyncio.create_task(runtime.rank(EmbeddingQuery("expired"), identities, 6))
    await asyncio.sleep(0)
    assert await runtime.rank(EmbeddingQuery("queue full"), identities, 6) == ()
    assert await second == ()
    closing = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0.01)
    assert not closing.done() and not provider.closed
    provider.gate.set()
    await asyncio.wait_for(closing, 3)
    assert provider.maximum == 1
    assert [item.text for item in provider.queries] == ["first"]


async def test_query_priority_between_refresh_jobs(tmp_path: Path) -> None:
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    runtime = SkillVectorRuntime(repo, SkillVectorConfig(), lambda _: None, provider=Provider())
    runtime.start()
    try:
        await runtime.wait_idle()
        gate = asyncio.Event()
        started = asyncio.Event()
        order = []

        async def busy():
            started.set()
            await gate.wait()

        async def refresh():
            order.append("refresh")

        async def query():
            order.append("query")

        current = runtime._enqueue(busy, 1)
        await started.wait()
        background = runtime._enqueue(refresh, 1)
        urgent = runtime._enqueue(query, 0)
        gate.set()
        await asyncio.gather(current, background, urgent)
        assert order == ["query", "refresh"]
    finally:
        await runtime.aclose()


async def test_disabled_and_missing_models_are_nonblocking(tmp_path: Path) -> None:
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    warnings = []
    runtime = SkillVectorRuntime(repo, SkillVectorConfig(), warnings.append)
    runtime.start()
    try:
        assert not await runtime.wait_ready()
        assert warnings and not (tmp_path / ".hammer-code/skill/state/embeddings.sqlite3").exists()
        assert await runtime.rank(EmbeddingQuery("x"), frozenset(), 6) == ()
    finally:
        await runtime.aclose()
    disabled = SkillVectorRuntime(repo, SkillVectorConfig(enabled=False), warnings.append)
    disabled.start()
    assert disabled._periodic_task is None
    await disabled.aclose()


async def test_cancelled_close_still_waits_for_initialization(tmp_path: Path, monkeypatch) -> None:
    from hammer_code.skill import local_embedding

    repo = SkillRepository(tmp_path)
    await repo.initialize()
    entered, release = asyncio.Event(), asyncio.Event()
    instances = []

    class Loading(Provider):
        def __init__(self, root, executor):
            super().__init__()
            instances.append(self)

        async def initialize(self):
            entered.set()
            await release.wait()

    monkeypatch.setattr(local_embedding, "LocalEmbeddingProvider", Loading)
    runtime = SkillVectorRuntime(repo, SkillVectorConfig(), lambda _: None)
    runtime.start()
    await entered.wait()
    closing = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0.01)
    assert not closing.done() and not instances[0].closed
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert instances[0].closed and runtime.state == "closed"
    assert runtime._worker_task is None


async def test_cancelled_query_keeps_actual_capacity(tmp_path: Path) -> None:
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    provider = Provider()
    runtime = SkillVectorRuntime(repo, SkillVectorConfig(), lambda _: None, provider=provider)
    runtime.start()
    await runtime.wait_idle()
    try:
        provider.gate.clear()
        query = asyncio.create_task(runtime.rank(EmbeddingQuery("cancelled"), frozenset(), 6))
        await provider.started.wait()
        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert provider.active == 1
        provider.gate.set()
        await runtime._queue.join()
        assert provider.active == 0
    finally:
        await runtime.aclose()

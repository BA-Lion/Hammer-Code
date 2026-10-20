"""One process-owned embedding executor, bounded priority queue and refresh loop."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from hammer_code.config import SkillVectorConfig
from hammer_code.skill.embedding import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingQuery,
    Identity,
    SkillMetadata,
)
from hammer_code.skill.models import CatalogSnapshot
from hammer_code.skill.search import SkillSearchService
from hammer_code.skill.vector_cache import VectorCache

if TYPE_CHECKING:
    from hammer_code.skill.repository import SkillRepository


class SkillVectorRuntime:
    def __init__(
        self,
        repository: SkillRepository,
        config: SkillVectorConfig,
        warning: Callable[[str], None],
        *,
        provider: EmbeddingProvider | None = None,
        capacity: int = 16,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.repository = repository
        self.config = config
        self.warning = warning
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skill-vector")
        self.provider = provider
        self.cache = VectorCache(repository.catalog.root.parent.parent)
        self.search = SkillSearchService(self)
        self._queue: asyncio.PriorityQueue[tuple[int, int, Any]] = asyncio.PriorityQueue(capacity)
        self._sequence = itertools.count()
        self._refresh_event = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._startup: asyncio.Task[None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._periodic_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._sleep = sleep
        self._closing = False
        self.ready = False
        self.state = "disabled" if not config.enabled else "starting"
        self._last_warning: str | None = None

    def start(self) -> None:
        if self._closing:
            raise EmbeddingError("runtime_closed")
        self.repository.bind_search(self.search, self._changed if self.config.enabled else None)
        if not self.config.enabled or self._startup is not None:
            return
        self._startup = asyncio.create_task(self._initialize())
        self._periodic_task = asyncio.create_task(self._periodic())

    async def wait_ready(self) -> bool:
        if self._startup is not None:
            await asyncio.shield(self._startup)
        return self.ready

    def _warn(self, kind: str) -> None:
        self.state = kind
        if kind != self._last_warning:
            self._last_warning = kind
            try:
                self.warning(
                    f"Skill vector search unavailable ({kind}); using BM25. "
                    "Run --prepare-skill-model if assets are missing."
                )
            except Exception:
                pass

    async def _thread(self, function: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    async def _initialize(self) -> None:
        if self._closing:
            return
        try:
            if self.provider is None:
                from hammer_code.skill.local_embedding import LocalEmbeddingProvider

                local = LocalEmbeddingProvider(self.cache.root, self.executor)
                self.provider = local
                await local.initialize()
            await self._thread(self.cache.open)
            if self._closing:
                return
            self.ready = True
            self.state = "ready"
            self._worker_task = asyncio.create_task(self._worker())
            self._refresh_task = asyncio.create_task(self._refresh())
            self.request_refresh()
        except Exception as exc:
            with suppress(Exception):
                await self._thread(self.cache.close)
            if self.provider is not None:
                with suppress(Exception):
                    await self.provider.aclose()
                self.provider = None
            safe_kinds = {
                "missing_assets",
                "invalid_assets",
                "missing_dependencies",
                "cache_unavailable",
                "incompatible_model_inputs",
                "incompatible_model_output",
                "local_provider_unavailable",
            }
            kind = (
                str(exc)
                if isinstance(exc, EmbeddingError) and str(exc) in safe_kinds
                else "initialization_failed"
            )
            self._warn(kind)

    def _changed(self, snapshot: CatalogSnapshot) -> None:
        self.request_refresh()

    def request_refresh(self) -> None:
        if not self._closing and self.config.enabled:
            self._idle.clear()
            self._refresh_event.set()

    async def wait_idle(self) -> None:
        await self.wait_ready()
        if self.ready:
            await self._idle.wait()

    async def _periodic(self) -> None:
        while not self._closing:
            await self._sleep(self.config.refresh_interval_seconds)
            if self._closing:
                return
            if not self.ready:
                await self.wait_ready()
                if not self.ready:
                    await self._initialize()
            self.request_refresh()
            if self.ready:
                await self._idle.wait()

    def _enqueue(
        self, operation: Callable[[], Awaitable[Any]], priority: int, expires: float | None = None
    ) -> asyncio.Future[Any]:
        if self._closing:
            raise EmbeddingError("runtime_closed")
        future = asyncio.get_running_loop().create_future()
        # Consume late failures after callers have timed out or been cancelled.
        future.add_done_callback(lambda item: None if item.cancelled() else item.exception())
        try:
            self._queue.put_nowait((priority, next(self._sequence), (operation, future, expires)))
        except asyncio.QueueFull as exc:
            future.cancel()
            raise EmbeddingError("queue_full") from exc
        return future

    async def _worker(self) -> None:
        while True:
            _, _, job = await self._queue.get()
            try:
                if job is None:
                    return
                operation, future, expires = job
                if expires is not None and asyncio.get_running_loop().time() >= expires:
                    future.set_exception(EmbeddingError("query_expired"))
                    continue
                try:
                    result = await operation()
                except Exception as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._queue.task_done()

    async def rank(
        self, query: EmbeddingQuery, identities: frozenset[Identity], limit: int
    ) -> tuple[tuple[Identity, float], ...]:
        if not self.ready or self._closing:
            return ()

        async def operation() -> tuple[tuple[Identity, float], ...]:
            assert self.provider is not None
            vector = await self.provider.embed_query(query)
            return await self._thread(self.cache.rank, vector, identities, limit)

        try:
            expires = asyncio.get_running_loop().time() + self.config.query_timeout_seconds
            future = self._enqueue(operation, 0, expires)
            result = await asyncio.wait_for(
                asyncio.shield(future), self.config.query_timeout_seconds
            )
            self._last_warning = None
            self.state = "ready"
            return result
        except (TimeoutError, EmbeddingError):
            self._warn("query_timeout_or_busy")
            return ()
        except Exception:
            self._warn("query_failed")
            return ()

    async def _refresh(self) -> None:
        while not self._closing:
            await self._refresh_event.wait()
            self._refresh_event.clear()
            try:
                snapshot = await self.repository.snapshot_for_turn()
                for identity, definition in sorted(snapshot.by_identity.items()):
                    if self._closing:
                        break

                    metadata = SkillMetadata.from_definition(definition)

                    async def operation(
                        identity: Identity = identity,
                        metadata: SkillMetadata = metadata,
                    ) -> None:
                        assert self.provider is not None
                        vectors = await self.provider.embed_documents((metadata,))
                        if len(vectors) != 1:
                            raise EmbeddingError("invalid_embedding_batch")
                        await self._thread(self.cache.upsert, identity, vectors[0])

                    while not self._closing:
                        try:
                            future = self._enqueue(operation, 1)
                            await asyncio.shield(future)
                            break
                        except EmbeddingError as exc:
                            if str(exc) == "queue_full":
                                await self._sleep(0.01)
                                continue
                            self._warn("refresh_failed")
                            break
                        except Exception:
                            self._warn("refresh_failed")
                            break
            except Exception:
                self._warn("refresh_failed")
            finally:
                if not self._refresh_event.is_set():
                    self._idle.set()

    async def rebuild_cache(self) -> None:
        if self._closing:
            raise EmbeddingError("runtime_closed")
        if self.ready:

            async def operation() -> None:
                await self._thread(self.cache.rebuild)

            await asyncio.shield(self._enqueue(operation, 0))
        else:
            await self._thread(self.cache.rebuild)
            await self._initialize()
        self.request_refresh()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def _close(self) -> None:
        self._closing = True
        if self._startup is not None:
            await asyncio.shield(self._startup)
        for task in (self._periodic_task, self._refresh_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._periodic_task, self._refresh_task) if task is not None),
            return_exceptions=True,
        )
        while not self._queue.empty():
            _, _, job = self._queue.get_nowait()
            if job is not None:
                job[1].set_exception(EmbeddingError("runtime_closed"))
            self._queue.task_done()
        if self._worker_task is not None:
            self._queue.put_nowait((2, next(self._sequence), None))
            await self._worker_task
        try:
            await self._thread(self.cache.close)
            if self.provider is not None:
                await self.provider.aclose()
        finally:
            await asyncio.to_thread(self.executor.shutdown, wait=True)
            self.ready = False
            self.state = "closed"

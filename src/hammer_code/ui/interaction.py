"""One FIFO owner for terminal reads shared by primary and background operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class InteractionCoordinator:
    """Serialize async UI reads; cancellation never permits a second concurrent reader."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def read(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            task = asyncio.ensure_future(operation())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise

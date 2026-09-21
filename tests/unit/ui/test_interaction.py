from __future__ import annotations

import asyncio

import pytest

from hammer_code.ui.interaction import InteractionCoordinator


@pytest.mark.asyncio
async def test_interaction_coordinator_serializes_readers() -> None:
    coordinator = InteractionCoordinator()
    active = maximum = 0
    release = asyncio.Event()

    async def reader() -> str:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await release.wait()
        active -= 1
        return "ok"

    first = asyncio.create_task(coordinator.read(reader))
    await asyncio.sleep(0)
    second = asyncio.create_task(coordinator.read(reader))
    await asyncio.sleep(0)
    assert maximum == 1
    release.set()
    assert await first == "ok"
    assert await second == "ok"
    assert maximum == 1


@pytest.mark.asyncio
async def test_waiting_reader_cancellation_does_not_block_the_owner() -> None:
    coordinator = InteractionCoordinator()
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "done"

    owner = asyncio.create_task(coordinator.read(slow))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(coordinator.read(slow))
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    release.set()
    assert await owner == "done"


@pytest.mark.asyncio
async def test_active_reader_cancellation_waits_for_its_input_to_finish() -> None:
    coordinator = InteractionCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow() -> str:
        entered.set()
        await release.wait()
        return "done"

    owner = asyncio.create_task(coordinator.read(slow))
    await entered.wait()
    owner.cancel()
    await asyncio.sleep(0)
    assert not owner.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner

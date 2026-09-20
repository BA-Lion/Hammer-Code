import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from hammer_code.app.agent import PrimaryAgent
from hammer_code.app.chat_loop import ChatLoop
from hammer_code.app.commands import CommandRegistry, register_builtin_commands
from hammer_code.app.runtime import AgentFactory
from hammer_code.errors import HammerCodeError
from hammer_code.memory.store import MemoryStore
from hammer_code.session.manager import SessionManager
from hammer_code.ui.console import ConsolePort


class _UI:
    def __init__(self, *inputs: str) -> None:
        self.inputs: asyncio.Queue[str] = asyncio.Queue()
        for value in inputs:
            self.inputs.put_nowait(value)
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    async def prompt(self) -> str:
        return await self.inputs.get()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def memory_warning(self, message: str) -> None:
        self.warnings.append(message)

    def skill_warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def __getattr__(self, _: str):
        def ignored(*_args: object, **_kwargs: object) -> None:
            return None

        return ignored


class _Agent:
    def __init__(
        self, session_id: str, *, block_drain: bool = False, fail_drain: bool = False
    ) -> None:
        self.session_id = session_id
        self.turns: list[str] = []
        self.drain_started = asyncio.Event()
        self.release_drain = asyncio.Event()
        self.block_drain = block_drain
        self.fail_drain = fail_drain
        self.cancelled = 0

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(protocol="protocol")

    async def run_turn(self, text: str) -> None:
        self.turns.append(text)

    async def drain(self) -> None:
        self.drain_started.set()
        if self.fail_drain:
            raise RuntimeError("drain failed")
        if self.block_drain:
            await self.release_drain.wait()

    async def cancel(self) -> None:
        self.cancelled += 1
        self.release_drain.set()


class _Factory:
    def __init__(
        self, replacement: _Agent | None = None, *, resume_error: Exception | None = None
    ) -> None:
        self.replacement = replacement
        self.resume_error = resume_error
        self.create_calls = 0
        self.resume_calls: list[str] = []

    async def create_new(self) -> PrimaryAgent:
        self.create_calls += 1
        assert self.replacement is not None
        return cast(PrimaryAgent, self.replacement)

    async def resume(self, session_id: str) -> PrimaryAgent:
        self.resume_calls.append(session_id)
        if self.resume_error is not None:
            raise self.resume_error
        assert self.replacement is not None
        return cast(PrimaryAgent, self.replacement)


def _loop(current: _Agent, factory: _Factory, ui: _UI) -> ChatLoop:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    return ChatLoop(
        cast(PrimaryAgent, current),
        cast(AgentFactory, factory),
        registry,
        cast(SessionManager, object()),
        cast(MemoryStore, object()),
        cast(ConsolePort, ui),
    )


@pytest.mark.asyncio
async def test_switches_immediately_while_previous_agent_drains() -> None:
    old = _Agent("old", block_drain=True)
    replacement = _Agent("new")
    ui = _UI("/session new")
    loop = _loop(old, _Factory(replacement), ui)

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(old.drain_started.wait(), timeout=1)
    assert loop.current is replacement

    await ui.inputs.put("new session input")
    for _ in range(20):
        if replacement.turns:
            break
        await asyncio.sleep(0)
    assert replacement.turns == ["new session input"]

    old.release_drain.set()
    await ui.inputs.put("/exit")
    await running


@pytest.mark.asyncio
async def test_resume_failure_keeps_the_existing_agent_current() -> None:
    current = _Agent("current")
    ui = _UI()
    loop = _loop(current, _Factory(resume_error=HammerCodeError("cannot restore")), ui)

    await loop._resume_and_switch("missing")

    assert loop.current is current
    assert not loop.is_session_draining(current.session_id)
    assert ui.errors == ["cannot restore"]


@pytest.mark.asyncio
async def test_exit_drains_existing_work_without_force_cancellation() -> None:
    current = _Agent("current", block_drain=True)
    ui = _UI("/exit")
    loop = _loop(current, _Factory(), ui)

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(current.drain_started.wait(), timeout=1)
    assert current.cancelled == 0
    current.release_drain.set()
    await running
    assert current.cancelled == 0
    assert ui.infos == ["Draining session maintenance before exit."]


@pytest.mark.asyncio
async def test_shutdown_reports_a_drain_failure_without_abandoning_shutdown() -> None:
    current = _Agent("current", fail_drain=True)
    ui = _UI()
    loop = _loop(current, _Factory(), ui)

    await loop._graceful_shutdown()

    assert ui.warnings == ["Session current could not finish draining."]


@pytest.mark.asyncio
async def test_force_shutdown_cancels_drain_then_converges_agent_cleanup() -> None:
    current = _Agent("current", block_drain=True)
    ui = _UI()
    loop = _loop(current, _Factory(), ui)
    task = asyncio.create_task(current.drain())
    loop._draining[current.session_id] = (cast(PrimaryAgent, current), task)
    await asyncio.wait_for(current.drain_started.wait(), timeout=1)

    await loop._force_shutdown()

    assert task.cancelled()
    assert current.cancelled == 1

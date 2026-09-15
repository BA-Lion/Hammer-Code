"""Outer input loop, command dispatch, session switching, and draining ownership."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

from hammer_code.app.agent import PrimaryAgent
from hammer_code.app.commands import (
    CommandAction,
    CommandContext,
    CommandHost,
    CommandRegistry,
    dispatch,
    parse_command,
)
from hammer_code.app.runtime import AgentFactory
from hammer_code.errors import HammerCodeError
from hammer_code.memory.models import MemoryCategory, validate_path
from hammer_code.memory.store import MemoryStore
from hammer_code.session.manager import SessionManager
from hammer_code.ui.console import ConsolePort


class ChatLoop(CommandHost):
    def __init__(
        self,
        current: PrimaryAgent,
        factory: AgentFactory,
        registry: CommandRegistry,
        sessions: SessionManager,
        memory_store: MemoryStore,
        ui: ConsolePort,
    ) -> None:
        self.current, self.factory, self.registry = current, factory, registry
        self.sessions, self.memory_store, self.ui = sessions, memory_store, ui
        self._draining: dict[str, tuple[PrimaryAgent, asyncio.Task[None]]] = {}

    async def run(self) -> None:
        try:
            while True:
                await self._reap_draining()
                try:
                    text = await self.ui.prompt()
                except (EOFError, KeyboardInterrupt):
                    break
                if not text.strip():
                    continue
                try:
                    invocation = parse_command(text)
                except HammerCodeError as exc:
                    self.ui.error(str(exc))
                    continue
                if invocation is None:
                    task = asyncio.create_task(self.current.run_turn(text))
                    try:
                        await task
                    except KeyboardInterrupt:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        self.ui.error("Request cancelled.")
                    continue
                outcome = await dispatch(
                    self.registry, CommandContext(self.current, self, self.ui), invocation
                )
                if outcome.action is CommandAction.EXIT:
                    break
                if outcome.action is CommandAction.NEW_SESSION:
                    await self._create_and_switch()
                elif outcome.action is CommandAction.RESUME_SESSION and outcome.session_id:
                    await self._resume_and_switch(outcome.session_id)
        finally:
            await self._graceful_shutdown()

    async def list_sessions(self) -> tuple[object, ...]:
        items = await asyncio.to_thread(self.sessions.list, self.current.status().protocol)
        return tuple(
            SimpleNamespace(
                id=item.id,
                title=("[current] " if item.id == self.current.session_id else "")
                + ("[draining] " if item.id in self._draining else "")
                + item.title,
                last_active=item.last_active,
            )
            for item in items
        )

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self.sessions.delete, session_id)

    def is_session_draining(self, session_id: str) -> bool:
        return session_id in self._draining

    async def list_memory(self, category: str | None = None) -> tuple[object, ...]:
        catalog = await asyncio.to_thread(self.memory_store.load_catalog)
        categories = (MemoryCategory(category),) if category else tuple(MemoryCategory)
        return tuple(
            f"{kind.value}: indexed={entry.path} ({entry.description})"
            for kind in categories
            for entry in catalog.categories[kind].index.entries
        ) + tuple(
            f"{kind.value}: orphan={path}"
            for kind in categories
            for path in catalog.categories[kind].orphans
        )

    async def read_memory(self, category: str, relative_path: str) -> str:
        kind, path = MemoryCategory(category), validate_path(relative_path)
        catalog = await asyncio.to_thread(self.memory_store.load_catalog)
        try:
            return catalog.categories[kind].topics[path]
        except KeyError as exc:
            raise HammerCodeError("Memory topic does not exist") from exc

    async def _create_and_switch(self) -> None:
        try:
            replacement = await self.factory.create_new()
        except HammerCodeError as exc:
            self.ui.error(str(exc))
            return
        except Exception:
            self.ui.error("Unable to create a new session.")
            return
        self._switch_to(replacement)

    async def _resume_and_switch(self, session_id: str) -> None:
        if session_id == self.current.session_id:
            self.ui.info("Session is already current.")
            return
        if self.is_session_draining(session_id):
            self.ui.error("Session is still draining; try again later.")
            return
        try:
            replacement = await self.factory.resume(session_id)
        except HammerCodeError as exc:
            self.ui.error(str(exc))
            return
        except Exception:
            self.ui.error("Unable to resume that session.")
            return
        self._switch_to(replacement)

    def _switch_to(self, replacement: PrimaryAgent) -> None:
        previous = self.current
        self.current = replacement
        self._draining[previous.session_id] = (previous, asyncio.create_task(previous.drain()))

    async def _reap_draining(self) -> None:
        for session_id, (_, task) in tuple(self._draining.items()):
            if task.done():
                self._draining.pop(session_id)
                try:
                    await task
                except BaseException:
                    self.ui.memory_warning(f"Session {session_id} could not finish draining.")

    async def _graceful_shutdown(self) -> None:
        current = self.current
        if current.session_id not in self._draining:
            self._draining[current.session_id] = (current, asyncio.create_task(current.drain()))
        self.ui.info("Draining session maintenance before exit.")
        try:
            results = await asyncio.gather(
                *(task for _, task in self._draining.values()), return_exceptions=True
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            await self._force_shutdown()
            return
        for (session_id, _), result in zip(tuple(self._draining.items()), results, strict=True):
            if isinstance(result, BaseException):
                self.ui.memory_warning(f"Session {session_id} could not finish draining.")

    async def _force_shutdown(self) -> None:
        """Cancel owned drain waits, then make every Agent converge its own cleanup."""

        draining = tuple(self._draining.items())
        for _, (_, task) in draining:
            if not task.done():
                task.cancel()
        if draining:
            await asyncio.gather(*(task for _, (_, task) in draining), return_exceptions=True)
        for session_id, (agent, _) in draining:
            try:
                await agent.cancel()
            except BaseException:
                self.ui.memory_warning(f"Session {session_id} could not finish cancellation.")

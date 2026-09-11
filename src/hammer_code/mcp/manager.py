"""Background, event-driven MCP connection management and prompt caching."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from hammer_code.config import McpConfig
from hammer_code.mcp.client import McpClient
from hammer_code.mcp.types import McpClientStatus, McpManagerStatus
from hammer_code.tools.registry import ToolRegistry

ClientBuilder = Callable[[McpConfig, Path], McpClient]


class McpManager:
    def __init__(
        self,
        configs: tuple[McpConfig, ...],
        registry: ToolRegistry,
        workspace_root: Path,
        *,
        client_builder: ClientBuilder = McpClient,
        ui: object | None = None,
    ) -> None:
        self.registry = registry
        self.workspace_root = workspace_root.resolve()
        self._client_builder = client_builder
        self._ui = ui
        self._lifecycle_lock = asyncio.Lock()
        self._load_task: asyncio.Task[None] | None = None
        self._started = False
        self.configs: tuple[McpConfig, ...] = ()
        self.clients: dict[str, McpClient] = {}
        self.statuses: dict[str, McpClientStatus] = {}
        self.status = McpManagerStatus.SETTLED
        self.pending_count = 0
        self.prompt = ""
        self._apply_configs(configs)

    def start(self) -> asyncio.Task[None] | None:
        if self._started:
            return self._load_task
        self._started = True
        if not self.clients:
            return None
        self._load_task = asyncio.create_task(self._load_clients(), name="mcp-background-load")
        return self._load_task

    async def reconnect(self, name: str) -> None:
        async with self._lifecycle_lock:
            if self._load_task is not None and not self._load_task.done():
                raise RuntimeError("MCP initial load is still running")
            client = self.clients[name]
            self.registry.remove_owner(name)
            self.status = McpManagerStatus.CONNECTING
            self.statuses[name] = McpClientStatus.CONNECTING
            self.pending_count = 1
            self._rebuild_prompt()
            await self._connect_one(name, client, reconnect=True)

    async def reload(self, configs: tuple[McpConfig, ...]) -> None:
        async with self._lifecycle_lock:
            await self._cancel_load()
            for name in tuple(self.clients):
                self.registry.remove_owner(name)
            await self._close_clients()
            self._started = True
            self._apply_configs(configs)
            if self.clients:
                self._load_task = asyncio.create_task(
                    self._load_clients(), name="mcp-background-reload"
                )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._cancel_load()
            await self._close_clients()

    async def _load_clients(self) -> None:
        for name, client in tuple(self.clients.items()):
            await self._connect_one(name, client)

    async def _connect_one(self, name: str, client: McpClient, *, reconnect: bool = False) -> None:
        try:
            async with asyncio.timeout(client.config.connect_timeout_seconds):
                wrappers = await (client.reconnect() if reconnect else client.connect())
            self.registry.replace_owner(name, wrappers)
        except asyncio.CancelledError:
            self.registry.remove_owner(name)
            raise
        except Exception as exc:
            self.registry.remove_owner(name)
            with suppress(Exception):
                await client.close()
            self._complete(name, McpClientStatus.FAILED, type(exc).__name__)
        else:
            self._complete(name, McpClientStatus.CONNECTED, None)

    def _complete(self, name: str, state: McpClientStatus, detail: str | None) -> None:
        self.statuses[name] = state
        self.pending_count -= 1
        if self.pending_count <= 0:
            self.pending_count = 0
            self.status = McpManagerStatus.SETTLED
        self._rebuild_prompt()
        callback = getattr(self._ui, "mcp_status", None)
        if callable(callback):
            callback(name, state.value, detail)

    async def _cancel_load(self) -> None:
        task, self._load_task = self._load_task, None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _close_clients(self) -> None:
        for client in reversed(tuple(self.clients.values())):
            with suppress(Exception):
                await client.close()

    def _apply_configs(self, configs: tuple[McpConfig, ...]) -> None:
        final: dict[str, McpConfig] = {}
        for config in configs:
            final.pop(config.name, None)
            final[config.name] = config
        self.configs = tuple(final.values())
        self.clients = {
            config.name: self._client_builder(config, self.workspace_root)
            for config in self.configs
        }
        self.statuses = {name: McpClientStatus.CONNECTING for name in self.clients}
        self.pending_count = len(self.clients)
        self.status = McpManagerStatus.CONNECTING if self.clients else McpManagerStatus.SETTLED
        self._rebuild_prompt()

    def _rebuild_prompt(self) -> None:
        if not self.configs:
            self.prompt = ""
            return
        lines = ["# MCP", "", f"Overall status: {self.status.value}", ""]
        lines.extend(
            f"- {config.name} [{self.statuses[config.name].value}]: {config.description}"
            for config in self.configs
        )
        self.prompt = "\n".join(lines) + "\n"

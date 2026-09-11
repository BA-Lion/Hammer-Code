"""One MCP SDK client and its resources, contained inside the adapter boundary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import (
    AudioContent,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from hammer_code.config import McpConfig, McpHttpConfig, McpStdioConfig
from hammer_code.errors import McpConfigurationError
from hammer_code.mcp.types import McpCallResult, McpClientStatus
from hammer_code.mcp.wrapper import McpToolWrapper

ClientFactory = Callable[[Any], Any]


class McpClient:
    """Own exactly one SDK Client/session/exit stack for one configured server."""

    def __init__(
        self,
        config: McpConfig,
        workspace_root: Path,
        client_factory: ClientFactory = Client,
    ) -> None:
        self.config = config
        self.name = config.name
        self._workspace_root = workspace_root.resolve()
        self._client_factory = client_factory
        self.sdk_client: Any | None = None
        self.clientsession: Any | None = None
        self.stack: AsyncExitStack | None = None
        self.status = McpClientStatus.CONNECTING
        self._lock = asyncio.Lock()
        self._wrappers: tuple[McpToolWrapper, ...] = ()
        self._raw_tool_names: dict[str, str] = {}

    async def connect(self) -> tuple[McpToolWrapper, ...]:
        async with self._lock:
            return await self._connect_unlocked()

    async def stdio_connect(self) -> tuple[McpToolWrapper, ...]:
        if not isinstance(self.config, McpStdioConfig):
            raise McpConfigurationError("MCP client transport does not match stdio")
        return await self.connect()

    async def http_connect(self) -> tuple[McpToolWrapper, ...]:
        if not isinstance(self.config, McpHttpConfig):
            raise McpConfigurationError("MCP client transport does not match Streamable HTTP")
        return await self.connect()

    async def reconnect(self) -> tuple[McpToolWrapper, ...]:
        async with self._lock:
            await self._close_unlocked()
            self.status = McpClientStatus.CONNECTING
            return await self._connect_unlocked()

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def get_tools(self) -> tuple[McpToolWrapper, ...]:
        if self.sdk_client is None:
            raise McpConfigurationError("MCP client is not connected")
        listed = await self.sdk_client.list_tools()
        wrappers = tuple(
            McpToolWrapper(
                self,
                self.name,
                item.name,
                item.description,
                dict(item.input_schema),
            )
            for item in listed.tools
        )
        public_names = tuple(wrapper.name for wrapper in wrappers)
        if len(set(public_names)) != len(public_names):
            raise McpConfigurationError("MCP server contains colliding public tool names")
        self._raw_tool_names = {wrapper.name: wrapper.original_tool_name for wrapper in wrappers}
        return wrappers

    async def execute(self, tool_name: str, arguments: dict[str, object]) -> McpCallResult:
        client = self.sdk_client
        if client is None or self.status is not McpClientStatus.CONNECTED:
            return McpCallResult("MCP tool call failed", True)
        try:
            async with asyncio.timeout(self.config.tool_timeout_seconds):
                result = await client.call_tool(tool_name, arguments)
            return _result_text(result)
        except TimeoutError:
            return McpCallResult("MCP tool call timed out", True)
        except asyncio.CancelledError:
            raise
        except Exception:
            return McpCallResult("MCP tool call failed", True)

    async def _connect_unlocked(self) -> tuple[McpToolWrapper, ...]:
        if self.sdk_client is not None:
            return self._wrappers
        self.status = McpClientStatus.CONNECTING
        stack = AsyncExitStack()
        try:
            target: object
            if isinstance(self.config, McpStdioConfig):
                target = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=self._environment(),
                    cwd=self._cwd(),
                )
            else:
                target = self.config.endpoint
            client = self._client_factory(target)
            entered = await stack.enter_async_context(client)
            self.stack = stack
            self.sdk_client = entered
            self.clientsession = getattr(entered, "session", None)
            wrappers = await self.get_tools()
            self._wrappers = wrappers
            self.status = McpClientStatus.CONNECTED
            return wrappers
        except BaseException:
            self.status = McpClientStatus.FAILED
            self.sdk_client = None
            self.clientsession = None
            self._wrappers = ()
            self._raw_tool_names = {}
            self.stack = None
            await stack.aclose()
            raise

    async def _close_unlocked(self) -> None:
        stack, self.stack = self.stack, None
        self.sdk_client = None
        self.clientsession = None
        self._wrappers = ()
        self._raw_tool_names = {}
        self.status = McpClientStatus.FAILED
        if stack is not None:
            await stack.aclose()

    def _environment(self) -> dict[str, str]:
        assert isinstance(self.config, McpStdioConfig)
        environment: dict[str, str] = {}
        for child_name, source_name in self.config.env.items():
            try:
                environment[child_name] = os.environ[source_name]
            except KeyError as exc:
                raise McpConfigurationError(
                    "MCP configured environment variable is unavailable"
                ) from exc
        return environment

    def _cwd(self) -> Path:
        assert isinstance(self.config, McpStdioConfig)
        candidate = self._workspace_root if self.config.cwd is None else Path(self.config.cwd)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._workspace_root / candidate).resolve()
        )
        if not resolved.is_dir() or not resolved.is_relative_to(self._workspace_root):
            raise McpConfigurationError(
                "MCP working directory is outside the workspace or unavailable"
            )
        return resolved


def _result_text(result: Any) -> McpCallResult:
    """Convert public SDK content blocks to bounded-safe plain text at the boundary."""
    content = getattr(result, "content", ())
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                parts.append(resource.text)
            else:
                parts.append(
                    f"[MCP blob resource: {resource.mime_type or 'unknown'}, "
                    f"{_blob_size(resource.blob)} bytes omitted]"
                )
        elif isinstance(block, ResourceLink):
            title = block.title or block.name
            details = "; ".join(
                value for value in (block.uri, block.description, block.mime_type) if value
            )
            parts.append(f"[MCP resource link: {title}; {details}]")
        elif isinstance(block, ImageContent):
            parts.append(f"[MCP image: {block.mime_type}, {len(block.data)} encoded bytes omitted]")
        elif isinstance(block, AudioContent):
            parts.append(f"[MCP audio: {block.mime_type}, {len(block.data)} encoded bytes omitted]")
        else:
            content_type = getattr(block, "type", type(block).__name__)
            parts.append(f"[unsupported MCP content: {content_type}]")
    if not parts and getattr(result, "structured_content", None) is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False, sort_keys=True))
    return McpCallResult("\n".join(parts), bool(getattr(result, "is_error", False)))


def _blob_size(value: str) -> int:
    try:
        return len(base64.b64decode(value, validate=False))
    except (ValueError, TypeError):
        return len(value)

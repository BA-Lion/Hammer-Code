from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hammer_code.config import McpHttpConfig, McpStdioConfig
from hammer_code.mcp.client import McpClient
from hammer_code.mcp.manager import McpManager
from hammer_code.mcp.types import McpClientStatus, McpManagerStatus
from hammer_code.tools.registry import ToolRegistry


class _Sdk:
    def __init__(self, *, fail_tools: bool = False) -> None:
        self.session = object()
        self.closed = False
        self.fail_tools = fail_tools

    async def __aenter__(self) -> _Sdk:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def list_tools(self) -> SimpleNamespace:
        if self.fail_tools:
            raise RuntimeError("bad server")
        return SimpleNamespace(
            tools=(
                SimpleNamespace(
                    name="echo",
                    description="Echo a value",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                ),
            )
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            content=(), structured_content={"name": name, **arguments}, is_error=False
        )


def _config(name: str = "test") -> McpStdioConfig:
    return McpStdioConfig(
        name=name,
        description="Test MCP",
        transport="stdio",
        command="python",
        args=("server.py",),
        env={"CHILD_TOKEN": "SOURCE_TOKEN"},
    )


@pytest.mark.asyncio
async def test_client_owns_sdk_context_and_never_leaks_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_TOKEN", "secret")
    captured: list[object] = []
    sdk = _Sdk()
    client = McpClient(_config(), tmp_path, lambda target: captured.append(target) or sdk)
    wrappers = await client.connect()
    assert client.status is McpClientStatus.CONNECTED
    assert len(wrappers) == 1
    parameters = captured[0]
    assert parameters.env == {"CHILD_TOKEN": "secret"}  # type: ignore[union-attr]
    result = await client.execute("echo", {"value": "x"})
    assert result.content == '{"name": "echo", "value": "x"}'
    await client.close()
    assert sdk.closed and client.status is McpClientStatus.FAILED


@pytest.mark.asyncio
async def test_client_failure_closes_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_TOKEN", "secret")
    sdk = _Sdk(fail_tools=True)
    client = McpClient(_config(), tmp_path, lambda _: sdk)
    with pytest.raises(RuntimeError):
        await client.connect()
    assert sdk.closed and client.status is McpClientStatus.FAILED


@pytest.mark.asyncio
async def test_http_client_factory_receives_the_configured_streamable_endpoint(
    tmp_path: Path,
) -> None:
    config = McpHttpConfig(
        name="local-http",
        description="Local HTTP fixture",
        transport="streamable_http",
        endpoint="http://127.0.0.1:8000/mcp",
    )
    captured: list[object] = []
    sdk = _Sdk()
    client = McpClient(config, tmp_path, lambda target: captured.append(target) or sdk)
    try:
        await client.connect()
        assert captured == ["http://127.0.0.1:8000/mcp"]
    finally:
        await client.close()


class _FakeClient:
    def __init__(self, config: McpStdioConfig, _: Path, order: list[str], fail: bool) -> None:
        self.config = config
        self.order = order
        self.fail = fail
        self.closed = False

    async def connect(self) -> tuple:
        self.order.append(self.config.name)
        if self.fail:
            raise RuntimeError("unavailable")
        return ()

    async def reconnect(self) -> tuple:
        return await self.connect()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_manager_connects_in_order_isolates_failure_and_updates_cached_prompt(
    tmp_path: Path,
) -> None:
    order: list[str] = []

    def builder(config: McpStdioConfig, root: Path) -> _FakeClient:
        return _FakeClient(config, root, order, config.name == "bad")

    registry = ToolRegistry()
    manager = McpManager(
        (_config("bad"), _config("good")),
        registry,
        tmp_path,
        client_builder=builder,  # type: ignore[arg-type]
    )
    initial_prompt = manager.prompt
    assert "Overall status: connecting" in initial_prompt
    task = manager.start()
    assert task is not None
    await task
    assert order == ["bad", "good"]
    assert manager.status is McpManagerStatus.SETTLED
    assert "bad [failed]" in manager.prompt
    assert "good [connected]" in manager.prompt
    assert initial_prompt != manager.prompt
    await manager.close()

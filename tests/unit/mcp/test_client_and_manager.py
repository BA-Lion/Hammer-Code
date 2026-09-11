from __future__ import annotations

import asyncio
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


class _HttpClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    async def __aenter__(self) -> _HttpClient:
        self.events.append("http enter")
        return self

    async def __aexit__(self, *args: object) -> None:
        self.events.append("http exit")
        self.closed = True


class _OrderedSdk(_Sdk):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def __aenter__(self) -> _OrderedSdk:
        self.events.append("sdk enter")
        return self

    async def __aexit__(self, *args: object) -> None:
        self.events.append("sdk exit")
        await super().__aexit__(*args)


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


@pytest.mark.asyncio
async def test_http_bearer_uses_managed_client_and_closes_it_after_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "test_bearer_token"
    monkeypatch.setenv("TEST_MCP_BEARER", sentinel)
    config = McpHttpConfig(
        name="authenticated-http",
        description="Authenticated HTTP fixture",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env="TEST_MCP_BEARER",
    )
    events: list[str] = []
    received: dict[str, object] = {}
    http_options: dict[str, object] = {}
    http_client = _HttpClient(events)
    sdk = _OrderedSdk(events)

    def http_client_factory(**kwargs: object) -> _HttpClient:
        http_options.update(kwargs)
        return http_client

    def transport_factory(endpoint: str, *, http_client: _HttpClient) -> object:
        received["endpoint"] = endpoint
        received["transport_http_client"] = http_client
        return object()

    client = McpClient(
        config,
        tmp_path,
        lambda _: sdk,
        http_client_factory=http_client_factory,
        http_transport_factory=transport_factory,
    )
    await client.connect()
    await client.close()

    headers = http_options["headers"]
    assert headers == {"Authorization": f"Bearer {sentinel}"}
    assert set(http_options) == {"headers", "timeout"}
    assert received["endpoint"] == "https://example.test/mcp"
    assert received["transport_http_client"] is http_client
    assert events == ["http enter", "sdk enter", "sdk exit", "http exit"]
    assert http_client.closed and sdk.closed


@pytest.mark.asyncio
async def test_http_bearer_failure_and_cancellation_close_entered_http_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_MCP_BEARER", "test_bearer_token")
    config = McpHttpConfig(
        name="authenticated-http",
        description="Authenticated HTTP fixture",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env="TEST_MCP_BEARER",
    )

    transport_http = _HttpClient([])
    transport_failure = McpClient(
        config,
        tmp_path,
        lambda _: _Sdk(),
        http_client_factory=lambda **_: transport_http,
        http_transport_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("transport unavailable")
        ),
    )
    with pytest.raises(RuntimeError, match="transport unavailable"):
        await transport_failure.connect()
    assert transport_http.closed

    class _BlockingSdk(_Sdk):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()

        async def __aenter__(self) -> _BlockingSdk:
            self.entered.set()
            return self

        async def list_tools(self) -> SimpleNamespace:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    cancellation_http = _HttpClient([])
    blocking_sdk = _BlockingSdk()
    cancellation = McpClient(
        config,
        tmp_path,
        lambda _: blocking_sdk,
        http_client_factory=lambda **_: cancellation_http,
        http_transport_factory=lambda *_args, **_kwargs: object(),
    )
    task = asyncio.create_task(cancellation.connect())
    await blocking_sdk.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation_http.closed and blocking_sdk.closed
    assert cancellation.status is McpClientStatus.FAILED


@pytest.mark.asyncio
async def test_http_bearer_reconnect_recreates_resources_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_MCP_BEARER", "test_bearer_token")
    config = McpHttpConfig(
        name="authenticated-http",
        description="Authenticated HTTP fixture",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env="TEST_MCP_BEARER",
    )
    http_clients: list[_HttpClient] = []
    sdk_clients: list[_Sdk] = []

    def http_client_factory(**_: object) -> _HttpClient:
        client = _HttpClient([])
        http_clients.append(client)
        return client

    def client_factory(_: object) -> _Sdk:
        client = _Sdk()
        sdk_clients.append(client)
        return client

    client = McpClient(
        config,
        tmp_path,
        client_factory,
        http_client_factory=http_client_factory,
        http_transport_factory=lambda *_args, **_kwargs: object(),
    )
    await client.connect()
    await client.reconnect()
    await client.close()
    await client.close()

    assert len(http_clients) == len(sdk_clients) == 2
    assert all(item.closed for item in http_clients)
    assert all(item.closed for item in sdk_clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "", " token", "token ", "bad\nvalue", "bad=value+"])
async def test_http_bearer_missing_or_invalid_token_is_safely_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    from hammer_code.errors import McpConfigurationError

    variable = "TEST_MCP_BEARER"
    if token is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, token)
    config = McpHttpConfig(
        name="authenticated-http",
        description="Authenticated HTTP fixture",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env=variable,
    )
    client = McpClient(config, tmp_path, lambda _: _Sdk())

    with pytest.raises(
        McpConfigurationError,
        match="MCP bearer token environment variable is unavailable or invalid",
    ):
        await client.connect()
    assert client.status is McpClientStatus.FAILED
    assert client.sdk_client is None


@pytest.mark.asyncio
async def test_http_bearer_failure_is_isolated_from_following_manager_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_MCP_BEARER", raising=False)
    order: list[str] = []
    authenticated = McpHttpConfig(
        name="authenticated",
        description="Authenticated MCP",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env="MISSING_MCP_BEARER",
    )

    def builder(config: McpHttpConfig | McpStdioConfig, root: Path) -> McpClient | _FakeClient:
        if config.name == "authenticated":
            return McpClient(config, root, lambda _: _Sdk())
        assert isinstance(config, McpStdioConfig)
        return _FakeClient(config, root, order, False)

    manager = McpManager(
        (authenticated, _config("following")),
        ToolRegistry(),
        tmp_path,
        client_builder=builder,  # type: ignore[arg-type]
    )
    task = manager.start()
    assert task is not None
    await task
    assert order == ["following"]
    assert manager.status is McpManagerStatus.SETTLED
    assert "authenticated [failed]" in manager.prompt
    assert "MISSING_MCP_BEARER" not in manager.prompt
    await manager.close()


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

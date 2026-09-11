import sys
from pathlib import Path

import pytest

from hammer_code.config import McpStdioConfig
from hammer_code.mcp.client import McpClient


@pytest.mark.asyncio
async def test_real_sdk_stdio_client_lists_calls_and_closes_local_echo_server(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).with_name("mcp_stdio_echo_server.py")
    config = McpStdioConfig(
        name="echo",
        description="Local echo fixture",
        transport="stdio",
        command=sys.executable,
        args=(str(fixture),),
        cwd=str(tmp_path),
        connect_timeout_seconds=15,
    )
    client = McpClient(config, tmp_path)
    try:
        tools = await client.connect()
        assert [tool.original_tool_name for tool in tools] == ["echo"]
        result = await client.execute("echo", {"value": "hello"})
        assert result.content == "hello"
        assert not result.is_error
    finally:
        await client.close()

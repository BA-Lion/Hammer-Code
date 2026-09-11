"""A no-side-effect MCP stdio fixture used by the SDK contract test."""

from mcp.server import MCPServer

server = MCPServer("hammer-code-test")


@server.tool()
def echo(value: str) -> str:
    """Return a value without reading or writing anything."""
    return value


if __name__ == "__main__":
    server.run()

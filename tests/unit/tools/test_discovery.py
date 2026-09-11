from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from hammer_code.mcp.types import McpManagerStatus
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.builtin.discovery import ToolSearchTool
from hammer_code.tools.registry import ToolRegistry


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str


class _Deferred(Tool):
    input_model = _Input
    category = ToolCategory.COMMAND
    concurrency_policy = ConcurrencyPolicy.SERIAL
    should_defer = True

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        return ToolExecutionResult("ok")


@pytest.mark.asyncio
async def test_tool_search_discovers_top_candidates_without_connecting() -> None:
    registry = ToolRegistry()
    registry.register(_Deferred("mcp__alpha", "Search repository issues"), owner="alpha")
    registry.register(_Deferred("mcp__beta", "Browse documents"), owner="beta")
    manager = SimpleNamespace(configs=(object(),), status=McpManagerStatus.SETTLED)
    tool = ToolSearchTool(registry, manager)  # type: ignore[arg-type]
    result = await tool.execute(
        ToolExecutionContext(Path("."), Path("."), Path("."), {}),
        tool.input_model.model_validate({"query": "SEARCH issues"}),
    )
    assert result.content.splitlines()[0].startswith("mcp__alpha (score 2)")
    assert registry.exposed("mcp__alpha") and registry.exposed("mcp__beta")
    assert result.content.endswith("No more loaded, undiscovered MCP tools are available.")

"""The always-public MCP tool discovery tool."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hammer_code.mcp.manager import McpManager
from hammer_code.mcp.types import McpManagerStatus
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.registry import ToolRegistry


class ToolSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(min_length=1, max_length=1000)

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\0" in normalized:
            raise ValueError("query must be non-empty and contain no NUL")
        return normalized


class ToolSearchTool(Tool):
    name = "toolSearch"
    description = (
        "Search loaded MCP tools by a short, specific capability intent before using them."
    )
    input_model = ToolSearchInput
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    def __init__(self, registry: ToolRegistry, manager: McpManager) -> None:
        self._registry = registry
        self._manager = manager

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, ToolSearchInput):
            return ToolExecutionResult("toolSearch received invalid arguments", True)
        query = arguments.query.casefold()
        words = tuple(dict.fromkeys(re.findall(r"\w+", query)))
        candidates = []
        for tool in self._registry.deferred_tools():
            description = tool.description.strip()
            score = (5 if query == description.casefold() else 0) + sum(
                3 * (word in tool.name.casefold()) + (word in description.casefold())
                for word in words
            )
            candidates.append((score, tool))
        selected = sorted(candidates, key=lambda item: (-item[0], item[1].name))[:5]
        lines = []
        for score, tool in selected:
            self._registry.discover(tool.name)
            lines.append(f"{tool.name} (score {score}): {tool.description}")
        if len(selected) < 5:
            if not self._manager.configs:
                lines.append("No MCP servers are configured.")
            elif self._manager.status is McpManagerStatus.CONNECTING:
                lines.append("Some MCP servers are still loading.")
            else:
                lines.append("No more loaded, undiscovered MCP tools are available.")
        return ToolExecutionResult("\n".join(lines) or "No matching MCP tools found")

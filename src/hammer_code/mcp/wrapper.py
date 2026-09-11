"""SDK-free Hammer Code tools that delegate to a single MCP client."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from hammer_code.domain.events import ToolDefinition
from hammer_code.mcp.schema import input_model_from_schema
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from hammer_code.mcp.client import McpClient


class McpToolWrapper(Tool):
    should_defer = True
    category = ToolCategory.COMMAND
    concurrency_policy = ConcurrencyPolicy.SERIAL

    def __init__(
        self,
        client: McpClient,
        client_name: str,
        original_tool_name: str,
        description: str | None,
        input_schema: dict[str, object],
    ) -> None:
        self.client = client
        self.client_name = client_name
        self.original_tool_name = original_tool_name
        self.name = public_tool_name(client_name, original_tool_name)
        self.input_model, self._input_schema = input_model_from_schema(input_schema, self.name)
        self.description = (
            f"[MCP: {client_name}] {description.strip()}"
            if description and description.strip()
            else f"[MCP: {client_name}] Tool {original_tool_name} from MCP server {client_name}."
        )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, copy.deepcopy(self._input_schema))

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        result = await self.client.execute(
            self.original_tool_name,
            arguments.model_dump(mode="json", exclude_unset=True),
        )
        return ToolExecutionResult(result.content, result.is_error)


def public_tool_name(server: str, tool: str) -> str:
    server_slug = _slug(server, "server")[:18]
    tool_slug = _slug(tool, "tool")[:22]
    digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:10]
    return f"mcp__{server_slug}__{tool_slug}__{digest}"


def _slug(value: str, fallback: str) -> str:
    result = re.sub(r"[^a-z0-9_-]+", "_", value.casefold()).strip("_-")
    return result or fallback

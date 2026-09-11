"""Tool registration and schema projection for model requests."""

from __future__ import annotations

from hammer_code.domain.events import ToolDefinition
from hammer_code.errors import ToolError
from hammer_code.tools.base import Tool


class ToolRegistry:
    def __init__(self, disabled: tuple[str, ...] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled = set(disabled)
        self._discovered_tools: set[str] = set()
        self._owners: dict[str, set[str]] = {}
        self._tool_owners: dict[str, str] = {}

    def register(self, tool: Tool, *, owner: str | None = None) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._validate_tool(tool)
        self._tools[tool.name] = tool
        if owner is not None:
            self._owners.setdefault(owner, set()).add(tool.name)
            self._tool_owners[tool.name] = owner

    def replace_owner(self, owner: str, tools: tuple[Tool, ...]) -> None:
        """Atomically replace every tool belonging to one MCP client."""
        names = tuple(tool.name for tool in tools)
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate tool names for owner {owner}")
        for tool in tools:
            self._validate_tool(tool)
            existing_owner = self._tool_owners.get(tool.name)
            if tool.name in self._tools and existing_owner != owner:
                raise ValueError(f"Tool already registered: {tool.name}")

        old_names = self._owners.get(owner, set())
        replacement = dict(self._tools)
        for name in old_names:
            replacement.pop(name, None)
        for tool in tools:
            replacement[tool.name] = tool
        self._tools = replacement
        self._discovered_tools.difference_update(old_names)
        for name in old_names:
            self._tool_owners.pop(name, None)
        if tools:
            self._owners[owner] = set(names)
            self._tool_owners.update({name: owner for name in names})
        else:
            self._owners.pop(owner, None)

    def remove_owner(self, owner: str) -> None:
        names = self._owners.pop(owner, set())
        for name in names:
            self._tools.pop(name, None)
            self._tool_owners.pop(name, None)
        self._discovered_tools.difference_update(names)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def enabled(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled

    def discover(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None or not tool.should_defer or not self.enabled(name):
            raise ToolError("Tool is not an enabled deferred tool")
        self._discovered_tools.add(name)
        return tool

    def deferred_tools(self) -> tuple[Tool, ...]:
        return tuple(
            self._tools[name]
            for name in sorted(self._tools)
            if self.enabled(name)
            and self._tools[name].should_defer
            and name not in self._discovered_tools
        )

    def exposed(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(
            tool
            and self.enabled(name)
            and (not tool.should_defer or name in self._discovered_tools)
        )

    def clear_discovered(self) -> None:
        self._discovered_tools.clear()

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            self._tools[name].definition() for name in sorted(self._tools) if self.exposed(name)
        )

    def exposed_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions())

    @staticmethod
    def _validate_tool(tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        tool.definition()

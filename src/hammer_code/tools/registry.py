"""Tool registration and schema projection for model requests."""

from __future__ import annotations

from hammer_code.domain.events import ToolDefinition
from hammer_code.tools.base import Tool


def _remove_titles(value: object) -> object:
    if isinstance(value, dict):
        return {key: _remove_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_remove_titles(item) for item in value]
    return value


class ToolRegistry:
    def __init__(self, disabled: tuple[str, ...] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled = set(disabled)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def enabled(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled

    def definitions(self) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        for name in sorted(self._tools):
            if not self.enabled(name):
                continue
            tool = self._tools[name]
            schema = _remove_titles(tool.input_model.model_json_schema())
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError(f"Tool {name} must have an object input schema")
            schema["additionalProperties"] = False
            definitions.append(ToolDefinition(tool.name, tool.description, schema))
        return tuple(definitions)

    def exposed_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions())

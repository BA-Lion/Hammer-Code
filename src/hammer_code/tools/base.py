"""Stable tool contract, independent of any model SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from hammer_code.domain.events import ToolDefinition


class ToolCategory(StrEnum):
    READ = "read"
    WRITE = "write"
    COMMAND = "command"


class ConcurrencyPolicy(StrEnum):
    PARALLEL_READ = "parallel_read"
    SERIAL = "serial"


@dataclass(frozen=True)
class ToolExecutionContext:
    workspace_root: Path
    cwd: Path
    runtime_dir: Path
    sanitized_env: Mapping[str, str]


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    is_error: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


class Tool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    category: ToolCategory
    concurrency_policy: ConcurrencyPolicy
    should_defer: bool = False

    def definition(self) -> ToolDefinition:
        """Project the local Pydantic contract into a model-facing tool definition."""
        schema = _remove_titles(self.input_model.model_json_schema())
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"Tool {self.name} must have an object input schema")
        schema["additionalProperties"] = False
        return ToolDefinition(self.name, self.description, schema)

    @abstractmethod
    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult: ...


def _remove_titles(value: object) -> object:
    if isinstance(value, dict):
        return {key: _remove_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_remove_titles(item) for item in value]
    return value

"""Stable tool contract, independent of any model SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


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

    @abstractmethod
    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult: ...

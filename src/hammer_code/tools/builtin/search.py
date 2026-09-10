"""Bounded recursive text search and glob tools."""

from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hammer_code.permissions.paths import PathPolicy
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)

SKIPPED_NAMES = {".git", ".venv", ".hammer-code", "__pycache__"}


class GrepInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pattern: str
    path: str = "."
    include_glob: str | None = None
    case_sensitive: bool = True
    max_matches: int = Field(default=200, ge=1, le=2000)


class GlobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pattern: str
    path: str = "."
    max_results: int = Field(default=1000, ge=1, le=10000)


def _files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return [
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and not any(part in SKIPPED_NAMES for part in candidate.parts)
    ]


class GrepTool(Tool):
    name = "grep"
    description = "Search UTF-8 project files with a Python regular expression."
    input_model = GrepInput
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, GrepInput):
            return ToolExecutionResult("grep received invalid arguments", True)
        try:
            return await asyncio.to_thread(self._execute, context, arguments)
        except (OSError, UnicodeError, ValueError, re.error) as exc:
            return ToolExecutionResult(f"grep failed: {exc}", True)

    @staticmethod
    def _execute(context: ToolExecutionContext, arguments: GrepInput) -> ToolExecutionResult:
        target = PathPolicy(context.workspace_root).canonicalize(arguments.path)
        pattern = re.compile(arguments.pattern, 0 if arguments.case_sensitive else re.IGNORECASE)
        matches: list[str] = []
        skipped = 0
        for file_path in _files(target):
            if arguments.include_glob and not fnmatch.fnmatch(
                file_path.name, arguments.include_glob
            ):
                continue
            try:
                raw = file_path.read_bytes()
                if b"\0" in raw:
                    skipped += 1
                    continue
                text = raw.decode("utf-8")
            except (OSError, UnicodeError):
                skipped += 1
                continue
            relative = file_path.relative_to(context.workspace_root).as_posix()
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{relative}:{number}:{line}")
                    if len(matches) >= arguments.max_matches:
                        break
            if len(matches) >= arguments.max_matches:
                break
        suffix = f"\n[Skipped {skipped} binary or invalid UTF-8 files]" if skipped else ""
        return ToolExecutionResult("\n".join(matches) + suffix or "No matches found")


class GlobTool(Tool):
    name = "glob"
    description = "List project paths matching a glob pattern."
    input_model = GlobInput
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, GlobInput):
            return ToolExecutionResult("glob received invalid arguments", True)
        try:
            return await asyncio.to_thread(self._execute, context, arguments)
        except (OSError, ValueError) as exc:
            return ToolExecutionResult(f"glob failed: {exc}", True)

    @staticmethod
    def _execute(context: ToolExecutionContext, arguments: GlobInput) -> ToolExecutionResult:
        target = PathPolicy(context.workspace_root).canonicalize(arguments.path)
        paths = sorted(
            {
                item.relative_to(context.workspace_root).as_posix()
                for item in target.glob(arguments.pattern)
                if not any(part in SKIPPED_NAMES for part in item.parts)
            }
        )
        clipped = paths[: arguments.max_results]
        suffix = (
            f"\n[Truncated after {arguments.max_results} paths]"
            if len(paths) > len(clipped)
            else ""
        )
        return ToolExecutionResult("\n".join(clipped) + suffix or "No paths found")

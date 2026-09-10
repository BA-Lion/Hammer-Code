"""UTF-8, project-bound read and atomic text-edit tools."""

from __future__ import annotations

import asyncio
import os
import tempfile
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

MAX_FILE_BYTES = 4 * 1024 * 1024


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=400, ge=1, le=2000)


class EditFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    old_text: str = Field(min_length=1)
    new_text: str


class CreateFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    content: str


def _text_file(policy: PathPolicy, value: str) -> tuple[Path, str]:
    target = policy.verify_before_use(value)
    if not target.is_file():
        raise ValueError("Path must be an existing regular file")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds 4 MiB limit")
    data = target.read_bytes()
    if b"\0" in data:
        raise ValueError("Binary files are not supported")
    return target, data.decode("utf-8")


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file under the project root with line numbers."
    input_model = ReadFileInput
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.PARALLEL_READ

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, ReadFileInput):
            return ToolExecutionResult("read_file received invalid arguments", True)
        try:
            return await asyncio.to_thread(self._execute, context, arguments)
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolExecutionResult(f"read_file failed: {exc}", True)

    @staticmethod
    def _execute(context: ToolExecutionContext, arguments: ReadFileInput) -> ToolExecutionResult:
        target, text = _text_file(PathPolicy(context.workspace_root), arguments.path)
        lines = text.splitlines()
        start = arguments.start_line - 1
        if start >= len(lines):
            raise ValueError("start_line is beyond the end of the file")
        selected = lines[start : start + arguments.line_count]
        body = "\n".join(f"{index + start + 1}: {line}" for index, line in enumerate(selected))
        relative = target.relative_to(context.workspace_root).as_posix()
        return ToolExecutionResult(
            f"{relative} (total lines: {len(lines)}, "
            f"showing {start + 1}-{start + len(selected)})\n{body}"
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace one unique UTF-8 text fragment in an existing project file."
    input_model = EditFileInput
    category = ToolCategory.WRITE
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, EditFileInput):
            return ToolExecutionResult("edit_file received invalid arguments", True)
        try:
            return await asyncio.to_thread(self._execute, context, arguments)
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolExecutionResult(f"edit_file failed: {exc}", True)

    @staticmethod
    def _execute(context: ToolExecutionContext, arguments: EditFileInput) -> ToolExecutionResult:
        policy = PathPolicy(context.workspace_root)
        target, text = _text_file(policy, arguments.path)
        count = text.count(arguments.old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once (found {count})")
        replacement = text.replace(arguments.old_text, arguments.new_text, 1)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(replacement)
                handle.flush()
                os.fsync(handle.fileno())
            policy.verify_before_use(target)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ToolExecutionResult(
            f"Updated {target.relative_to(context.workspace_root).as_posix()}"
        )


class CreateFileTool(Tool):
    name = "create_file"
    description = (
        "Create a new UTF-8 project file without creating directories or overwriting files."
    )
    input_model = CreateFileInput
    category = ToolCategory.WRITE
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, CreateFileInput):
            return ToolExecutionResult("create_file received invalid arguments", True)
        try:
            return await asyncio.to_thread(self._execute, context, arguments)
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolExecutionResult(f"create_file failed: {exc}", True)

    @staticmethod
    def _execute(context: ToolExecutionContext, arguments: CreateFileInput) -> ToolExecutionResult:
        encoded = arguments.content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise ValueError("Content exceeds 4 MiB limit")
        policy = PathPolicy(context.workspace_root)
        target = policy.verify_before_use(arguments.path)
        if not target.parent.is_dir():
            raise ValueError("Parent directory does not exist")
        policy.verify_before_use(target.parent)
        with target.open("x", encoding="utf-8", newline="") as handle:
            handle.write(arguments.content)
        return ToolExecutionResult(
            f"Created {target.relative_to(context.workspace_root).as_posix()}"
        )

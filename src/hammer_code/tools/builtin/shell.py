"""A time-bounded, environment-sanitized PowerShell command tool."""

from __future__ import annotations

import asyncio
import os
import subprocess

from pydantic import BaseModel, ConfigDict, Field

from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)


class ShellInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: str = Field(min_length=1, max_length=16384)
    timeout_seconds: int = Field(default=120, ge=1, le=1800)


def sanitized_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if not re_sensitive(key) and not key.casefold().endswith("api_key_env")
    }


def re_sensitive(key: str) -> bool:
    parts = ("key", "token", "secret", "password", "credential")
    return any(part in key.casefold() for part in parts)


class ShellTool(Tool):
    name = "shell"
    description = "Run one approved PowerShell command in the project root with a time limit."
    input_model = ShellInput
    category = ToolCategory.COMMAND
    concurrency_policy = ConcurrencyPolicy.SERIAL

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        if not isinstance(arguments, ShellInput):
            return ToolExecutionResult("shell received invalid arguments", True)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            arguments.command,
            cwd=context.workspace_root,
            env=dict(context.sanitized_env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=flags,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), arguments.timeout_seconds
            )
        except asyncio.CancelledError:
            await self._terminate_tree(process)
            raise
        except TimeoutError:
            await self._terminate_tree(process)
            return ToolExecutionResult("shell timed out", True)
        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")
        content = output + (f"\n[stderr]\n{errors}" if errors else "")
        return ToolExecutionResult(
            content or "Command completed with no output",
            process.returncode != 0,
            output,
            errors,
            process.returncode,
        )

    @staticmethod
    async def _terminate_tree(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await process.wait()

"""A time-bounded, environment-sanitized PowerShell command tool."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.command import (
    run_powershell,
    terminate_process_tree,
)
from hammer_code.tools.command import sanitized_environment as _sanitized_environment


class ShellInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: str = Field(min_length=1, max_length=16384)
    timeout_seconds: int = Field(default=120, ge=1, le=1800)


def sanitized_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Compatibility import location for existing runtime assembly."""
    return _sanitized_environment(environment)


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
        return await run_powershell(context, arguments.command, arguments.timeout_seconds)

    @staticmethod
    async def _terminate_tree(process: asyncio.subprocess.Process) -> None:
        await terminate_process_tree(process)

"""Shared, bounded PowerShell execution used by tools and configured Hooks."""

from __future__ import annotations

import asyncio
import os
import subprocess

from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult


def sanitized_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if not _is_sensitive_environment_name(key) and not key.casefold().endswith("api_key_env")
    }


async def run_powershell(
    context: ToolExecutionContext, command: str, timeout_seconds: int
) -> ToolExecutionResult:
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    process = await asyncio.create_subprocess_exec(
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        command,
        cwd=context.cwd,
        env=dict(context.sanitized_env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=flags,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except asyncio.CancelledError:
        await terminate_process_tree(process)
        raise
    except TimeoutError:
        await terminate_process_tree(process)
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


async def terminate_process_tree(process: asyncio.subprocess.Process) -> None:
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


def _is_sensitive_environment_name(key: str) -> bool:
    return any(
        part in key.casefold() for part in ("key", "token", "secret", "password", "credential")
    )

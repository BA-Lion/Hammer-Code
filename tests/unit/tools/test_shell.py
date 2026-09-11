import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from hammer_code.tools.base import ToolExecutionContext
from hammer_code.tools.builtin.shell import ShellInput, ShellTool, sanitized_environment

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell tool is Windows-only")


def _context(root: Path) -> ToolExecutionContext:
    return ToolExecutionContext(root, root, root, sanitized_environment())


@pytest.mark.asyncio
async def test_shell_timeout_terminates_a_real_harmless_process(tmp_path: Path) -> None:
    if shutil.which("powershell.exe") is None:
        pytest.skip("PowerShell is unavailable")

    result = await ShellTool().execute(
        _context(tmp_path), ShellInput(command="Start-Sleep -Seconds 5", timeout_seconds=1)
    )

    assert result.is_error
    assert result.content == "shell timed out"


@pytest.mark.asyncio
async def test_shell_cancellation_terminates_a_real_harmless_process(tmp_path: Path) -> None:
    if shutil.which("powershell.exe") is None:
        pytest.skip("PowerShell is unavailable")

    task = asyncio.create_task(
        ShellTool().execute(
            _context(tmp_path), ShellInput(command="Start-Sleep -Seconds 5", timeout_seconds=5)
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

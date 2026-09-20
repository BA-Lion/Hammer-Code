from __future__ import annotations

from pathlib import Path

import pytest

from hammer_code.hooks.manager import HookManager
from hammer_code.hooks.models import Action, ActionType, Hook, HookContext, LifecycleEvent
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult


class _Permissions:
    def __init__(self) -> None:
        self.requests = []

    async def authorize(self, request: object) -> None:
        self.requests.append(request)


class _UI:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        del message

    def error(self, message: str) -> None:
        self.errors.append(message)


@pytest.mark.asyncio
async def test_prompt_is_one_shot_and_once_hook_is_not_repeated(tmp_path: Path) -> None:
    manager = HookManager(
        (
            Hook(
                "turn-note",
                LifecycleEvent.TURN_START,
                Action(ActionType.PROMPT, prompt="Handle $MESSAGE"),
                once=True,
            ),
        ),
        _Permissions(),  # type: ignore[arg-type]
        ToolExecutionContext(tmp_path, tmp_path, tmp_path, {}),
        _UI(),  # type: ignore[arg-type]
    )
    await manager.dispatch(LifecycleEvent.TURN_START, HookContext(message="one"))
    await manager.dispatch(LifecycleEvent.TURN_START, HookContext(message="two"))
    batch = manager.snapshot_prompt()
    assert batch.text == "Handle one"
    manager.commit_prompt(batch)
    assert not manager.snapshot_prompt()


@pytest.mark.asyncio
async def test_pre_tool_reject_is_independent_of_prompt_action(tmp_path: Path) -> None:
    manager = HookManager(
        (
            Hook(
                "deny-shell",
                LifecycleEvent.PRE_TOOL_USE,
                Action(ActionType.PROMPT, prompt="No $TOOL_NAME"),
                reject=True,
            ),
        ),
        _Permissions(),  # type: ignore[arg-type]
        ToolExecutionContext(tmp_path, tmp_path, tmp_path, {}),
        _UI(),  # type: ignore[arg-type]
    )
    result = await manager.dispatch(LifecycleEvent.PRE_TOOL_USE, HookContext(tool_name="shell"))
    assert result.rejected
    assert manager.snapshot_prompt().text == "No shell"


@pytest.mark.asyncio
async def test_command_uses_shell_permission_identity_before_runner(tmp_path: Path) -> None:
    permissions = _Permissions()
    commands: list[str] = []

    async def runner(
        context: ToolExecutionContext, command: str, timeout: int
    ) -> ToolExecutionResult:
        del context, timeout
        commands.append(command)
        return ToolExecutionResult("ok")

    manager = HookManager(
        (
            Hook(
                "command",
                LifecycleEvent.TURN_START,
                Action(ActionType.COMMAND, command="Write-Output $MESSAGE"),
            ),
        ),
        permissions,  # type: ignore[arg-type]
        ToolExecutionContext(tmp_path, tmp_path, tmp_path, {}),
        _UI(),  # type: ignore[arg-type]
        runner,
    )
    await manager.dispatch(LifecycleEvent.TURN_START, HookContext(message="hello"))
    assert commands == ["Write-Output hello"]
    assert permissions.requests[0].tool_name == "shell"

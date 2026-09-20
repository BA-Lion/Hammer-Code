"""Per-Agent Hook dispatch, prompt queue, and background action ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from hammer_code.hooks.http import request as http_request
from hammer_code.hooks.models import (
    Action,
    ActionType,
    Hook,
    HookContext,
    HookDispatchResult,
    LifecycleEvent,
    PromptBatch,
    PromptItem,
    expand_template,
    matches_condition,
)
from hammer_code.permissions.models import PermissionRequest
from hammer_code.permissions.service import PermissionService
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult
from hammer_code.tools.builtin.shell import ShellInput
from hammer_code.tools.command import run_powershell
from hammer_code.ui.console import ConsolePort

CommandRunner = Callable[[ToolExecutionContext, str, int], Awaitable[ToolExecutionResult]]


class HookManager:
    def __init__(
        self,
        hooks: tuple[Hook, ...],
        permissions: PermissionService,
        context: ToolExecutionContext,
        ui: ConsolePort,
        command_runner: CommandRunner = run_powershell,
    ) -> None:
        self.hooks = hooks
        self.permissions = permissions
        self.context = context
        self.ui = ui
        self.command_runner = command_runner
        self._prompts: list[PromptItem] = []
        self._next_prompt_identity = 0
        self._background: set[asyncio.Task[None]] = set()
        self._accepting = True

    async def dispatch(
        self, event: LifecycleEvent, context: HookContext | None = None
    ) -> HookDispatchResult:
        if not self._accepting:
            return HookDispatchResult()
        context = context or HookContext()
        if not context.event_name:
            context = HookContext(
                event_name=event.value,
                tool_name=context.tool_name,
                tool_args=context.tool_args,
                file_path=context.file_path,
                message=context.message,
                error=context.error,
            )
        rejected = False
        for hook in self.hooks:
            if hook.event is not event or (hook.once and hook.executed):
                continue
            if not matches_condition(hook.condition, context):
                continue
            if hook.once:
                hook.executed = True
            rejected = rejected or hook.reject
            await self._dispatch_action(hook, context)
        return HookDispatchResult(rejected)

    def snapshot_prompt(self) -> PromptBatch:
        return PromptBatch(tuple(self._prompts))

    def commit_prompt(self, batch: PromptBatch) -> None:
        if batch.items and tuple(self._prompts[: len(batch.items)]) == batch.items:
            del self._prompts[: len(batch.items)]

    async def drain(self) -> None:
        self._accepting = False
        if self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    async def cancel(self) -> None:
        self._accepting = False
        tasks = tuple(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_action(self, hook: Hook, context: HookContext) -> None:
        try:
            action = self._expand_action(hook.action, context)
            if hook.async_exec and action.type in {ActionType.COMMAND, ActionType.HTTP}:
                if action.type is ActionType.COMMAND:
                    await self._authorize_command(action)
                self._track(self._execute_action(hook, action, authorized=True))
            else:
                await self._execute_action(hook, action)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._diagnose(hook, f"{type(exc).__name__}")

    async def _execute_action(
        self, hook: Hook, action: Action, *, authorized: bool = False
    ) -> None:
        try:
            if action.type is ActionType.PROMPT:
                self._next_prompt_identity += 1
                self._prompts.append(PromptItem(self._next_prompt_identity, action.prompt))
            elif action.type is ActionType.COMMAND:
                if not authorized:
                    await self._authorize_command(action)
                result = await self.command_runner(self.context, action.command, action.timeout)
                if result.is_error:
                    self._diagnose(hook, "command failed")
                else:
                    self.ui.info(f"Hook {hook.id} ({hook.event.value}, command) completed.")
            elif action.type is ActionType.HTTP:
                ok, detail = await http_request(
                    url=action.url,
                    method=action.method,
                    body=action.body,
                    headers=dict(action.headers),
                    timeout=action.timeout,
                )
                if ok:
                    self.ui.info(f"Hook {hook.id} ({hook.event.value}, http) completed.")
                else:
                    self._diagnose(hook, detail)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._diagnose(hook, type(exc).__name__)

    def _expand_action(self, action: Action, context: HookContext) -> Action:
        return Action(
            type=action.type,
            command=expand_template(action.command, context) if action.command else "",
            prompt=expand_template(action.prompt, context) if action.prompt else "",
            url=expand_template(action.url, context) if action.url else "",
            method=action.method,
            body=expand_template(action.body, context) if action.body else "",
            headers={key: expand_template(value, context) for key, value in action.headers.items()},
            timeout=action.timeout,
        )

    async def _authorize_command(self, action: Action) -> None:
        validated = ShellInput(command=action.command, timeout_seconds=action.timeout)
        request = PermissionRequest.for_command(
            "shell", validated.command, self.context.workspace_root, self.context.cwd
        )
        await self.permissions.authorize(request)

    def _track(self, coroutine: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coroutine)
        self._background.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self._background.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        task.add_done_callback(done)

    def _diagnose(self, hook: Hook, category: str) -> None:
        safe_category = category.replace("\n", " ")[:500]
        self.ui.error(
            f"Hook {hook.id} ({hook.event.value}, {hook.action.type.value}) failed: {safe_category}"
        )

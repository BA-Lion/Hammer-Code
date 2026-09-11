"""Local validation, authorization, bounded execution, and stable batch ordering."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from hammer_code.domain.messages import TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.permissions.models import PermissionRequest
from hammer_code.permissions.paths import PathPolicy
from hammer_code.permissions.service import PermissionService
from hammer_code.tools.base import ConcurrencyPolicy, ToolExecutionContext
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.results import bound_result
from hammer_code.tools.runtime import RuntimeStore


class ToolBatchCancelled(asyncio.CancelledError):
    """Cancellation carrying a result for every call that was not executed."""

    def __init__(self, results: tuple[ToolResultBlock, ...]) -> None:
        super().__init__("Tool execution was cancelled")
        self.results = results


@dataclass(frozen=True)
class ToolExecutor:
    registry: ToolRegistry
    permissions: PermissionService
    context: ToolExecutionContext
    runtime: RuntimeStore

    async def execute_batch(self, calls: tuple[ToolCallBlock, ...]) -> tuple[ToolResultBlock, ...]:
        prepared: list[tuple[ToolCallBlock, BaseModel, PermissionRequest] | ToolResultBlock] = []
        seen: set[str] = set()
        for call in calls:
            if call.call_id in seen:
                prepared.append(self._error(call.call_id, "duplicate call id"))
                continue
            seen.add(call.call_id)
            tool = self.registry.get(call.name)
            if tool is None or not self.registry.enabled(call.name):
                prepared.append(self._error(call.call_id, "unknown or disabled tool"))
                continue
            try:
                arguments = tool.input_model.model_validate(dict(call.arguments))
                paths = self._paths_for(tool.name, arguments)
                request = PermissionRequest.for_tool(
                    tool.name,
                    tool.category.value,
                    arguments.model_dump(mode="json"),
                    paths,
                    self.context.workspace_root,
                    self.context.cwd,
                )
                if tool.name == "shell":
                    request = PermissionRequest.for_command(
                        tool.name,
                        str(arguments.model_dump()["command"]),
                        self.context.workspace_root,
                        self.context.cwd,
                    )
                await self.permissions.authorize(request)
                prepared.append((call, arguments, request))
            except (ValidationError, ValueError, Exception) as exc:
                prepared.append(self._error(call.call_id, str(exc)))
        results: list[ToolResultBlock | None] = [None] * len(prepared)
        index = 0
        try:
            while index < len(prepared):
                item = prepared[index]
                if isinstance(item, ToolResultBlock):
                    results[index] = item
                    index += 1
                    continue
                call, arguments, _ = item
                tool = self.registry.get(call.name)
                assert tool is not None
                if tool.concurrency_policy is ConcurrencyPolicy.SERIAL:
                    results[index] = await self._run(call, arguments)
                    index += 1
                    continue
                end = index
                parallel: list[tuple[ToolCallBlock, BaseModel, PermissionRequest]] = []
                while end < len(prepared) and not isinstance(prepared[end], ToolResultBlock):
                    candidate = prepared[end]
                    assert not isinstance(candidate, ToolResultBlock)
                    candidate_tool = self.registry.get(candidate[0].name)
                    assert candidate_tool is not None
                    if candidate_tool.concurrency_policy is ConcurrencyPolicy.SERIAL:
                        break
                    parallel.append(candidate)
                    end += 1
                for position, result in zip(
                    range(index, end),
                    await asyncio.gather(*(self._run(item[0], item[1]) for item in parallel)),
                    strict=True,
                ):
                    results[position] = result
                index = end
        except asyncio.CancelledError as exc:
            cancelled = tuple(
                result
                if result is not None
                else self._error(
                    item[0].call_id if isinstance(item, tuple) else item.call_id,
                    "tool call cancelled before completion",
                )
                for item, result in zip(prepared, results, strict=True)
            )
            raise ToolBatchCancelled(cancelled) from exc
        return tuple(result for result in results if result is not None)

    def _paths_for(self, name: str, arguments: BaseModel) -> tuple:
        if name in {"read_file", "edit_file", "create_file", "grep", "glob"}:
            path = str(arguments.model_dump()["path"])
            return (PathPolicy(self.context.workspace_root).canonicalize(path),)
        return ()

    async def _run(self, call: ToolCallBlock, arguments: BaseModel) -> ToolResultBlock:
        tool = self.registry.get(call.name)
        assert tool is not None
        try:
            raw = await tool.execute(self.context, arguments)
            bounded = bound_result(raw.content, call.call_id, self.runtime)
            return ToolResultBlock(
                call.call_id, (TextBlock(bounded.content or "(empty output)"),), raw.is_error
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._error(call.call_id, "tool execution failed")

    @staticmethod
    def _error(call_id: str, message: str) -> ToolResultBlock:
        return ToolResultBlock(call_id, (TextBlock(f"Error: {message}"),), True)

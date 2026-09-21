"""Independent bounded model/tool loop for one Subagent invocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.domain.events import (
    ModelRequest,
    ResponseCompleted,
    StopReason,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.service import ApprovalPort, PermissionService
from hammer_code.subagent.models import SubagentContext, SubagentInvocation
from hammer_code.subagent.prompts import build_subagent_system
from hammer_code.subagent.usage import SubagentUsageTracker
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore


class SubagentRunner:
    MAX_TOOL_CALLS = 50
    MAX_UNKNOWN_TOOL_CALLS = 2

    def __init__(
        self,
        client: ModelClient,
        runtime: RuntimeStore,
        max_output_tokens: int,
        permissions: PermissionService,
        record_usage: Callable[[str, str, TokenUsage], None],
        estimator: TokenEstimator | None = None,
    ) -> None:
        self.client = client
        self.runtime = runtime
        self.max_output_tokens = max_output_tokens
        self.permissions = permissions
        self.record_usage = record_usage
        self.estimator = estimator or TokenEstimator()

    async def run(
        self,
        invocation: SubagentInvocation,
        registry: ToolRegistry,
        context: ToolExecutionContext,
        operation_id: str,
        usage_tracker: SubagentUsageTracker,
        approval_port: ApprovalPort | None = None,
    ) -> ToolExecutionResult:
        prefix = (
            invocation.parent.system_prompt
            if invocation.context is SubagentContext.FORK
            else "\n\n".join(
                piece
                for piece in (
                    invocation.parent.base_system_prompt,
                    invocation.parent.project_instructions,
                    invocation.parent.mcp_prompt,
                )
                if piece.strip()
            )
        )
        definition = _definition(invocation)
        messages = list(
            invocation.parent.messages if invocation.context is SubagentContext.FORK else ()
        )
        messages.append(Message(Role.USER, (TextBlock(invocation.task),)))
        child_context = replace(context, skill_invoker=None, subagent_invoker=None)
        child_permissions = PermissionService(
            self.permissions.checker,
            approval_port or self.permissions.approvals,
            self.permissions.rules,
        )
        executor = ToolExecutor(registry, child_permissions, child_context, self.runtime)
        calls = unknown = 0
        for _ in range(invocation.max_iterations):
            request_id = str(uuid4())
            completed: ResponseCompleted | None = None
            announced: set[str] = set()
            system_prompt = build_subagent_system(prefix, definition)
            tool_definitions = registry.definitions()
            request_messages = tuple(messages)
            estimated_input = self.estimator.estimate_request(
                system_prompt, request_messages, tool_definitions
            )
            allowed_max_output = usage_tracker.reserve(
                request_id, estimated_input, self.max_output_tokens
            )
            if allowed_max_output is None:
                return ToolExecutionResult("Error: Subagent token budget exhausted.", True)
            request = ModelRequest(
                request_id,
                operation_id,
                system_prompt,
                request_messages,
                tool_definitions,
                allowed_max_output,
            )
            try:
                async for event in self.client.stream(request):
                    if event.request_id != request_id:
                        return ToolExecutionResult(
                            "Error: Subagent stream request id mismatch.", True
                        )
                    if isinstance(event, ResponseCompleted):
                        if completed is not None:
                            return ToolExecutionResult(
                                "Error: Subagent received duplicate completion.", True
                            )
                        completed = event
                        self.record_usage(operation_id, request_id, event.response.usage)
                        usage_tracker.observe(request_id, event.response.usage)
                    elif isinstance(event, ToolCallCompleted):
                        announced.add(event.tool_call.call_id)
                    elif isinstance(event, UsageUpdated):
                        self.record_usage(operation_id, request_id, event.usage)
                        usage_tracker.observe(request_id, event.usage)
            except Exception:
                return ToolExecutionResult("Error: Subagent model request failed.", True)
            if completed is None:
                return ToolExecutionResult("Error: Subagent model request did not complete.", True)
            response = completed.response
            tool_calls = tuple(
                item for item in response.message.content if isinstance(item, ToolCallBlock)
            )
            if response.stop_reason is StopReason.END_TURN:
                text = "".join(
                    item.text for item in response.message.content if isinstance(item, TextBlock)
                )
                return ToolExecutionResult(
                    _clip_visible(text) or "Error: Subagent returned no visible text.",
                    not bool(text),
                )
            if response.stop_reason is not StopReason.TOOL_CALL:
                return ToolExecutionResult("Error: Subagent did not complete normally.", True)
            if not tool_calls or calls + len(tool_calls) > self.MAX_TOOL_CALLS:
                return ToolExecutionResult("Error: Subagent tool-call limit reached.", True)
            if {item.call_id for item in tool_calls} != announced:
                return ToolExecutionResult(
                    "Error: Subagent tool-call events did not match response.", True
                )
            calls += len(tool_calls)
            messages.append(response.message)
            results = await executor.execute_batch(tool_calls)
            unknown += sum(
                1 for item in results if item.is_error and "unknown" in item.content[0].text
            )
            if unknown > self.MAX_UNKNOWN_TOOL_CALLS:
                return ToolExecutionResult(
                    "Error: Subagent made too many unknown tool calls.", True
                )
            messages.append(Message(Role.USER, results))
            if usage_tracker.snapshot().exhausted:
                return ToolExecutionResult("Error: Subagent token budget exhausted.", True)
        return ToolExecutionResult("Error: Subagent iteration limit reached.", True)


def _definition(invocation: SubagentInvocation):
    from hammer_code.subagent.models import SubagentDefinition

    return SubagentDefinition(
        invocation.name,
        invocation.name,
        None,
        invocation.prompt,
        invocation.context,
        invocation.execution,
        invocation.allowed_tools,
        invocation.disallowed_tools,
        invocation.max_iterations,
        Path("."),
    )


def _clip_visible(value: str) -> str:
    if len(value) <= 32000:
        return value
    return value[:31968] + "\n[Subagent result truncated.]"

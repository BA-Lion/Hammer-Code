"""Event-driven Team model runner without a per-Agent iteration budget."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from hammer_code.agent_team.usage import TeamUsageTracker
from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.domain.events import (
    ModelRequest,
    ResponseCompleted,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.service import PermissionService
from hammer_code.tools.base import ToolExecutionContext, ToolExecutionResult
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore


class AgentTeamRunner:
    MAX_TOOL_CALLS = 50
    MAX_UNKNOWN = 3

    def __init__(
        self,
        client: ModelClient,
        runtime: RuntimeStore,
        max_output_tokens: int,
        permissions: PermissionService,
        record_usage: Callable[[str, str, TokenUsage], None],
    ) -> None:
        self.client = client
        self.runtime = runtime
        self.max_output_tokens = max_output_tokens
        self.permissions = permissions
        self.record_usage = record_usage
        self.estimator = TokenEstimator()

    async def run(
        self,
        *,
        operation_id: str,
        system_prompt: str,
        task: str,
        registry: ToolRegistry,
        context: ToolExecutionContext,
        usage: TeamUsageTracker,
        terminal: Callable[[], bool],
    ) -> ToolExecutionResult:
        messages = [Message(Role.USER, (TextBlock(task),))]
        calls = unknown = 0
        while not terminal():
            request_id = str(uuid4())
            definitions = registry.definitions()
            estimate = self.estimator.estimate_request(system_prompt, tuple(messages), definitions)
            allowed = await usage.reserve(request_id, estimate, self.max_output_tokens)
            if allowed is None:
                return ToolExecutionResult("Error: AgentTeam token budget exhausted.", True)
            completed: ResponseCompleted | None = None
            announced: set[str] = set()
            request = ModelRequest(
                request_id, operation_id, system_prompt, tuple(messages), definitions, allowed
            )
            try:
                async for event in self.client.stream(request):
                    if event.request_id != request_id:
                        return ToolExecutionResult("Error: AgentTeam request id mismatch.", True)
                    if isinstance(event, UsageUpdated):
                        self.record_usage(operation_id, request_id, event.usage)
                        await usage.observe(request_id, event.usage)
                    elif isinstance(event, ToolCallCompleted):
                        announced.add(event.tool_call.call_id)
                    elif isinstance(event, ResponseCompleted):
                        if completed is not None:
                            return ToolExecutionResult(
                                "Error: AgentTeam received duplicate completion.", True
                            )
                        completed = event
                        self.record_usage(operation_id, request_id, event.response.usage)
                        await usage.observe(request_id, event.response.usage)
            except Exception:
                return ToolExecutionResult("Error: AgentTeam model request failed.", True)
            if completed is None:
                return ToolExecutionResult("Error: AgentTeam model request did not complete.", True)
            response = completed.response
            tool_calls = tuple(
                item for item in response.message.content if isinstance(item, ToolCallBlock)
            )
            if not tool_calls or response.stop_reason.value != "tool_call":
                messages.append(response.message)
                messages.append(
                    Message(
                        Role.USER,
                        (
                            TextBlock(
                                "Protocol correction: use a Team tool, wait_for_message, "
                                "or explicit finish."
                            ),
                        ),
                    )
                )
                continue
            if (
                len(tool_calls) + calls > self.MAX_TOOL_CALLS
                or {item.call_id for item in tool_calls} != announced
            ):
                return ToolExecutionResult(
                    "Error: AgentTeam tool-call protocol limit reached.", True
                )
            calls += len(tool_calls)
            messages.append(response.message)
            executor = ToolExecutor(registry, self.permissions, context, self.runtime)
            results = await executor.execute_batch(tool_calls)
            unknown += sum(item.is_error and "unknown" in item.content[0].text for item in results)
            if unknown > self.MAX_UNKNOWN:
                return ToolExecutionResult(
                    "Error: AgentTeam made too many unknown tool calls.", True
                )
            messages.append(Message(Role.USER, results))
        return ToolExecutionResult("AgentTeam runtime finished.")

"""Context budgeting, history compaction, and non-authoritative file recovery hints."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import ContextConfig
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.events import (
    CompactEvent,
    ModelRequest,
    ResponseCompleted,
    StopReason,
    ToolDefinition,
    UsageUpdated,
)
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.errors import ContextCompactionError, HammerCodeError
from hammer_code.llm.client import ModelClient
from hammer_code.permissions.paths import PathPolicy
from hammer_code.prompts import COMPACT_BOUNDARY_MESSAGE, SUMMARY_PROMPT, build_recovery_prompt
from hammer_code.tools.results import clip_prefix
from hammer_code.tools.runtime import RuntimeStore

_SUMMARY = re.compile(r"\A<analysis>(.*?)</analysis>\s*<summary>(.*?)</summary>\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class RecoveryEntry:
    path: str
    content: str


class RecoveryState:
    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._entries: OrderedDict[str, RecoveryEntry] = OrderedDict()

    def record(self, path: str, content: str) -> None:
        key = Path(path).as_posix()
        if not key or key == ".":
            return
        self._entries[key] = RecoveryEntry(
            key, clip_prefix(content, self._config.recovery_file_tokens, TokenEstimator())
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._config.recovery_file_count:
            self._entries.popitem(last=False)

    def entries(self) -> tuple[RecoveryEntry, ...]:
        return tuple(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()


@dataclass(frozen=True)
class ContextPreparation:
    compact_event: CompactEvent | None = None
    cleanup_failed: bool = False


class ContextManager:
    def __init__(
        self,
        manager: ConversationManager,
        client: ModelClient,
        runtime: RuntimeStore,
        config: ContextConfig,
        estimator: TokenEstimator,
        recovery: RecoveryState,
        base_system_prompt: str,
        max_output_tokens: int,
    ) -> None:
        self.manager = manager
        self.client = client
        self.runtime = runtime
        self.config = config
        self.estimator = estimator
        self.recovery = recovery
        self.base_system_prompt = base_system_prompt
        self.max_output_tokens = max_output_tokens
        self.has_compacted = False

    @property
    def recovery_prompt(self) -> str:
        return build_recovery_prompt(self.recovery.entries()) if self.has_compacted else ""

    def observe_tool_result(self, call: ToolCallBlock, arguments: BaseModel, raw: object) -> None:
        if call.name != "read_file" or bool(getattr(raw, "is_error", True)):
            return
        try:
            path = PathPolicy(self.runtime.root.parent.parent).canonicalize(
                str(arguments.model_dump()["path"])
            )
            relative = path.relative_to(self.runtime.root.parent.parent).as_posix()
            self.recovery.record(relative, str(getattr(raw, "content", "")))
        except (KeyError, OSError, ValueError):
            return

    async def prepare_before_request(
        self,
        *,
        current_suffix: tuple[Message, ...],
        mcp_prompt: str,
        tools: tuple[ToolDefinition, ...],
    ) -> ContextPreparation:
        self._trim_stale_tool_results()
        before = self._estimate(
            self.manager.snapshot_committed(), current_suffix, mcp_prompt, tools
        )
        if before < self.config.compact_trigger_tokens:
            return ContextPreparation()
        return await self._compact(current_suffix, mcp_prompt, tools, before, manual=False)

    async def compact_now(
        self, *, mcp_prompt: str, tools: tuple[ToolDefinition, ...]
    ) -> ContextPreparation:
        self._trim_stale_tool_results()
        committed = self.manager.snapshot_committed()
        if len(self._split_turns(committed)) < 3:
            return ContextPreparation()
        before = self._estimate(committed, (), mcp_prompt, tools)
        return await self._compact((), mcp_prompt, tools, before, manual=True)

    def clear(self) -> bool:
        self.has_compacted = False
        self.recovery.clear()
        return self.runtime.clear_results()

    def _estimate(
        self,
        committed: tuple[Message, ...],
        suffix: tuple[Message, ...],
        mcp_prompt: str,
        tools: tuple[ToolDefinition, ...],
    ) -> int:
        system = "\n\n".join(
            piece.strip()
            for piece in (self.base_system_prompt, mcp_prompt, self.recovery_prompt)
            if piece.strip()
        )
        return self.estimator.estimate_request(system, committed + suffix, tools)

    def _split_turns(self, messages: tuple[Message, ...]) -> list[tuple[Message, ...]]:
        turns: list[list[Message]] = []
        for message in messages:
            starts = message.role is Role.USER and any(
                not isinstance(item, ToolResultBlock) for item in message.content
            )
            if starts:
                turns.append([message])
            elif turns:
                turns[-1].append(message)
            else:
                raise ContextCompactionError("conversation history has an invalid turn structure")
        return [tuple(turn) for turn in turns]

    def _trim_stale_tool_results(self) -> None:
        expected = self.manager.snapshot_committed()
        turns = self._split_turns(expected) if expected else []
        if len(turns) <= self.config.stale_after_turns:
            return
        old = turns[: -self.config.stale_after_turns]
        replacement: list[Message] = []
        changed = False
        for turn in turns:
            for message in turn:
                blocks = []
                for block in message.content:
                    if turn in old and isinstance(block, ToolResultBlock):
                        text = "\n".join(part.text for part in block.content)
                        clipped = clip_prefix(
                            text, self.config.stale_tool_result_tokens, self.estimator
                        )
                        clipped += "\n[Older tool result truncated for context budget.]"
                        if clipped != text:
                            changed = True
                        blocks.append(
                            ToolResultBlock(block.call_id, (TextBlock(clipped),), block.is_error)
                        )
                    else:
                        blocks.append(block)
                replacement.append(Message(message.role, tuple(blocks)))
        if changed:
            self.manager.replace_committed_history(expected, tuple(replacement))

    async def _compact(
        self,
        suffix: tuple[Message, ...],
        mcp_prompt: str,
        tools: tuple[ToolDefinition, ...],
        before: int,
        manual: bool,
    ) -> ContextPreparation:
        expected = self.manager.snapshot_committed()
        turns = self._split_turns(expected)
        if len(turns) < 3:
            if manual:
                return ContextPreparation()
            raise ContextCompactionError(
                "the context is too large but has no safely compactable history"
            )
        first, middle, last = turns[0], turns[1:-1], turns[-1]
        if not middle:
            return ContextPreparation()
        candidate = list(middle)
        last_error: ContextCompactionError | None = None
        for attempt in range(3):
            summary_input = tuple(message for turn in (first, *candidate, last) for message in turn)
            try:
                summary = await self._request_summary(summary_input)
                summary_message = Message(
                    Role.USER, (TextBlock(f"[Conversation summary]\n{summary}"),)
                )
                boundary = Message(Role.ASSISTANT, (TextBlock(COMPACT_BOUNDARY_MESSAGE),))
                replacement = tuple((*first, summary_message, boundary, *last))
                was_compacted = self.has_compacted
                self.has_compacted = True
                after = self._estimate(replacement, suffix, mcp_prompt, tools)
                if (manual and after >= before) or (
                    not manual and after >= self.config.compact_trigger_tokens
                ):
                    self.has_compacted = was_compacted
                    if manual:
                        raise ContextCompactionError(
                            "the summary did not reduce the estimated context size "
                            f"({before} -> {after} tokens)"
                        )
                    raise ContextCompactionError(
                        "the compacted context estimate remained at "
                        f"{after} tokens, not below the "
                        f"{self.config.compact_trigger_tokens}-token trigger"
                    )
                try:
                    self.manager.replace_committed_history(expected, replacement)
                except Exception as exc:
                    self.has_compacted = was_compacted
                    raise ContextCompactionError(
                        "the conversation history could not be atomically replaced"
                    ) from exc
                old_ids = self._tool_result_ids(expected)
                kept_ids = self._tool_result_ids(replacement)
                cleanup_failed = bool(self.runtime.delete_results(old_ids - kept_ids))
                return ContextPreparation(
                    CompactEvent(before, after, max(0, before - after)), cleanup_failed
                )
            except ContextCompactionError as exc:
                last_error = exc
                if attempt == 2:
                    break
                drop = max(1, math.ceil(len(candidate) * 0.2))
                candidate = candidate[drop:]
        if last_error is None:
            raise ContextCompactionError("compaction ended without a result")
        raise ContextCompactionError(
            f"three compaction attempts failed; last reason: {last_error.reason}"
        ) from last_error

    async def _request_summary(self, messages: tuple[Message, ...]) -> str:
        operation_id = f"compact-{uuid4()}"
        request_id = str(uuid4())
        request = ModelRequest(
            request_id,
            operation_id,
            SUMMARY_PROMPT,
            messages,
            (),
            min(self.config.summary_target_tokens, self.max_output_tokens),
        )
        completed: ResponseCompleted | None = None
        try:
            async for event in self.client.stream(request):
                if event.request_id != request_id:
                    raise ContextCompactionError(
                        "the summary stream returned an unexpected request id"
                    )
                if isinstance(event, UsageUpdated):
                    self.manager.record_maintenance_usage(operation_id, request_id, event.usage)
                elif isinstance(event, ResponseCompleted):
                    if completed is not None:
                        raise ContextCompactionError(
                            "the summary stream returned more than one completion event"
                        )
                    completed = event
                    self.manager.record_maintenance_usage(
                        operation_id, request_id, event.response.usage
                    )
        except ContextCompactionError:
            raise
        except HammerCodeError as exc:
            raise ContextCompactionError(f"the summary request failed: {exc}") from exc
        except Exception as exc:
            raise ContextCompactionError("the summary request failed unexpectedly") from exc
        if completed is None:
            raise ContextCompactionError("the summary stream ended without a completion event")
        if completed.response.stop_reason is StopReason.TOOL_CALL:
            raise ContextCompactionError("the summary model attempted to call a tool")
        if any(isinstance(block, ToolCallBlock) for block in completed.response.message.content):
            raise ContextCompactionError("the summary response contained an unexpected tool call")
        text = "".join(
            block.text
            for block in completed.response.message.content
            if isinstance(block, TextBlock)
        )
        try:
            summary = self._extract_summary(text)
        except ContextCompactionError as exc:
            if completed.response.stop_reason is StopReason.MAX_TOKENS:
                raise ContextCompactionError(
                    "the summary response reached the model output limit before producing "
                    "a complete summary envelope"
                ) from exc
            raise
        if self.estimator.estimate_text(summary) > self.config.summary_target_tokens:
            raise ContextCompactionError(
                "the summary exceeded its local target "
                f"({self.estimator.estimate_text(summary)} > "
                f"{self.config.summary_target_tokens} estimated tokens)"
            )
        return summary

    @staticmethod
    def _extract_summary(text: str) -> str:
        match = _SUMMARY.fullmatch(text)
        if match is None or not match.group(1).strip() or not match.group(2).strip():
            raise ContextCompactionError(
                "the model response did not contain one valid, non-empty "
                "<analysis>...</analysis><summary>...</summary> envelope"
            )
        return match.group(2).strip()

    @staticmethod
    def _tool_result_ids(messages: tuple[Message, ...]) -> set[str]:
        return {
            block.call_id
            for message in messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }

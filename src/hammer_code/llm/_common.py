"""Private SDK-edge helpers shared without sharing protocol state machines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypeVar

from hammer_code.domain.messages import (
    ContentBlock,
    Message,
    ProviderStateBlock,
    ReasoningBlock,
    Role,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelClientError,
    PermissionDeniedError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTimeoutError,
    StreamProtocolError,
    TransportError,
)

T = TypeVar("T")


def get(value: object, name: str, default: T | None = None) -> T | None:
    if isinstance(value, Mapping):
        return value.get(name, default)  # type: ignore[return-value]
    return getattr(value, name, default)


def make_tool_call(call_id: str | None, name: str | None, raw: str | None) -> ToolCallBlock:
    if not call_id or not name or raw is None:
        raise StreamProtocolError("Provider sent an incomplete tool call")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StreamProtocolError("Provider sent invalid tool call JSON") from exc
    if not isinstance(parsed, dict):
        raise StreamProtocolError("Tool call JSON must be an object")
    return ToolCallBlock(call_id, name, parsed, raw)


def text_from_message(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def normalize_openai_usage(raw: object | None, status: UsageStatus) -> TokenUsage:
    if raw is None:
        return TokenUsage(None, None, status=UsageStatus.UNAVAILABLE)
    details = get(raw, "input_tokens_details")
    output_details = get(raw, "output_tokens_details")
    return TokenUsage(
        get(raw, "input_tokens", get(raw, "prompt_tokens")),
        get(raw, "output_tokens", get(raw, "completion_tokens")),
        get(details, "cached_tokens", get(get(raw, "prompt_tokens_details"), "cached_tokens", 0))
        or 0,
        0,
        get(
            output_details,
            "reasoning_tokens",
            get(get(raw, "completion_tokens_details"), "reasoning_tokens", 0),
        )
        or 0,
        status,
    )


def map_exception(exc: Exception) -> ModelClientError:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status == 401 or "authentication" in name:
        return AuthenticationError("Model provider rejected authentication")
    if status == 403 or "permission" in name:
        return PermissionDeniedError("Model provider denied permission")
    if status == 429 or "ratelimit" in name:
        return RateLimitError("Model provider rate limit reached")
    if status == 400 or "badrequest" in name:
        return InvalidRequestError("Model provider rejected the request")
    if "timeout" in name:
        return RequestTimeoutError("Model request timed out")
    if status is not None and status >= 500:
        return ProviderUnavailableError("Model provider is temporarily unavailable")
    return TransportError("Model transport failed")


def as_openai_chat_messages(messages: tuple[Message, ...]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        blocks: list[dict[str, object]] = []
        tool_calls: list[dict[str, object]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCallBlock):
                tool_calls.append(
                    {
                        "id": block.call_id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": block.raw_arguments},
                    }
                )
            elif isinstance(block, ToolResultBlock):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.call_id,
                        "content": text_from_tool_result(block),
                    }
                )
        if tool_calls:
            result.append(
                {"role": "assistant", "content": blocks or None, "tool_calls": tool_calls}
            )
        elif blocks:
            result.append(
                {
                    "role": message.role.value,
                    "content": blocks if len(blocks) > 1 else blocks[0]["text"],
                }
            )
    return result


def text_from_tool_result(block: ToolResultBlock) -> str:
    return "".join(item.text for item in block.content)


def as_anthropic_messages(messages: tuple[Message, ...]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        blocks: list[dict[str, object]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ReasoningBlock):
                blocks.append({"type": "thinking", "thinking": block.text})
            elif isinstance(block, ProviderStateBlock) and block.protocol == "anthropic_messages":
                blocks.append(dict(block.data))
            elif isinstance(block, ToolCallBlock):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.call_id,
                        "name": block.name,
                        "input": dict(block.arguments),
                    }
                )
            elif isinstance(block, ToolResultBlock):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.call_id,
                        "content": text_from_tool_result(block),
                        "is_error": block.is_error,
                    }
                )
        if blocks:
            if result and result[-1]["role"] == message.role.value:
                existing = result[-1]["content"]
                if not isinstance(existing, list):
                    raise ValueError("Anthropic message content must be a list")
                result[-1]["content"] = existing + blocks
            else:
                result.append({"role": message.role.value, "content": blocks})
    return result


def blocks_from_parts(parts: Sequence[ContentBlock]) -> Message:
    if not parts:
        raise StreamProtocolError("Provider completed without assistant content")
    return Message(Role.ASSISTANT, tuple(parts))

"""Internal request, response and stream-event contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from hammer_code.domain.messages import Message, ReasoningVisibility, ToolCallBlock
from hammer_code.domain.usage import TokenUsage


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_CALL = "tool_call"
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class CompactEvent:
    before_tokens: int
    after_tokens: int
    saved_tokens: int

    def __post_init__(self) -> None:
        if min(self.before_tokens, self.after_tokens, self.saved_tokens) < 0:
            raise ValueError("Context estimates cannot be negative")


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    turn_id: str
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    max_output_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    provider_response_id: str
    model: str
    message: Message
    stop_reason: StopReason
    usage: TokenUsage


@dataclass(frozen=True)
class ResponseStarted:
    request_id: str
    provider_response_id: str
    model: str


@dataclass(frozen=True)
class TextDelta:
    request_id: str
    block_index: int
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    request_id: str
    block_index: int
    text: str
    visibility: ReasoningVisibility


@dataclass(frozen=True)
class ToolCallCompleted:
    request_id: str
    block_index: int
    tool_call: ToolCallBlock


@dataclass(frozen=True)
class UsageUpdated:
    request_id: str
    usage: TokenUsage


@dataclass(frozen=True)
class ResponseCompleted:
    request_id: str
    response: ModelResponse


ModelEvent: TypeAlias = (
    ResponseStarted
    | TextDelta
    | ReasoningDelta
    | ToolCallCompleted
    | UsageUpdated
    | ResponseCompleted
)

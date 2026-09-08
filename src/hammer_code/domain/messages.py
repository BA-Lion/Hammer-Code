"""Immutable, SDK-free message and content-block types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ReasoningVisibility(StrEnum):
    SUMMARY = "summary"
    VISIBLE = "visible"


@dataclass(frozen=True)
class TextBlock:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("TextBlock text must not be empty")


@dataclass(frozen=True)
class ReasoningBlock:
    text: str
    visibility: ReasoningVisibility

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("ReasoningBlock text must not be empty")


@dataclass(frozen=True, repr=False)
class ProviderStateBlock:
    protocol: str
    kind: str
    data: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.protocol or not self.kind:
            raise ValueError("Provider state needs protocol and kind")
        object.__setattr__(self, "data", dict(self.data))

    def __repr__(self) -> str:
        return (
            f"ProviderStateBlock(protocol={self.protocol!r}, kind={self.kind!r}, data=<redacted>)"
        )


@dataclass(frozen=True)
class RefusalBlock:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("Refusal reason must not be empty")


@dataclass(frozen=True)
class ToolCallBlock:
    call_id: str
    name: str
    arguments: Mapping[str, JSONValue]
    raw_arguments: str

    def __post_init__(self) -> None:
        if not self.call_id or not self.name or not self.raw_arguments:
            raise ValueError("Tool call needs id, name and raw arguments")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolResultBlock:
    call_id: str
    content: tuple[TextBlock, ...]
    is_error: bool

    def __post_init__(self) -> None:
        if not self.call_id or not self.content:
            raise ValueError("Tool result needs id and text content")


ContentBlock: TypeAlias = (
    TextBlock | ReasoningBlock | ProviderStateBlock | RefusalBlock | ToolCallBlock | ToolResultBlock
)


@dataclass(frozen=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...]

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Message must contain at least one content block")

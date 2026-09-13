"""Strict serializable data models for project-local sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hammer_code.domain.messages import Message


class RecordType(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    COMPRESSION = "compression"


class SessionRecord(BaseModel):
    """One JSONL record. Content is validated by the serializer as tagged blocks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: RecordType
    content: list[dict[str, object]]
    timestamp: datetime
    turn_index: int = Field(gt=0)
    tool_use_id: str | None = None
    is_error: bool = False

    @field_validator("timestamp")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Session timestamps must include a timezone")
        return value


class SessionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = ""
    protocol: str = ""
    message_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    created_at: datetime
    last_active: datetime
    memory_cursor: int = Field(default=0, ge=0)

    @field_validator("created_at", "last_active")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Session timestamps must include a timezone")
        return value


@dataclass(frozen=True)
class SessionSummary:
    id: str
    title: str
    protocol: str
    last_active: datetime


@dataclass(frozen=True)
class RestoreResult:
    messages: tuple[Message, ...]
    records: tuple[SessionRecord, ...]
    latest_durable_turn: int
    completed_turns: frozenset[int]
    safe_byte_boundary: int
    repaired: bool = False
    degraded: bool = False

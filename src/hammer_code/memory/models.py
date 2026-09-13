"""Strict model-owned memory operation contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

_ID = Annotated[str, StringConstraints(pattern=r"[a-z0-9][a-z0-9-]{0,63}")]


class MemoryCategory(StrEnum):
    USER_PREFERENCES = "user-preferences"
    USER_EXPERIENCE = "user-experience"
    PROJECT_KNOWLEDGE = "project-knowledge"
    REFERENCES = "references"


class _Operation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    category: MemoryCategory


class AddOperation(_Operation):
    action: Literal["add"]
    memory_id: _ID
    path: str
    content: str = Field(min_length=1)
    index_description: str


class MergeOperation(_Operation):
    action: Literal["merge"]
    memory_id: _ID | None = None
    path: str
    content: list[str] = Field(min_length=1)
    index_description: str | None = None


class OverrideOperation(_Operation):
    action: Literal["override"]
    memory_id: _ID | None = None
    path: str
    content: str = Field(min_length=1)
    index_description: str


class SkipOperation(_Operation):
    action: Literal["skip"]
    reason: str = Field(min_length=1)


MemoryOperation = Annotated[
    AddOperation | MergeOperation | OverrideOperation | SkipOperation, Field(discriminator="action")
]


class MemoryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operations: list[MemoryOperation]


class IndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    memory_id: _ID
    path: str
    description: str


class MemoryIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    entries: tuple[IndexEntry, ...] = ()


def validate_path(path: str) -> str:
    if (
        not path.endswith(".md")
        or path == "index.md"
        or "\\" in path
        or "\x00" in path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Memory path must be a normalized relative .md file")
    return path


def validate_description(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 200 or any(char in cleaned for char in "\r\n\x00"):
        raise ValueError("Memory index description must be one non-empty line up to 200 characters")
    return cleaned


def normalize_document(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("Memory document content must be non-empty and NUL-free")
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def validate_merge_lines(value: list[str]) -> list[str]:
    result: list[str] = []
    for item in value:
        normalized = item.rstrip()
        if (
            not normalized.startswith("- ")
            or normalized == "- "
            or any(c in item for c in "\r\n\x00")
        ):
            raise ValueError("Memory merge content must contain independent one-line '- ' entries")
        result.append(normalized)
    return result

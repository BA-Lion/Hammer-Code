"""Immutable, SDK-independent contracts for local Skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from hammer_code.domain.messages import Message

if TYPE_CHECKING:
    from hammer_code.skill.retrieval import Bm25Index


class SkillScope(StrEnum):
    PROJECT = "project"
    USER = "user"


class SkillSource(StrEnum):
    GENERATED = "generated"
    MANUAL = "manual"
    IMPORTED = "imported"


class SkillContext(StrEnum):
    INLINE = "inline"
    FORK = "fork"


@dataclass(frozen=True, order=True)
class SkillRef:
    scope: SkillScope
    name: str
    version: str


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    version: str
    created_at: datetime
    updated_at: datetime
    source: SkillSource
    prompt_template: str
    scope: SkillScope
    skill_dir: Path
    user_invocable: bool = True
    model_invocable: bool = True
    context: SkillContext = SkillContext.INLINE
    when_to_use: str | None = None
    arguments_required: bool = False
    argument_hint: str | None = None
    allowed_tools: tuple[str, ...] | None = None

    @property
    def ref(self) -> SkillRef:
        return SkillRef(self.scope, self.name, self.version)


@dataclass(frozen=True)
class CatalogSnapshot:
    generation: int
    fingerprint: str
    effective: Mapping[str, SkillDefinition]
    by_identity: Mapping[tuple[SkillScope, str], SkillDefinition]
    index: Bm25Index | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective", MappingProxyType(dict(self.effective)))
        object.__setattr__(self, "by_identity", MappingProxyType(dict(self.by_identity)))


@dataclass(frozen=True)
class RetrievedSkill:
    ref: SkillRef
    raw_score: float
    relative_score: float
    coverage: float


@dataclass(frozen=True)
class SkillObservation:
    snapshot_generation: int
    retrieved: tuple[RetrievedSkill, ...] = ()
    used: tuple[SkillRef, ...] = ()
    completed: bool = False


@dataclass(frozen=True)
class UsageEntry:
    source: SkillSource | None = None
    version: str | None = None
    retrieve: int = 0
    relevant: int = 0
    used: int = 0
    version_retrieve: int = 0
    version_relevant: int = 0
    version_used: int = 0
    last_retrieved_at: str | None = None
    last_relevant_at: str | None = None
    last_used_at: str | None = None
    latest_reason: str | None = None


@dataclass(frozen=True)
class EvolutionQueueItem:
    messages: tuple[Message, ...]
    current_input: str
    previous_observation: SkillObservation | None
    forced_ref: SkillRef | None = None
    feedback: str | None = None


class SkillEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: SkillScope
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*")
    version: str = Field(pattern=r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
    relevant: bool
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def _bounded_reason(cls, value: str) -> str:
        return _one_line(value, "reason", 500)


class ExtractedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*")
    description: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=32000)
    when_to_use: str | None = Field(default=None, max_length=1000)
    user_invocable: bool = True
    model_invocable: bool = True
    context: SkillContext = SkillContext.INLINE
    arguments_required: bool = False
    argument_hint: str | None = Field(default=None, max_length=500)
    allowed_tools: tuple[str, ...] | None = None
    evolution_note: str = Field(min_length=1, max_length=1000)

    @field_validator("description", "when_to_use", "argument_hint", "evolution_note")
    @classmethod
    def _single_line_fields(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        assert info.field_name is not None
        limits = {
            "description": 500,
            "when_to_use": 1000,
            "argument_hint": 500,
            "evolution_note": 1000,
        }
        return _one_line(value, getattr(info, "field_name", "field"), limits[info.field_name])


class ExtractorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluations: tuple[SkillEvaluation, ...] = ()
    candidate_action: str
    candidate: ExtractedCandidate | None = None
    discard_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _candidate_shape(self) -> ExtractorResponse:
        if self.candidate_action not in {"candidate", "discard"}:
            raise ValueError("candidate_action must be candidate or discard")
        if self.candidate_action == "candidate" and (
            self.candidate is None or self.discard_reason is not None
        ):
            raise ValueError("candidate action requires candidate and no discard_reason")
        if self.candidate_action == "discard" and (
            self.candidate is not None or self.discard_reason is None
        ):
            raise ValueError("discard action requires discard_reason and no candidate")
        return self

    @field_validator("discard_reason")
    @classmethod
    def _discard_reason(cls, value: str | None) -> str | None:
        return None if value is None else _one_line(value, "discard_reason", 500)


class MaintenanceTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: SkillScope
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*")


class MaintenanceSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*")
    description: str = Field(min_length=1, max_length=500)
    when_to_use: str | None = Field(default=None, max_length=1000)
    user_invocable: bool = True
    model_invocable: bool = True
    context: SkillContext = SkillContext.INLINE
    arguments_required: bool = False
    argument_hint: str | None = Field(default=None, max_length=500)
    allowed_tools: tuple[str, ...] | None = None
    body: str = Field(min_length=1, max_length=32000)

    @field_validator("description", "when_to_use", "argument_hint")
    @classmethod
    def _single_line_fields(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        assert info.field_name is not None
        limits = {"description": 500, "when_to_use": 1000, "argument_hint": 500}
        return _one_line(value, getattr(info, "field_name", "field"), limits[info.field_name])


class MaintenanceOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: str
    target: MaintenanceTarget | None = None
    skill: MaintenanceSkill | None = None
    evolution_note: str | None = Field(default=None, min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _action_shape(self) -> MaintenanceOperation:
        if self.action not in {"add", "merge", "discard"}:
            raise ValueError("action must be add, merge, or discard")
        if self.action == "discard" and any(
            value is not None for value in (self.target, self.skill, self.evolution_note)
        ):
            raise ValueError("discard must not contain target, skill, or evolution_note")
        if self.action == "add" and (
            self.target is not None or self.skill is None or self.evolution_note is None
        ):
            raise ValueError("add requires skill and evolution_note, but no target")
        if self.action == "merge" and (
            self.target is None or self.skill is None or self.evolution_note is None
        ):
            raise ValueError("merge requires target, skill, and evolution_note")
        return self

    @field_validator("reason", "evolution_note")
    @classmethod
    def _bounded_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _one_line(
            value,
            getattr(info, "field_name", "field"),
            1000 if info.field_name == "evolution_note" else 500,
        )


def _one_line(value: str, field: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > limit or any(char in value for char in "\r\n\x00"):
        raise ValueError(f"{field} must be a bounded single line")
    return normalized


def empty_snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(0, "", {}, {}, None)

"""Immutable contracts shared by Subagent catalog and runtime code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from hammer_code.domain.events import ToolDefinition
from hammer_code.domain.messages import Message


class SubagentSource(StrEnum):
    PREDEFINED = "predefined"
    DYNAMIC = "dynamic"


class SubagentContext(StrEnum):
    ISOLATED = "isolated"
    FORK = "fork"


class SubagentExecution(StrEnum):
    INLINE = "inline"
    BACKGROUND = "background"


class BackgroundTaskStatus(StrEnum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SubagentDefinition:
    name: str
    description: str
    when_to_use: str | None
    prompt: str
    context: SubagentContext
    execution: SubagentExecution
    allowed_tools: tuple[str, ...] | None
    disallowed_tools: tuple[str, ...]
    max_iterations: int
    source_path: Path


@dataclass(frozen=True)
class SubagentCatalogSnapshot:
    generation: int
    fingerprint: str
    definitions: Mapping[str, SubagentDefinition]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))


def empty_catalog_snapshot() -> SubagentCatalogSnapshot:
    return SubagentCatalogSnapshot(generation=0, fingerprint="", definitions={})


@dataclass(frozen=True)
class ParentRequestSnapshot:
    """Frozen main-request input used by one already-created child invocation."""

    catalog: SubagentCatalogSnapshot
    system_prompt: str
    messages: tuple[Message, ...]
    tool_definitions: tuple[ToolDefinition, ...]
    tool_names: frozenset[str]
    base_system_prompt: str
    project_instructions: str
    mcp_prompt: str


@dataclass(frozen=True)
class SubagentInvocation:
    source: SubagentSource
    name: str
    task: str
    prompt: str
    context: SubagentContext
    execution: SubagentExecution
    allowed_tools: tuple[str, ...] | None
    disallowed_tools: tuple[str, ...]
    max_iterations: int
    parent: ParentRequestSnapshot

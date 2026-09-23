"""Immutable long-lived AgentTeam definition contracts.

Templates are deliberately distinct from the per-run runtime types in
``runtime.py``.  In particular, no template contains a mailbox, conversation,
worktree, or model object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from hammer_code.subagent.models import (
    SubagentContext,
    SubagentExecution,
    SubagentWorkspace,
)


@dataclass(frozen=True)
class TeamAgentTemplate:
    name: str
    description: str
    when_to_use: str | None
    prompt: str
    context: SubagentContext
    workspace: SubagentWorkspace
    allowed_tools: tuple[str, ...] | None
    disallowed_tools: tuple[str, ...]
    source_path: Path | None = None


@dataclass(frozen=True)
class AgentTeamDefinition:
    name: str
    description: str
    leader: TeamAgentTemplate
    execution: SubagentExecution
    members: tuple[str, ...]
    source_path: Path | None = None

    def catalog_view(self) -> dict[str, object]:
        """The only durable definition shape exposed to the Primary model."""
        return {
            "name": self.name,
            "description": self.description,
            "leader": {
                "name": self.leader.name,
                "description": self.leader.description,
                "when_to_use": self.leader.when_to_use,
                "context": self.leader.context.value,
                "execution": self.execution.value,
                "workspace": self.leader.workspace.value,
                "allowed_tools": self.leader.allowed_tools,
                "disallowed_tools": self.leader.disallowed_tools,
                "prompt": self.leader.prompt,
            },
        }


@dataclass(frozen=True)
class AgentTeamCatalogSnapshot:
    generation: int
    fingerprint: str
    definitions: Mapping[str, AgentTeamDefinition]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))

    def catalog_views(self) -> tuple[dict[str, object], ...]:
        return tuple(item.catalog_view() for item in self.definitions.values())


def empty_catalog_snapshot() -> AgentTeamCatalogSnapshot:
    return AgentTeamCatalogSnapshot(generation=0, fingerprint="", definitions={})

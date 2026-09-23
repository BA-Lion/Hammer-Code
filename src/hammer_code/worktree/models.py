"""Immutable public facts for the process-local worktree manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from hammer_code.subagent.models import SubagentWorkspace


class WorktreeLeaseState(StrEnum):
    ACTIVE = "active"
    PENDING_INTEGRATION = "pending_integration"
    CONFLICT = "conflict"
    CLOSED = "closed"


@dataclass(frozen=True)
class WorktreeLease:
    task_id: str
    path: Path
    baseline_commit: str
    relative_cwd: Path
    created_at: datetime
    state: WorktreeLeaseState
    parent_task_id: str | None = None


@dataclass(frozen=True)
class WorktreeResultMetadata:
    task_id: str
    workspace: SubagentWorkspace
    change_state: str
    baseline_commit: str | None
    result_commit: str | None
    changed_files: tuple[str, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class WorktreeInspection:
    task_id: str
    state: WorktreeLeaseState
    baseline_commit: str
    result_commit: str | None
    changed_files: tuple[str, ...]
    truncated: bool
    inspection_id: str | None = None
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorktreeResolution:
    task_id: str
    change_state: str
    main_workspace_changed: bool
    files: tuple[str, ...]
    next_action: str
    error_code: str | None = None

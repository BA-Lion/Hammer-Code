"""In-process AgentTeam task board and durable-but-ephemeral run metadata."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from hammer_code.agent_team.models import AgentTeamDefinition, TeamAgentTemplate
from hammer_code.agent_team.usage import TeamUsageTracker
from hammer_code.subagent.models import SubagentExecution, SubagentWorkspace


class TeamTaskState(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssignmentPhase(StrEnum):
    RUNNING_LLM = "RUNNING_LLM"
    RUNNING_TOOL = "RUNNING_TOOL"
    IDLE_WAITING = "IDLE_WAITING"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class TeamTask:
    id: str
    title: str
    description: str
    dependencies: tuple[str, ...]
    state: TeamTaskState
    current_assignment_id: str | None = None


@dataclass
class AssignmentRuntime:
    id: str
    template_id: str
    task_id: str
    is_leader: bool
    workspace: SubagentWorkspace
    phase: AssignmentPhase = AssignmentPhase.RUNNING_LLM
    terminal_reason: str | None = None
    lease_id: str | None = None
    mail_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True)
class AgentTeamRunSnapshot:
    id: str
    team_name: str
    accepting: bool
    terminal: str | None
    tasks: tuple[TeamTask, ...]
    assignments: tuple[AssignmentRuntime, ...]


class AgentTeamRuntimeStore:
    """Owns only validated temporary run directories; it never restores them."""

    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root.resolve() / ".hammer-code" / "runtime" / "agent-teams"

    def run_path(self, team_name: str, run_id: str) -> Path:
        return self.root / team_name / run_id

    def cleanup_orphans(self) -> None:
        if not self.root.exists():
            return
        if not self.root.is_dir() or self.root.is_symlink():
            raise RuntimeError("AgentTeam runtime root is unsafe")
        for team in self.root.iterdir():
            if not team.is_dir() or team.is_symlink():
                raise RuntimeError("AgentTeam runtime entry is unsafe")
            for run in team.iterdir():
                try:
                    UUID(run.name)
                except ValueError as exc:
                    raise RuntimeError(
                        "AgentTeam runtime contains an invalid run directory"
                    ) from exc
                resolved = run.resolve()
                if not run.is_dir() or run.is_symlink() or self.root not in resolved.parents:
                    raise RuntimeError("AgentTeam runtime path is unsafe")
                shutil.rmtree(_native_path(resolved))


class AgentTeamRun:
    """Single-process task board.  All mutations share one ordering lock."""

    MAX_TASKS = 256
    MAX_ASSIGNMENTS = 512

    def __init__(
        self,
        definition: AgentTeamDefinition,
        task: str,
        runtime_store: AgentTeamRuntimeStore,
        max_active_assignments: int,
        tokens: int,
        execution: SubagentExecution,
    ) -> None:
        self.id = str(uuid4())
        self.definition = definition
        self.execution = execution
        self.runtime_store = runtime_store
        self.path = runtime_store.run_path(definition.name, self.id)
        self.max_active_assignments = max_active_assignments
        self.usage = TeamUsageTracker(tokens)
        self.lock = asyncio.Lock()
        self.accepting = True
        self.terminal: str | None = None
        self.summary: str | None = None
        self.tasks: dict[str, TeamTask] = {}
        self.assignments: dict[str, AssignmentRuntime] = {}
        self.root_task_id = str(uuid4())
        self.leader_assignment_id = str(uuid4())
        root_task = TeamTask(
            self.root_task_id,
            "Root task",
            task[:8000],
            (),
            TeamTaskState.IN_PROGRESS,
            self.leader_assignment_id,
        )
        self.tasks[root_task.id] = root_task
        self.assignments[self.leader_assignment_id] = AssignmentRuntime(
            self.leader_assignment_id,
            definition.leader.name,
            root_task.id,
            True,
            definition.leader.workspace,
        )

    async def initialize(self) -> None:
        async with self.lock:
            self.path.mkdir(parents=True, exist_ok=False)
            self._write_json(
                self.path / "run.json",
                {
                    "id": self.id,
                    "team": self.definition.name,
                    "execution": self.execution.value,
                    "root_task_id": self.root_task_id,
                    "leader_assignment_id": self.leader_assignment_id,
                    "created_at": _utc(),
                },
            )
            self._write_task(self.tasks[self.root_task_id])
            self._write_assignment(self.assignments[self.leader_assignment_id])
            (self.path / "assignments" / self.leader_assignment_id / "mailbox").mkdir(
                parents=True, exist_ok=False
            )

    async def create_task(
        self, title: str, description: str, dependencies: tuple[str, ...] = ()
    ) -> TeamTask:
        if (
            not title.strip()
            or len(title) > 500
            or not description.strip()
            or len(description) > 8000
        ):
            raise ValueError("task title or description is invalid")
        async with self.lock:
            if not self.accepting or len(self.tasks) >= self.MAX_TASKS:
                raise ValueError("TeamRun cannot accept another task")
            if any(item not in self.tasks for item in dependencies) or len(
                set(dependencies)
            ) != len(dependencies):
                raise ValueError("task dependencies are invalid")
            value = TeamTask(
                str(uuid4()),
                title.strip(),
                description.strip(),
                dependencies,
                TeamTaskState.PENDING,
            )
            self.tasks[value.id] = value
            self._write_task(value)
            return value

    async def update_task(
        self, task_id: str, *, state: TeamTaskState | None = None, assignment_id: str | None = None
    ) -> TeamTask:
        async with self.lock:
            task = self._task(task_id)
            if assignment_id is not None and assignment_id not in self.assignments:
                raise ValueError("task assignment is unknown")
            updated = TeamTask(
                task.id,
                task.title,
                task.description,
                task.dependencies,
                state or task.state,
                assignment_id if assignment_id is not None else task.current_assignment_id,
            )
            self.tasks[task_id] = updated
            self._write_task(updated)
            return updated

    async def create_assignment(
        self, template: TeamAgentTemplate, task_id: str
    ) -> AssignmentRuntime:
        async with self.lock:
            task = self._task(task_id)
            active = (
                sum(
                    item.phase is not AssignmentPhase.FINISHED for item in self.assignments.values()
                )
                - 1
            )
            if (
                not self.accepting
                or active >= self.max_active_assignments
                or len(self.assignments) >= self.MAX_ASSIGNMENTS
            ):
                raise ValueError("Team assignment capacity is full")
            if task.current_assignment_id is not None:
                raise ValueError("task already has a current assignment")
            value = AssignmentRuntime(
                str(uuid4()), template.name, task.id, False, template.workspace
            )
            self.assignments[value.id] = value
            self.tasks[task.id] = TeamTask(
                task.id,
                task.title,
                task.description,
                task.dependencies,
                TeamTaskState.IN_PROGRESS,
                value.id,
            )
            self._write_assignment(value)
            self._write_task(self.tasks[task.id])
            (self.path / "assignments" / value.id / "mailbox").mkdir(parents=True, exist_ok=False)
            return value

    async def set_phase(self, assignment_id: str, phase: AssignmentPhase) -> None:
        async with self.lock:
            item = self._assignment(assignment_id)
            if item.phase is AssignmentPhase.FINISHED:
                return
            item.phase = phase
            self._write_assignment(item)

    async def finish_assignment(
        self, assignment_id: str, reason: str, summary: str = ""
    ) -> AssignmentRuntime:
        async with self.lock:
            item = self._assignment(assignment_id)
            if item.phase is AssignmentPhase.FINISHED:
                return item
            item.phase = AssignmentPhase.FINISHED
            item.terminal_reason = reason[:500]
            item.mail_event.set()
            self._write_assignment(item)
            mailbox = self.path / "assignments" / assignment_id / "mailbox"
            if mailbox.exists():
                shutil.rmtree(_native_path(mailbox))
            if not item.is_leader:
                try:
                    self._send_locked(
                        assignment_id,
                        self.leader_assignment_id,
                        json.dumps(
                            {
                                "type": "assignment_finished",
                                "assignment_id": assignment_id,
                                "reason": item.terminal_reason,
                                "summary": summary[:8000],
                            },
                            ensure_ascii=False,
                        ),
                    )
                except ValueError:
                    self.terminal = "failed"
            return item

    async def finish_team(self, status: str, summary: str) -> None:
        if (
            status not in {"completed", "failed", "cancelled"}
            or not summary.strip()
            or len(summary) > 8000
        ):
            raise ValueError("team status or summary is invalid")
        async with self.lock:
            children = [
                item
                for item in self.assignments.values()
                if not item.is_leader and item.phase is not AssignmentPhase.FINISHED
            ]
            if children:
                raise ValueError("Team assignments are still active")
            if status == "completed" and any(
                item.state is not TeamTaskState.COMPLETED for item in self.tasks.values()
            ):
                raise ValueError("all Team tasks must be completed")
            self.accepting = False
            self.terminal = status
            self.summary = summary.strip()
            self._write_json(
                self.path / "run.json",
                {
                    "id": self.id,
                    "team": self.definition.name,
                    "terminal": status,
                    "summary": self.summary,
                    "ended_at": _utc(),
                },
            )

    async def snapshot(self) -> AgentTeamRunSnapshot:
        async with self.lock:
            return AgentTeamRunSnapshot(
                self.id,
                self.definition.name,
                self.accepting,
                self.terminal,
                tuple(self.tasks.values()),
                tuple(self.assignments.values()),
            )

    async def cleanup(self) -> None:
        async with self.lock:
            self.accepting = False
            for assignment in self.assignments.values():
                assignment.phase = AssignmentPhase.FINISHED
                assignment.mail_event.set()
            if self.path.exists():
                shutil.rmtree(_native_path(self.path))

    def _task(self, task_id: str) -> TeamTask:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise ValueError("Team task is unknown") from exc

    def _assignment(self, assignment_id: str) -> AssignmentRuntime:
        try:
            return self.assignments[assignment_id]
        except KeyError as exc:
            raise ValueError("Team assignment is unknown") from exc

    def _write_task(self, task: TeamTask) -> None:
        self._write_json(
            self.path / "tasks" / f"{task.id}.json",
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "dependencies": task.dependencies,
                "state": task.state.value,
                "current_assignment_id": task.current_assignment_id,
            },
        )

    def _write_assignment(self, item: AssignmentRuntime) -> None:
        self._write_json(
            self.path / "assignments" / item.id / "assignment.json",
            {
                "id": item.id,
                "template_id": item.template_id,
                "task_id": item.task_id,
                "is_leader": item.is_leader,
                "workspace": item.workspace.value,
                "phase": item.phase.value,
                "terminal_reason": item.terminal_reason,
            },
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the name short enough for Windows tmp_path test roots.
        temporary = path.with_name(f".{uuid4().hex}.tmp")
        with open(_native_path(temporary), "w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        os.replace(_native_path(temporary), _native_path(path))

    def _send_locked(self, sender_id: str, recipient_id: str, body: str) -> None:
        # Implemented by mailbox service at runtime; a narrow fallback keeps finish atomic.
        recipient = self._assignment(recipient_id)
        if recipient.phase is AssignmentPhase.FINISHED:
            raise ValueError("recipient has finished")
        from hammer_code.agent_team.mailbox import write_message

        write_message(self.path, sender_id, recipient_id, body)
        recipient.mail_event.set()


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _native_path(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" and not value.startswith("\\\\?\\") else value

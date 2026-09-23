"""Atomic persistent AgentTeam catalog with strict initial and LKG reloads."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from hammer_code.agent_team.models import (
    AgentTeamCatalogSnapshot,
    AgentTeamDefinition,
    TeamAgentTemplate,
    empty_catalog_snapshot,
)
from hammer_code.agent_team.parser import AgentTeamParseError, is_reparse, parse_member, parse_team
from hammer_code.errors import ConfigurationError
from hammer_code.subagent.models import SubagentExecution

_MAX_TEAMS = 64


class AgentTeamRepositoryError(ValueError):
    pass


class AgentTeamRepository:
    def __init__(self, workspace_root: Path, warning: Callable[[str], None] | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.root = self.workspace_root / ".hammer-code" / "agent-teams"
        self._warning = warning
        self._current = empty_catalog_snapshot()
        self._lock = asyncio.Lock()
        self._last_warning: str | None = None

    def _paths_and_fingerprint(self) -> tuple[tuple[Path, ...], str]:
        if not self.root.exists():
            return (), hashlib.sha256().hexdigest()
        if not self.root.is_dir() or is_reparse(self.root):
            raise AgentTeamRepositoryError("AgentTeam catalog root is unsafe")
        try:
            entries = tuple(sorted(self.root.iterdir(), key=lambda item: item.name))
        except OSError as exc:
            raise AgentTeamRepositoryError("AgentTeam catalog root could not be scanned") from exc
        if len(entries) > _MAX_TEAMS:
            raise AgentTeamRepositoryError("AgentTeam catalog exceeds 64 teams")
        configs: list[Path] = []
        digest = hashlib.sha256()
        for entry in entries:
            if not entry.is_dir() or is_reparse(entry):
                raise AgentTeamRepositoryError("AgentTeam catalog contains an unsafe entry")
            config = entry / "config.md"
            if not config.is_file() or is_reparse(config):
                raise AgentTeamRepositoryError("AgentTeam definition is missing config.md")
            try:
                resolved = config.resolve()
                if (
                    resolved.parent != entry.resolve()
                    or entry.resolve().parent != self.root.resolve()
                ):
                    raise AgentTeamRepositoryError("AgentTeam definition escapes catalog")
                digest.update(entry.name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(config.read_bytes())
                members = entry / "members"
                if members.exists():
                    if not members.is_dir() or is_reparse(members):
                        raise AgentTeamRepositoryError("AgentTeam members directory is unsafe")
                    for item in sorted(members.iterdir(), key=lambda child: child.name):
                        if not item.is_file() or item.suffix != ".md" or is_reparse(item):
                            raise AgentTeamRepositoryError("AgentTeam member is unsafe")
                        if item.resolve().parent != members.resolve():
                            raise AgentTeamRepositoryError("AgentTeam member escapes catalog")
                        digest.update(item.name.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(item.read_bytes())
            except OSError as exc:
                raise AgentTeamRepositoryError("AgentTeam definition could not be read") from exc
            configs.append(config)
        return tuple(configs), digest.hexdigest()

    def _build_snapshot(
        self, previous: AgentTeamCatalogSnapshot | None
    ) -> AgentTeamCatalogSnapshot:
        configs, fingerprint = self._paths_and_fingerprint()
        if previous is not None and previous.fingerprint == fingerprint:
            return previous
        definitions: dict[str, AgentTeamDefinition] = {}
        for config in configs:
            try:
                definition = parse_team(config)
                members_dir = config.parent / "members"
                found: dict[str, TeamAgentTemplate] = {}
                if members_dir.exists():
                    for member_path in sorted(members_dir.iterdir(), key=lambda item: item.name):
                        found[member_path.stem] = parse_member(member_path)
                if tuple(found) != definition.members or set(found) != set(definition.members):
                    raise AgentTeamRepositoryError("AgentTeam members do not match config.md")
            except AgentTeamParseError as exc:
                raise AgentTeamRepositoryError(
                    f"Invalid AgentTeam definition '{config.parent.name}': {exc}"
                ) from exc
            if definition.name in definitions:
                raise AgentTeamRepositoryError("Duplicate AgentTeam name")
            definitions[definition.name] = definition
        return AgentTeamCatalogSnapshot(
            (previous.generation + 1) if previous else 1, fingerprint, definitions
        )

    async def initialize(self) -> AgentTeamCatalogSnapshot:
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self._build_snapshot, None)
            except AgentTeamRepositoryError as exc:
                raise ConfigurationError("AgentTeam catalog configuration is invalid") from exc
            self._last_warning = None
            return self._current

    async def snapshot_for_request(self) -> AgentTeamCatalogSnapshot:
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self._build_snapshot, self._current)
                self._last_warning = None
            except AgentTeamRepositoryError as exc:
                if self._warning is not None and str(exc) != self._last_warning:
                    self._warning(f"AgentTeam catalog reload ignored: {exc}")
                    self._last_warning = str(exc)
            return self._current

    async def create_team(
        self,
        description: str,
        leader: TeamAgentTemplate,
        *,
        execution: SubagentExecution = SubagentExecution.INLINE,
    ) -> AgentTeamDefinition:
        """Persist a self-contained leader snapshot under an unambiguous name."""
        async with self._lock:
            return await asyncio.to_thread(self._create_team_sync, description, leader, execution)

    def _create_team_sync(
        self, description: str, leader: TeamAgentTemplate, execution: SubagentExecution
    ) -> AgentTeamDefinition:
        if self.root.exists() and (not self.root.is_dir() or is_reparse(self.root)):
            raise AgentTeamRepositoryError("AgentTeam catalog root is unsafe")
        self.root.mkdir(parents=True, exist_ok=True)
        existing = {item.name for item in self.root.iterdir()}
        base = leader.name
        name = base
        suffix = 2
        while name in existing:
            name = f"{base}-{suffix}"
            suffix += 1
        if len(name) > 64:
            raise AgentTeamRepositoryError("AgentTeam name is too long after suffix")
        temp = self.root / f".{name}-{uuid4().hex}.tmp"
        target = self.root / name
        try:
            temp.mkdir()
            (temp / "members").mkdir()
            (temp / "config.md").write_text(
                self._team_text(name, description, leader, execution, ()),
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temp, target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return parse_team(target / "config.md")

    async def create_member(
        self, team_name: str, template: TeamAgentTemplate
    ) -> AgentTeamDefinition:
        async with self._lock:
            return await asyncio.to_thread(self._create_member_sync, team_name, template)

    async def member(self, team_name: str, member_name: str) -> TeamAgentTemplate:
        """Read one current template without exposing its source path to a model."""
        async with self._lock:
            try:
                return await asyncio.to_thread(
                    parse_member, self.root / team_name / "members" / f"{member_name}.md"
                )
            except AgentTeamParseError as exc:
                raise ValueError("AgentTeam member template is unavailable") from exc

    def _create_member_sync(
        self, team_name: str, template: TeamAgentTemplate
    ) -> AgentTeamDefinition:
        config = self.root / team_name / "config.md"
        definition = parse_team(config)
        members_dir = config.parent / "members"
        members_dir.mkdir(exist_ok=True)
        target = members_dir / f"{template.name}.md"
        if target.exists() or template.name in definition.members:
            raise AgentTeamRepositoryError("AgentTeam member already exists")
        member_tmp = members_dir / f".{template.name}-{uuid4().hex}.tmp"
        config_tmp = config.parent / f".config-{uuid4().hex}.tmp"
        updated_members = (*definition.members, template.name)
        try:
            member_tmp.write_text(self._member_text(template), encoding="utf-8", newline="\n")
            os.replace(member_tmp, target)
            config_tmp.write_text(
                self._team_text(
                    definition.name,
                    definition.description,
                    definition.leader,
                    definition.execution,
                    updated_members,
                ),
                encoding="utf-8",
                newline="\n",
            )
            os.replace(config_tmp, config)
        except Exception:
            member_tmp.unlink(missing_ok=True)
            config_tmp.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        return parse_team(config)

    @staticmethod
    def _tools(values: tuple[str, ...] | None) -> str | None:
        return None if values is None else "[" + ", ".join(values) + "]"

    def _team_text(
        self,
        name: str,
        description: str,
        leader: TeamAgentTemplate,
        execution: SubagentExecution,
        members: tuple[str, ...],
    ) -> str:
        lines = [
            "---",
            f"name: {name}",
            f"description: {description}",
            f"leader-name: {leader.name}",
            f"leader-description: {leader.description}",
        ]
        if leader.when_to_use:
            lines.append(f"leader-when-to-use: {leader.when_to_use}")
        lines.extend(
            [
                f"leader-context: {leader.context.value}",
                f"leader-execution: {execution.value}",
                f"leader-workspace: {leader.workspace.value}",
            ]
        )
        if (tools := self._tools(leader.allowed_tools)) is not None:
            lines.append(f"leader-allowed-tools: {tools}")
        if leader.disallowed_tools:
            lines.append(f"leader-disallowed-tools: {self._tools(leader.disallowed_tools)}")
        lines.append(f"members: {self._tools(members)}")
        lines.extend(["---", leader.prompt, ""])
        return "\n".join(lines)

    def _member_text(self, template: TeamAgentTemplate) -> str:
        lines = ["---", f"name: {template.name}", f"description: {template.description}"]
        if template.when_to_use:
            lines.append(f"when-to-use: {template.when_to_use}")
        lines.extend(
            [f"context: {template.context.value}", f"workspace: {template.workspace.value}"]
        )
        if (tools := self._tools(template.allowed_tools)) is not None:
            lines.append(f"allowed-tools: {tools}")
        if template.disallowed_tools:
            lines.append(f"disallowed-tools: {self._tools(template.disallowed_tools)}")
        lines.extend(["---", template.prompt, ""])
        return "\n".join(lines)

"""Strict dependency-free parser for persistent AgentTeam definitions."""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from hammer_code.agent_team.models import AgentTeamDefinition, TeamAgentTemplate
from hammer_code.subagent.models import SubagentContext, SubagentExecution, SubagentWorkspace

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_TEAM_KEYS = {
    "name",
    "description",
    "leader-name",
    "leader-description",
    "leader-when-to-use",
    "leader-context",
    "leader-execution",
    "leader-workspace",
    "leader-allowed-tools",
    "leader-disallowed-tools",
    "members",
}
_MEMBER_KEYS = {
    "name",
    "description",
    "when-to-use",
    "context",
    "workspace",
    "allowed-tools",
    "disallowed-tools",
}
_LIST_KEYS = {
    "leader-allowed-tools",
    "leader-disallowed-tools",
    "allowed-tools",
    "disallowed-tools",
    "members",
}
_MAX_PROMPT = 32_000


class AgentTeamParseError(ValueError):
    """Definition invalid; deliberately never embeds the untrusted prompt body."""


def is_reparse(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _scalar(value: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise AgentTeamParseError("front matter scalar is invalid")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AgentTeamParseError("front matter JSON string is invalid") from exc
        if not isinstance(decoded, str) or any(char in decoded for char in "\r\n\x00"):
            raise AgentTeamParseError("front matter string must be one line")
        return decoded
    if "#" in value or value.startswith(("[", "{", "&", "*", "!", "|", ">")):
        raise AgentTeamParseError("unsupported front matter syntax")
    return value


def _list(value: str, field: str) -> tuple[str, ...]:
    if value == "[]":
        return ()
    if not value.startswith("[") or not value.endswith("]") or '"' in value:
        raise AgentTeamParseError(f"{field} must be a one-line plain string list")
    result = tuple(item.strip() for item in value[1:-1].split(","))
    if not result or any(
        not item or len(item) > 128 or any(char in item for char in " \t\r\n\x00[]()")
        for item in result
    ):
        raise AgentTeamParseError(f"{field} contains an invalid value")
    if len(set(result)) != len(result):
        raise AgentTeamParseError(f"{field} must not contain duplicates")
    return result


def _front_matter(path: Path, allowed: set[str]) -> tuple[dict[str, str | tuple[str, ...]], str]:
    try:
        if not path.is_file() or is_reparse(path):
            raise AgentTeamParseError("definition must be a regular non-reparse file")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AgentTeamParseError("definition must be UTF-8") from exc
    except OSError as exc:
        raise AgentTeamParseError("definition could not be read") from exc
    if "\x00" in text:
        raise AgentTeamParseError("definition must not contain NUL")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0] != "---":
        raise AgentTeamParseError("definition must begin with front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise AgentTeamParseError("definition front matter is not closed") from exc
    raw: dict[str, str | tuple[str, ...]] = {}
    for line in lines[1:end]:
        if not line or line[:1].isspace() or ":" not in line:
            raise AgentTeamParseError("front matter only supports key: scalar lines")
        key, value = line.split(":", 1)
        if key not in allowed or key in raw or not re.fullmatch(r"[a-z-]+", key):
            raise AgentTeamParseError("front matter has an unknown or duplicate key")
        value = value[1:] if value.startswith(" ") else ""
        raw[key] = _list(value, key) if key in _LIST_KEYS else _scalar(value)
    body = "\n".join(lines[end + 1 :]).strip("\n")
    if not body.strip() or len(body) > _MAX_PROMPT:
        raise AgentTeamParseError("definition body must be non-empty and at most 32000 characters")
    return raw, body


def _line(value: object, field: str, limit: int, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise AgentTeamParseError(f"{field} is invalid")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > limit or "\x00" in value:
        raise AgentTeamParseError(
            f"{field} must be a non-empty single line up to {limit} characters"
        )
    return normalized


def _template(
    raw: dict[str, str | tuple[str, ...]], body: str, *, leader: bool, path: Path
) -> TeamAgentTemplate:
    prefix = "leader-" if leader else ""
    name = raw.get(f"{prefix}name")
    if not isinstance(name, str) or not _NAME.fullmatch(name) or not 1 <= len(name) <= 64:
        raise AgentTeamParseError("name must be a normalized identifier")
    description = _line(raw.get(f"{prefix}description"), "description", 500, required=True)
    assert description is not None
    when = _line(raw.get(f"{prefix}when-to-use"), "when-to-use", 1000)
    try:
        context = SubagentContext(raw.get(f"{prefix}context", SubagentContext.ISOLATED))
        workspace = SubagentWorkspace(
            raw.get(
                f"{prefix}workspace",
                SubagentWorkspace.WORKTREE if leader else SubagentWorkspace.SHARED,
            )
        )
    except ValueError as exc:
        raise AgentTeamParseError("context or workspace is invalid") from exc
    allowed = raw.get(f"{prefix}allowed-tools")
    disallowed = raw.get(f"{prefix}disallowed-tools", ())
    if not isinstance(allowed, tuple | type(None)) or not isinstance(disallowed, tuple):
        raise AgentTeamParseError("tool lists are invalid")
    if set(allowed or ()).intersection(disallowed):
        raise AgentTeamParseError("allowed-tools and disallowed-tools must not overlap")
    return TeamAgentTemplate(
        name, description, when, body, context, workspace, allowed, disallowed, path.resolve()
    )


def parse_team(path: Path) -> AgentTeamDefinition:
    raw, body = _front_matter(path, _TEAM_KEYS)
    name = raw.get("name")
    if (
        not isinstance(name, str)
        or not _NAME.fullmatch(name)
        or not 1 <= len(name) <= 64
        or path.parent.name != name
    ):
        raise AgentTeamParseError("Team name must match its directory")
    description = _line(raw.get("description"), "description", 500, required=True)
    assert description is not None
    try:
        execution = SubagentExecution(raw.get("leader-execution", SubagentExecution.INLINE))
    except ValueError as exc:
        raise AgentTeamParseError("leader-execution is invalid") from exc
    members = raw.get("members", ())
    if (
        not isinstance(members, tuple)
        or len(members) > 64
        or any(not _NAME.fullmatch(item) for item in members)
    ):
        raise AgentTeamParseError("members must be normalized template names")
    return AgentTeamDefinition(
        name,
        description,
        _template(raw, body, leader=True, path=path),
        execution,
        members,
        path.resolve(),
    )


def parse_member(path: Path) -> TeamAgentTemplate:
    raw, body = _front_matter(path, _MEMBER_KEYS)
    template = _template(raw, body, leader=False, path=path)
    if path.stem != template.name:
        raise AgentTeamParseError("member name must match its filename")
    return template

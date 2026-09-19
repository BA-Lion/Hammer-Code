"""Strict, dependency-free ``SKILL.md`` parsing and rendering."""

from __future__ import annotations

import json
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from hammer_code.skill.models import SkillContext, SkillDefinition, SkillScope, SkillSource

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_KEYS = {
    "name",
    "description",
    "version",
    "created-at",
    "updated-at",
    "source",
    "user-invocable",
    "model-invocable",
    "context",
    "when-to-use",
    "arguments-required",
    "argument-hint",
    "allowed-tools",
}


class SkillParseError(ValueError):
    pass


class SkillInvocationError(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _single_line(value: str, *, name: str, limit: int, required: bool = False) -> str | None:
    normalized = " ".join(value.split())
    if not normalized and not required:
        return None
    if not normalized or len(normalized) > limit or "\x00" in value:
        raise SkillParseError(f"{name} must be a non-empty single line up to {limit} characters")
    return normalized


def _parse_scalar(value: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise SkillParseError("front matter scalar is invalid")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SkillParseError("front matter JSON string is invalid") from exc
        if not isinstance(parsed, str) or any(char in parsed for char in "\r\n\x00"):
            raise SkillParseError("front matter string must be one line")
        return parsed
    if "#" in value or value.startswith(("[", "{", "&", "*", "!", "|", ">")):
        raise SkillParseError("unsupported front matter syntax")
    return value


def _parse_list(value: str) -> tuple[str, ...]:
    if value == "[]":
        return ()
    if not value.startswith("[") or not value.endswith("]") or '"' in value:
        raise SkillParseError("allowed-tools must be a one-line plain string list")
    values = tuple(item.strip() for item in value[1:-1].split(","))
    if not values or any(
        not item or any(char in item for char in " \t\r\n\x00[]") for item in values
    ):
        raise SkillParseError("allowed-tools contains an invalid name")
    if len(set(values)) != len(values):
        raise SkillParseError("allowed-tools must not contain duplicates")
    return values


def _timestamp(value: str, field: str) -> datetime:
    if not _TIME.fullmatch(value):
        raise SkillParseError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise SkillParseError(f"{field} must be an RFC3339 UTC timestamp") from exc


def parse_skill(path: Path, scope: SkillScope) -> SkillDefinition:
    """Parse one regular UTF-8 file without ever inferring a writable path."""
    try:
        if not path.is_file() or _is_reparse(path):
            raise SkillParseError("SKILL.md must be a regular non-reparse file")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillParseError("SKILL.md must be UTF-8") from exc
    except OSError as exc:
        raise SkillParseError("SKILL.md could not be read") from exc
    if "\x00" in text:
        raise SkillParseError("SKILL.md must not contain NUL")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0] != "---":
        raise SkillParseError("SKILL.md must begin with front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise SkillParseError("SKILL.md front matter is not closed") from exc
    raw: dict[str, str | tuple[str, ...]] = {}
    for line in lines[1:end]:
        if not line or line[:1].isspace() or line.count(":") < 1:
            raise SkillParseError("front matter only supports key: scalar lines")
        key, value = line.split(":", 1)
        if key not in _KEYS or key in raw or not re.fullmatch(r"[a-z-]+", key):
            raise SkillParseError("front matter has an unknown or duplicate key")
        value = value[1:] if value.startswith(" ") else ""
        raw[key] = _parse_list(value) if key == "allowed-tools" else _parse_scalar(value)
    body = "\n".join(lines[end + 1 :]).strip("\n")
    if not body.strip():
        raise SkillParseError("SKILL.md body must not be empty")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name) or len(name) > 64:
        raise SkillParseError("name must be a normalized 1-64 character identifier")
    description = raw.get("description")
    if not isinstance(description, str):
        raise SkillParseError("description is required")
    description = _single_line(description, name="description", limit=500, required=True)
    assert description is not None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0)
    version = raw.get("version", "0.1.0")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise SkillParseError("version must be SemVer")
    created = raw.get("created-at")
    updated = raw.get("updated-at")
    created_at = _timestamp(created, "created-at") if isinstance(created, str) else mtime
    updated_at = _timestamp(updated, "updated-at") if isinstance(updated, str) else mtime
    source = raw.get("source")
    try:
        parsed_source = SkillSource(source) if isinstance(source, str) else SkillSource.IMPORTED
        context = SkillContext(raw.get("context", SkillContext.INLINE))
    except ValueError as exc:
        raise SkillParseError("source or context is invalid") from exc
    booleans: dict[str, bool] = {}
    for key, default in (
        ("user-invocable", True),
        ("model-invocable", True),
        ("arguments-required", False),
    ):
        value = raw.get(key, default)
        if value not in ("true", "false", True, False):
            raise SkillParseError(f"{key} must be true or false")
        booleans[key] = value is True or value == "true"
    when = raw.get("when-to-use")
    hint = raw.get("argument-hint")
    if when is not None and not isinstance(when, str):
        raise SkillParseError("when-to-use is invalid")
    if hint is not None and not isinstance(hint, str):
        raise SkillParseError("argument-hint is invalid")
    allowed = raw.get("allowed-tools")
    return SkillDefinition(
        name,
        description,
        version,
        created_at,
        updated_at,
        parsed_source,
        body,
        scope,
        path.parent.resolve(),
        booleans["user-invocable"],
        booleans["model-invocable"],
        context,
        _single_line(when, name="when-to-use", limit=1000) if when else None,
        booleans["arguments-required"],
        _single_line(hint, name="argument-hint", limit=500) if hint else None,
        allowed if isinstance(allowed, tuple) else None,
    )


def serialize_skill(definition: SkillDefinition) -> str:
    """Emit canonical, portable front matter. Paths never appear in output."""

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def stamp(value: datetime) -> str:
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "---",
        f"name: {definition.name}",
        f"description: {quote(definition.description)}",
        f"version: {definition.version}",
        f"created-at: {stamp(definition.created_at)}",
        f"updated-at: {stamp(definition.updated_at)}",
        f"source: {definition.source.value}",
        f"user-invocable: {str(definition.user_invocable).lower()}",
        f"model-invocable: {str(definition.model_invocable).lower()}",
        f"context: {definition.context.value}",
    ]
    if definition.when_to_use is not None:
        lines.append(f"when-to-use: {quote(definition.when_to_use)}")
    lines.append(f"arguments-required: {str(definition.arguments_required).lower()}")
    if definition.argument_hint is not None:
        lines.append(f"argument-hint: {quote(definition.argument_hint)}")
    if definition.allowed_tools is not None:
        lines.append(f"allowed-tools: [{', '.join(definition.allowed_tools)}]")
    return "\n".join((*lines, "---", definition.prompt_template.rstrip("\n"), ""))


def render_skill(definition: SkillDefinition, arguments: str) -> str:
    if definition.arguments_required and not arguments.strip():
        suffix = f": {definition.argument_hint}" if definition.argument_hint else ""
        raise SkillInvocationError(f"Skill '{definition.name}' requires arguments{suffix}")
    substitutions = {"{{arguments}}": arguments, "{{skill_dir}}": str(definition.skill_dir)}
    return re.sub(
        r"\{\{arguments\}\}|\{\{skill_dir\}\}",
        lambda match: substitutions[match[0]],
        definition.prompt_template,
    )

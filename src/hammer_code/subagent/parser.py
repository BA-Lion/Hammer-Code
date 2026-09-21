"""Strict, dependency-free parsing for project Subagent Markdown definitions."""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from hammer_code.config import SubagentConfig
from hammer_code.subagent.models import SubagentContext, SubagentDefinition, SubagentExecution

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_KEYS = {
    "name",
    "description",
    "when-to-use",
    "context",
    "execution",
    "allowed-tools",
    "disallowed-tools",
    "max-iterations",
}
_LIST_KEYS = {"allowed-tools", "disallowed-tools"}


class SubagentParseError(ValueError):
    """A definition is invalid without including its potentially sensitive body."""


def _is_reparse(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _scalar(value: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise SubagentParseError("front matter scalar is invalid")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SubagentParseError("front matter JSON string is invalid") from exc
        if not isinstance(parsed, str) or any(char in parsed for char in "\r\n\x00"):
            raise SubagentParseError("front matter string must be one line")
        return parsed
    if "#" in value or value.startswith(("[", "{", "&", "*", "!", "|", ">")):
        raise SubagentParseError("unsupported front matter syntax")
    return value


def _tool_list(value: str, field: str) -> tuple[str, ...]:
    if value == "[]":
        return ()
    if not value.startswith("[") or not value.endswith("]") or '"' in value:
        raise SubagentParseError(f"{field} must be a one-line plain string list")
    values = tuple(part.strip() for part in value[1:-1].split(","))
    if not values or any(
        not item or len(item) > 128 or any(char in item for char in " \t\r\n\x00[]()")
        for item in values
    ):
        raise SubagentParseError(f"{field} contains an invalid tool name")
    if len(set(values)) != len(values):
        raise SubagentParseError(f"{field} must not contain duplicates")
    return values


def _single_line(value: str, field: str, limit: int, *, required: bool = False) -> str | None:
    normalized = " ".join(value.split())
    if not normalized and not required:
        return None
    if not normalized or len(normalized) > limit or "\x00" in value:
        raise SubagentParseError(
            f"{field} must be a non-empty single line up to {limit} characters"
        )
    return normalized


def parse_subagent(path: Path, config: SubagentConfig) -> SubagentDefinition:
    """Parse one regular UTF-8 definition; caller owns catalog containment checks."""
    try:
        if not path.is_file() or _is_reparse(path):
            raise SubagentParseError("definition must be a regular non-reparse file")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SubagentParseError("definition must be UTF-8") from exc
    except OSError as exc:
        raise SubagentParseError("definition could not be read") from exc
    if "\x00" in text:
        raise SubagentParseError("definition must not contain NUL")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0] != "---":
        raise SubagentParseError("definition must begin with front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise SubagentParseError("definition front matter is not closed") from exc
    raw: dict[str, str | tuple[str, ...]] = {}
    for line in lines[1:end]:
        if not line or line[:1].isspace() or ":" not in line:
            raise SubagentParseError("front matter only supports key: scalar lines")
        key, value = line.split(":", 1)
        if key not in _KEYS or key in raw or not re.fullmatch(r"[a-z-]+", key):
            raise SubagentParseError("front matter has an unknown or duplicate key")
        value = value[1:] if value.startswith(" ") else ""
        raw[key] = _tool_list(value, key) if key in _LIST_KEYS else _scalar(value)
    body = "\n".join(lines[end + 1 :]).strip("\n")
    if not body.strip() or len(body) > 32000:
        raise SubagentParseError("definition body must be non-empty and at most 32000 characters")
    name = raw.get("name")
    if (
        not isinstance(name, str)
        or not _NAME.fullmatch(name)
        or not 1 <= len(name) <= 64
        or path.stem != name
    ):
        raise SubagentParseError("name must be a normalized filename-matching identifier")
    description = raw.get("description")
    if not isinstance(description, str):
        raise SubagentParseError("description is required")
    description = _single_line(description, "description", 500, required=True)
    assert description is not None
    when = raw.get("when-to-use")
    if when is not None and not isinstance(when, str):
        raise SubagentParseError("when-to-use is invalid")
    try:
        context = SubagentContext(raw.get("context", SubagentContext.ISOLATED))
        execution = SubagentExecution(raw.get("execution", SubagentExecution.INLINE))
    except ValueError as exc:
        raise SubagentParseError("context or execution is invalid") from exc
    max_iterations_raw = raw.get("max-iterations", str(config.default_max_iterations))
    if (
        not isinstance(max_iterations_raw, str)
        or not max_iterations_raw.isascii()
        or not max_iterations_raw.isdecimal()
    ):
        raise SubagentParseError("max-iterations must be an integer")
    max_iterations = int(max_iterations_raw)
    if not 1 <= max_iterations <= 50:
        raise SubagentParseError("max-iterations must be between 1 and 50")
    allowed = raw.get("allowed-tools")
    disallowed = raw.get("disallowed-tools", ())
    if not isinstance(allowed, tuple | type(None)) or not isinstance(disallowed, tuple):
        raise SubagentParseError("tool lists are invalid")
    if set(allowed or ()).intersection(disallowed):
        raise SubagentParseError("allowed-tools and disallowed-tools must not overlap")
    return SubagentDefinition(
        name=name,
        description=description,
        when_to_use=_single_line(when, "when-to-use", 1000) if when else None,
        prompt=body,
        context=context,
        execution=execution,
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        max_iterations=max_iterations,
        source_path=path.resolve(),
    )

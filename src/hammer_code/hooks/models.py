"""Provider-independent Hook runtime values and pure matching helpers."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class LifecycleEvent(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    PRE_SEND = "pre_send"
    POST_RECEIVE = "post_receive"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    COMPACT = "compact"
    ERROR = "error"


class ActionType(StrEnum):
    COMMAND = "command"
    PROMPT = "prompt"
    HTTP = "http"
    AGENT = "agent"


class ConditionMode(StrEnum):
    ALL = "all"
    ANY = "any"


class ConditionOperator(StrEnum):
    EQ = "=="
    NE = "!="
    REGEX = "=~"
    GLOB = "~="


@dataclass(frozen=True)
class Action:
    type: ActionType
    command: str = ""
    prompt: str = ""
    url: str = ""
    method: str = "POST"
    body: str = ""
    headers: Mapping[str, str] = dc_field(default_factory=dict)
    timeout: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


@dataclass(frozen=True)
class Condition:
    field: str
    operator: ConditionOperator
    value: str
    _pattern: re.Pattern[str] | None = dc_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pattern = re.compile(self.value) if self.operator is ConditionOperator.REGEX else None
        object.__setattr__(self, "_pattern", pattern)


@dataclass(frozen=True)
class ConditionGroup:
    mode: ConditionMode
    rules: tuple[Condition, ...]


@dataclass
class Hook:
    id: str
    event: LifecycleEvent
    action: Action
    condition: ConditionGroup | None = None
    reject: bool = False
    once: bool = False
    async_exec: bool = False
    executed: bool = False


@dataclass(frozen=True)
class HookContext:
    event_name: str = ""
    tool_name: str = ""
    tool_args: Mapping[str, Any] = dc_field(default_factory=dict)
    file_path: str = ""
    message: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_args", MappingProxyType(dict(self.tool_args)))


@dataclass(frozen=True)
class HookDispatchResult:
    rejected: bool = False


@dataclass(frozen=True)
class PromptItem:
    identity: int
    text: str


@dataclass(frozen=True)
class PromptBatch:
    items: tuple[PromptItem, ...] = ()

    @property
    def text(self) -> str:
        return "\n\n".join(item.text for item in self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


class HookExpansionError(ValueError):
    """An action template referenced a missing or unsupported Hook value."""


_PLACEHOLDER = re.compile(
    r"\$TOOL_ARGS\.([A-Za-z_][A-Za-z0-9_-]*)|"
    r"\$(EVENT|TOOL_NAME|FILE_PATH|MESSAGE|ERROR)(?![A-Za-z0-9_])"
)


def matches_condition(group: ConditionGroup | None, context: HookContext) -> bool:
    if group is None:
        return True
    matches = tuple(_matches_rule(rule, context) for rule in group.rules)
    return all(matches) if group.mode is ConditionMode.ALL else any(matches)


def expand_template(template: str, context: HookContext) -> str:
    """Expand only the fixed Hook placeholders in one regex scan."""

    if "$TOOL_ARGS." in template:
        for occurrence in re.finditer(r"\$TOOL_ARGS\.", template):
            suffix = template[occurrence.end() :]
            if not re.match(r"[A-Za-z_][A-Za-z0-9_-]*", suffix):
                raise HookExpansionError("incomplete tool argument placeholder")

    def replacement(match: re.Match[str]) -> str:
        argument_name, fixed_name = match.groups()
        if argument_name is not None:
            value = _normalize_scalar(context.tool_args.get(argument_name, _MISSING))
            if value is None:
                raise HookExpansionError("tool argument placeholder has no scalar value")
            return value
        field_name = {
            "EVENT": "event_name",
            "TOOL_NAME": "tool_name",
            "FILE_PATH": "file_path",
            "MESSAGE": "message",
            "ERROR": "error",
        }[fixed_name]
        value = _normalize_scalar(getattr(context, field_name))
        if value is None or value == "":
            raise HookExpansionError("hook placeholder has no value")
        return value

    return _PLACEHOLDER.sub(replacement, template)


_MISSING = object()


def _matches_rule(rule: Condition, context: HookContext) -> bool:
    value = _condition_value(rule.field, context)
    if value is None:
        return False
    if rule.operator is ConditionOperator.EQ:
        return value == rule.value
    if rule.operator is ConditionOperator.NE:
        return value != rule.value
    if rule.operator is ConditionOperator.REGEX:
        assert rule._pattern is not None
        return rule._pattern.search(value) is not None
    return fnmatch.fnmatchcase(value, rule.value)


def _condition_value(name: str, context: HookContext) -> str | None:
    if name.startswith("tool_args."):
        return _normalize_scalar(context.tool_args.get(name.removeprefix("tool_args."), _MISSING))
    if name not in {"event_name", "tool_name", "file_path", "message", "error"}:
        return None
    value = getattr(context, name)
    return _normalize_scalar(value) if value != "" else None


def _normalize_scalar(value: object) -> str | None:
    if value is _MISSING or value is None or isinstance(value, (list, dict, tuple, set)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return None

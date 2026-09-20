"""Strict loading of project-local Hook definitions."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hammer_code.errors import ConfigurationError
from hammer_code.hooks.models import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionMode,
    ConditionOperator,
    Hook,
    LifecycleEvent,
)

_HOOK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_CONDITION_FIELD = re.compile(
    r"(?:event_name|tool_name|file_path|message|error|tool_args\.[A-Za-z_][A-Za-z0-9_-]*)\Z"
)
_HTTP_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_CREDENTIAL_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
}


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: ActionType
    command: str = ""
    prompt: str = ""
    url: str = ""
    method: str = "POST"
    body: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: int = Field(default=30, ge=1, le=1800)

    @field_validator("method")
    @classmethod
    def _method(cls, value: str) -> str:
        normalized = value.upper()
        if not _HTTP_TOKEN.fullmatch(normalized):
            raise ValueError("must be a valid HTTP method token")
        return normalized

    @field_validator("headers")
    @classmethod
    def _headers(cls, value: dict[str, str]) -> dict[str, str]:
        for name, header_value in value.items():
            if not _HTTP_TOKEN.fullmatch(name) or any(char in header_value for char in "\r\n\0"):
                raise ValueError("headers must have valid names and single-line values")
            if name.casefold() in _CREDENTIAL_HEADERS:
                raise ValueError("credential headers are not supported")
        return dict(value)

    @model_validator(mode="after")
    def _combination(self) -> ActionConfig:
        if self.type is ActionType.AGENT:
            raise ValueError("agent actions are not supported in the initial release")
        if self.type is ActionType.COMMAND:
            if not self.command.strip():
                raise ValueError("command actions require a non-empty command")
            if self.prompt or self.url or self.body or self.headers or self.method != "POST":
                raise ValueError("command actions contain unsupported fields")
        elif self.type is ActionType.PROMPT:
            if not self.prompt.strip():
                raise ValueError("prompt actions require a non-empty prompt")
            if (
                self.command
                or self.url
                or self.body
                or self.headers
                or self.method != "POST"
                or self.timeout != 30
            ):
                raise ValueError("prompt actions contain unsupported fields")
        else:
            _validate_http_url(self.url)
            if self.command or self.prompt:
                raise ValueError("http actions contain unsupported fields")
        return self

    def runtime(self) -> Action:
        return Action(
            type=self.type,
            command=self.command,
            prompt=self.prompt,
            url=self.url,
            method=self.method,
            body=self.body,
            headers=self.headers,
            timeout=self.timeout,
        )


class ConditionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    operator: ConditionOperator
    value: str

    @field_validator("field")
    @classmethod
    def _field(cls, value: str) -> str:
        if not _CONDITION_FIELD.fullmatch(value):
            raise ValueError("unsupported condition field")
        return value

    @model_validator(mode="after")
    def _regex(self) -> ConditionConfig:
        if self.operator is ConditionOperator.REGEX:
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError("invalid regular expression") from exc
        return self

    def runtime(self) -> Condition:
        return Condition(self.field, self.operator, self.value)


class ConditionGroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: ConditionMode
    rules: tuple[ConditionConfig, ...] = Field(min_length=1)

    def runtime(self) -> ConditionGroup:
        return ConditionGroup(self.mode, tuple(rule.runtime() for rule in self.rules))


class HookDefinitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    event: LifecycleEvent
    action: ActionConfig
    condition: ConditionGroupConfig | None = None
    reject: bool = False
    once: bool = False
    async_exec: bool = False

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        normalized = value.strip()
        if not _HOOK_ID.fullmatch(normalized):
            raise ValueError("must be a valid 1-64 character Hook id")
        return normalized

    @model_validator(mode="after")
    def _semantics(self) -> HookDefinitionConfig:
        if self.reject and self.event is not LifecycleEvent.PRE_TOOL_USE:
            raise ValueError("reject is only valid for pre_tool_use")
        if self.async_exec and self.action.type is ActionType.PROMPT:
            raise ValueError("prompt actions cannot run asynchronously")
        return self

    def runtime(self) -> Hook:
        return Hook(
            id=self.id,
            event=self.event,
            action=self.action.runtime(),
            condition=self.condition.runtime() if self.condition else None,
            reject=self.reject,
            once=self.once,
            async_exec=self.async_exec,
        )


class HookFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hooks: tuple[HookDefinitionConfig, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> HookFileConfig:
        ids = [hook.id for hook in self.hooks]
        if len(ids) != len(set(ids)):
            raise ValueError("Hook ids must be unique")
        return self


def load_hook_definitions(path: Path) -> tuple[HookDefinitionConfig, ...]:
    if not path.exists():
        return ()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return HookFileConfig.model_validate(raw).hooks
    except OSError as exc:
        raise ConfigurationError("Unable to read Hook configuration") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError("Invalid Hook configuration: TOML syntax error") from exc
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        category = str(first["type"])
        raise ConfigurationError(
            f"Invalid Hook configuration at {location or 'root'} ({category})"
        ) from exc


def instantiate_hooks(definitions: tuple[HookDefinitionConfig, ...]) -> tuple[Hook, ...]:
    return tuple(definition.runtime() for definition in definitions)


def _validate_http_url(value: str) -> None:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("http actions require a safe absolute http(s) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port is not None
        and not 0 < port <= 65535
    ):
        raise ValueError("http actions require a safe absolute http(s) URL")

"""Strict project configuration, endpoint trust, and secret resolution."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from hammer_code.errors import ConfigurationError, UntrustedEndpointError

ProtocolName = Literal["openai_responses", "openai_chat_completions", "anthropic_messages"]
BUILTIN_TOOL_NAMES = {"read_file", "edit_file", "create_file", "grep", "glob", "shell"}
McpTransport = Literal["stdio", "streamable_http"]
_MCP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class BaseModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: str
    model: str
    base_url: str
    api_key_env: str
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = 0

    @field_validator("model", "api_key_env")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute http(s) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _no_retries(self) -> BaseModelProfile:
        if self.max_retries != 0:
            raise ValueError("max_retries must be 0 in the initial release")
        return self


class OpenAIResponsesProfile(BaseModelProfile):
    protocol: Literal["openai_responses"]  # pyright: ignore[reportIncompatibleVariableOverride]
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None


class OpenAIChatProfile(BaseModelProfile):
    protocol: Literal["openai_chat_completions"]  # pyright: ignore[reportIncompatibleVariableOverride]
    stream_include_usage: bool = True


class AnthropicMessagesProfile(BaseModelProfile):
    protocol: Literal["anthropic_messages"]  # pyright: ignore[reportIncompatibleVariableOverride]
    thinking_mode: Literal["disabled", "enabled", "adaptive"] = "disabled"
    thinking_budget: int | None = Field(default=None, gt=0)
    effort: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _thinking_combinations(self) -> AnthropicMessagesProfile:
        if self.thinking_budget is not None and self.thinking_mode != "enabled":
            raise ValueError("thinking_budget requires thinking_mode = 'enabled'")
        if self.effort is not None and self.thinking_mode == "disabled":
            raise ValueError("effort requires enabled or adaptive thinking")
        return self


Profile = Annotated[
    OpenAIResponsesProfile | OpenAIChatProfile | AnthropicMessagesProfile,
    Field(discriminator="protocol"),
]


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    show_reasoning: bool = False


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    disabled: tuple[str, ...] = ()

    @field_validator("disabled")
    @classmethod
    def _disabled_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(name not in BUILTIN_TOOL_NAMES for name in value):
            raise ValueError("disabled tools must be unique built-in tool names")
        return value


class McpBaseConfig(BaseModel):
    """Common, secret-free MCP configuration registered before connection."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    description: str
    transport: str
    connect_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    tool_timeout_seconds: float = Field(default=120.0, gt=0, le=1800)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _MCP_NAME.fullmatch(value):
            raise ValueError("must be 1-64 ASCII letters, digits, dots, underscores, or hyphens")
        return value

    @field_validator("description")
    @classmethod
    def _description(cls, value: str) -> str:
        normalized = value.strip()
        if not 1 <= len(normalized) <= 500 or any(char in normalized for char in "\r\n\0"):
            raise ValueError("must be a non-empty single line of at most 500 characters")
        return normalized


class McpStdioConfig(McpBaseConfig):
    transport: Literal["stdio"]  # pyright: ignore[reportIncompatibleVariableOverride]
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None

    @field_validator("command")
    @classmethod
    def _command(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("must be non-empty and contain no NUL")
        return value

    @field_validator("args")
    @classmethod
    def _args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("\0" in item for item in value):
            raise ValueError("arguments must not contain NUL")
        return value

    @field_validator("env")
    @classmethod
    def _env(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not _ENV_NAME.fullmatch(key) or not _ENV_NAME.fullmatch(source)
            for key, source in value.items()
        ):
            raise ValueError("environment names must be valid process variable names")
        return value


class McpHttpConfig(McpBaseConfig):
    transport: Literal["streamable_http"]  # pyright: ignore[reportIncompatibleVariableOverride]
    endpoint: str

    @field_validator("endpoint")
    @classmethod
    def _endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("must have a valid port") from exc
        host = (parsed.hostname or "").casefold()
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or port is not None
            and not 0 < port <= 65535
            or parsed.scheme == "http"
            and host not in _LOOPBACK_HOSTS
        ):
            raise ValueError("must be a safe absolute HTTPS URL (HTTP is loopback-only)")
        return value


McpConfig = Annotated[McpStdioConfig | McpHttpConfig, Field(discriminator="transport")]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    default_profile: str
    profiles: dict[str, Profile]
    ui: UIConfig = UIConfig()
    tools: ToolConfig = ToolConfig()
    mcp: tuple[McpConfig, ...] = ()

    @model_validator(mode="after")
    def _default_exists(self) -> AppConfig:
        if not self.default_profile or self.default_profile not in self.profiles:
            raise ValueError("default_profile must name an existing profile")
        if any(not name.strip() for name in self.profiles):
            raise ValueError("profile names must not be blank")
        final_mcp: dict[str, McpConfig] = {}
        for item in self.mcp:
            final_mcp.pop(item.name, None)
            final_mcp[item.name] = item
        object.__setattr__(self, "mcp", tuple(final_mcp.values()))
        return self


class ResolvedProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    profile: Profile
    api_key: SecretStr


def discover_config(explicit_path: Path | None, start_dir: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {path}")
        return path
    current = (start_dir or Path.cwd()).resolve()
    while True:
        candidate = current / ".hammer-code" / "config.toml"
        if candidate.is_file():
            return candidate.resolve()
        if (current / ".git").exists() or current.parent == current:
            break
        current = current.parent
    raise ConfigurationError(
        "No .hammer-code/config.toml found from the current directory to Git root"
    )


def load_config(path: Path) -> AppConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(AppConfig).validate_python(raw)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


class EndpointTrustPolicy:
    _OFFICIAL_ORIGINS = {
        ("https", "api.openai.com", 443),
        ("https", "api.anthropic.com", 443),
        ("https", "api.deepseek.com", 443),
    }

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int]:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port or (443 if scheme == "https" else 80)
        return scheme, host, port

    def validate_and_confirm(
        self, profile: BaseModelProfile, confirmer: Callable[[str], bool]
    ) -> None:
        scheme, host, port = self._origin(profile.base_url)
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if scheme != "https" and not loopback:
            raise UntrustedEndpointError("Remote model endpoints must use HTTPS")
        if (scheme, host, port) not in self._OFFICIAL_ORIGINS:
            if not confirmer(f"Trust model endpoint {scheme}://{host}:{port} for this run?"):
                raise UntrustedEndpointError("Custom endpoint was not confirmed")


def resolve_profile(
    config: AppConfig, requested_name: str | None, environ: Mapping[str, str] | None = None
) -> ResolvedProfile:
    name = requested_name or config.default_profile
    try:
        profile = config.profiles[name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown profile: {name}") from exc
    value = (environ or os.environ).get(profile.api_key_env)
    if not value:
        raise ConfigurationError(
            f"Environment variable {profile.api_key_env} is required for profile {name}"
        )
    return ResolvedProfile(name=name, profile=profile, api_key=SecretStr(value))

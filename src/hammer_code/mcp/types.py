"""Internal MCP adapter state that never crosses into the domain layer."""

from dataclasses import dataclass
from enum import StrEnum


class McpClientStatus(StrEnum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


class McpManagerStatus(StrEnum):
    CONNECTING = "connecting"
    SETTLED = "settled"


@dataclass(frozen=True)
class McpCallResult:
    content: str
    is_error: bool

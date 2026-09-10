"""Immutable, provider-independent permission values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class PermissionEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ApprovalChoice(StrEnum):
    ALLOW_ONCE = "allow_once"
    ALLOW_AND_PERSIST = "allow_and_persist"
    DENY = "deny"


class PermissionMode(StrEnum):
    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    STRICT = "strict"
    UNATTENDED = "unattended"


@dataclass(frozen=True)
class PermissionDecision:
    effect: PermissionEffect
    reason: str


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    category: str
    normalized_arguments: dict[str, Any]
    canonical_paths: tuple[Path, ...]
    workspace_root: Path
    cwd: Path
    normalized_command: str | None
    fingerprint: str

    @classmethod
    def for_command(
        cls, tool_name: str, command: str, workspace_root: Path, cwd: Path
    ) -> PermissionRequest:
        root = workspace_root.resolve()
        canonical_cwd = cwd.resolve()
        arguments: dict[str, Any] = {"command": command}
        fingerprint = _fingerprint(
            {"tool": tool_name, "arguments": arguments, "cwd": str(canonical_cwd)}
        )
        return cls(tool_name, "command", arguments, (), root, canonical_cwd, command, fingerprint)

    @classmethod
    def for_tool(
        cls,
        tool_name: str,
        category: str,
        arguments: dict[str, Any],
        paths: tuple[Path, ...],
        workspace_root: Path,
        cwd: Path,
    ) -> PermissionRequest:
        root = workspace_root.resolve()
        canonical_paths = tuple(path.resolve() for path in paths)
        payload = {
            "tool": tool_name,
            "arguments": arguments,
            "paths": [str(path) for path in canonical_paths],
            "cwd": str(cwd.resolve()),
        }
        return cls(
            tool_name,
            category,
            dict(arguments),
            canonical_paths,
            root,
            cwd.resolve(),
            None,
            _fingerprint(payload),
        )


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()

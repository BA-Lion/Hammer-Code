"""Fail-closed, workspace-bound TOML permission rules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from hammer_code.errors import RuleFileError
from hammer_code.permissions.models import PermissionEffect, PermissionRequest


@dataclass(frozen=True)
class Rule:
    effect: PermissionEffect
    tool: str
    fingerprint: str | None = None
    command: str | None = None
    cwd: str | None = None
    command_regex: str | None = None

    def matches(self, request: PermissionRequest) -> bool:
        if self.tool != request.tool_name:
            return False
        if self.fingerprint is not None:
            return self.fingerprint == request.fingerprint
        if request.normalized_command is None:
            return False
        if self.command is not None:
            return self.command == request.normalized_command and (
                self.cwd is None or self.cwd == str(request.cwd)
            )
        return (
            self.command_regex is not None
            and re.fullmatch(self.command_regex, request.normalized_command) is not None
        )


class RuleStore:
    def __init__(self, workspace_root: Path, base_dir: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        local = base_dir or (
            Path(os.environ["LOCALAPPDATA"]) if "LOCALAPPDATA" in os.environ else None
        )
        root_bytes = str(self.workspace_root).casefold().encode()
        workspace_hash = hashlib.sha256(root_bytes).hexdigest()[:32]
        self.path = (
            local / "HammerCode" / "permissions" / f"{workspace_hash}.toml"
            if local is not None
            else None
        )
        self._rules = self._load()

    def _load(self) -> tuple[Rule, ...]:
        if self.path is None or not self.path.exists():
            return ()
        try:
            raw = tomllib.loads(self.path.read_text(encoding="utf-8"))
            if set(raw) - {"version", "workspace", "allow", "deny"} or raw.get("version") != 1:
                raise RuleFileError("Permission rule file has an unsupported format")
            if raw.get("workspace") != str(self.workspace_root):
                raise RuleFileError("Permission rule file belongs to another workspace")
            rules = tuple(
                self._parse_rule(PermissionEffect.ALLOW, item) for item in raw.get("allow", [])
            ) + tuple(self._parse_rule(PermissionEffect.DENY, item) for item in raw.get("deny", []))
            return rules
        except (OSError, tomllib.TOMLDecodeError, TypeError, re.error) as exc:
            raise RuleFileError("Permission rule file is invalid") from exc

    @staticmethod
    def _parse_rule(effect: PermissionEffect, raw: object) -> Rule:
        if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
            raise RuleFileError("Permission rule must contain tool")
        allowed = {"tool", "fingerprint", "command", "cwd", "command_regex"}
        if set(raw) - allowed:
            raise RuleFileError("Permission rule contains unknown fields")
        selectors = [key for key in ("fingerprint", "command", "command_regex") if key in raw]
        if len(selectors) != 1 or any(not isinstance(raw[key], str) for key in selectors):
            raise RuleFileError("Permission rule must contain exactly one selector")
        if "cwd" in raw and ("command" not in raw or not isinstance(raw["cwd"], str)):
            raise RuleFileError("Permission rule cwd requires command")
        if "command_regex" in raw:
            re.compile(raw["command_regex"])
        return Rule(effect, **raw)

    def effect_for(self, request: PermissionRequest) -> PermissionEffect | None:
        for effect in (PermissionEffect.DENY, PermissionEffect.ALLOW):
            if any(rule.effect is effect and rule.matches(request) for rule in self._rules):
                return effect
        return None

    def persist_allow(self, request: PermissionRequest) -> None:
        if self.path is None:
            raise RuleFileError("Local application data is unavailable; cannot persist approval")
        rule = (
            Rule(
                PermissionEffect.ALLOW,
                "shell",
                command=request.normalized_command,
                cwd=str(request.cwd),
            )
            if request.normalized_command is not None
            else Rule(PermissionEffect.ALLOW, request.tool_name, fingerprint=request.fingerprint)
        )
        if rule in self._rules:
            return
        self._rules = (*self._rules, rule)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "version = 1",
            f"workspace = {json.dumps(str(self.workspace_root), ensure_ascii=False)}",
        ]
        for current in self._rules:
            heading = "allow" if current.effect is PermissionEffect.ALLOW else "deny"
            lines.append(f"\n[[{heading}]]")
            lines.append(f"tool = {json.dumps(current.tool, ensure_ascii=False)}")
            for key in ("fingerprint", "command", "cwd", "command_regex"):
                value = getattr(current, key)
                if value is not None:
                    lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        descriptor, temporary = tempfile.mkstemp(
            prefix="rules-", suffix=".toml", dir=self.path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

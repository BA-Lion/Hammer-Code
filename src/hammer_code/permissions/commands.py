"""Conservative allow and hard-deny matching for PowerShell command strings."""

from __future__ import annotations

import re

from hammer_code.permissions.models import PermissionEffect


class CommandPolicy:
    _UNSAFE_COMPOSITION = re.compile(r"[\r\n\x00|&;><`]|\$\(")
    _SAFE = (
        re.compile(
            r"git status(?: --short| --porcelain(?:=v[12])?| --branch| --show-stash|"
            r" --ahead-behind| --untracked-files=(?:no|normal|all))*",
            re.IGNORECASE,
        ),
        re.compile(r"git --version", re.IGNORECASE),
        re.compile(r"(?:python|py) (?:--version|-V)", re.IGNORECASE),
        re.compile(r"uv --version", re.IGNORECASE),
    )
    _DANGEROUS = (
        re.compile(r"\brm\s+(?:-[^\s]*[rf][^\s]*\s+|--recursive\s+--force\s+)", re.I),
        re.compile(r"\bremove-item\b[^\r\n]*(?:-recurse|/s)", re.I),
        re.compile(r"\b(?:rmdir|del)\b[^\r\n]*(?:/s|/S)", re.I),
        re.compile(r"\b(?:format-volume|clear-disk|initialize-disk|diskpart|format\.com)\b", re.I),
        re.compile(r"\\\\\.\\physicaldrive|/dev/sd[a-z]", re.I),
        re.compile(r"\bbcdedit\b", re.I),
        re.compile(
            r"(?:invoke-expression|\biex\b).*(?:http|invoke-webrequest|curl)|"
            r"(?:curl|invoke-webrequest).*(?:\||;).*(?:iex|powershell|pwsh|cmd|bash|sh)",
            re.I,
        ),
        re.compile(r"\b(?:fork\s*bomb|:\(\)\s*\{\s*:\|:\s*&\s*\})", re.I),
    )

    def safe_effect(self, command: str) -> PermissionEffect | None:
        if self._UNSAFE_COMPOSITION.search(command):
            return None
        return (
            PermissionEffect.ALLOW if any(rule.fullmatch(command) for rule in self._SAFE) else None
        )

    def dangerous_effect(self, command: str) -> PermissionEffect | None:
        return (
            PermissionEffect.DENY if any(rule.search(command) for rule in self._DANGEROUS) else None
        )

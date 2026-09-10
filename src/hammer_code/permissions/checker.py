"""The pure five-layer permission decision engine."""

from __future__ import annotations

from hammer_code.permissions.commands import CommandPolicy
from hammer_code.permissions.models import (
    PermissionDecision,
    PermissionEffect,
    PermissionMode,
    PermissionRequest,
)
from hammer_code.permissions.rules import RuleStore


class PermissionChecker:
    def __init__(self, mode: PermissionMode, rules: RuleStore | None = None) -> None:
        self.mode, self.rules, self.commands = mode, rules, CommandPolicy()

    def check(self, request: PermissionRequest) -> PermissionDecision:
        if request.normalized_command is not None:
            if self.commands.safe_effect(request.normalized_command) is PermissionEffect.ALLOW:
                return PermissionDecision(PermissionEffect.ALLOW, "strict built-in safe command")
            if self.commands.dangerous_effect(request.normalized_command) is PermissionEffect.DENY:
                return PermissionDecision(PermissionEffect.DENY, "dangerous command is forbidden")
        try:
            request.workspace_root.resolve(strict=True)
            if any(
                path != request.workspace_root and request.workspace_root not in path.parents
                for path in request.canonical_paths
            ):
                return PermissionDecision(PermissionEffect.DENY, "path is outside workspace root")
        except OSError:
            return PermissionDecision(PermissionEffect.DENY, "path normalization failed")
        if self.rules is not None:
            effect = self.rules.effect_for(request)
            if effect is not None:
                return PermissionDecision(effect, "matched user permission rule")
        if self.mode is PermissionMode.STRICT:
            return PermissionDecision(PermissionEffect.ASK, "strict mode")
        if self.mode is PermissionMode.UNATTENDED:
            return PermissionDecision(PermissionEffect.ALLOW, "unattended mode")
        if request.category == "read":
            return PermissionDecision(PermissionEffect.ALLOW, "read mode default")
        if request.category == "write" and self.mode is PermissionMode.ACCEPT_EDITS:
            return PermissionDecision(PermissionEffect.ALLOW, "accept-edits mode")
        return PermissionDecision(PermissionEffect.ASK, "mode requires approval")

"""Async bridge from ASK decisions to a UI approval port."""

from __future__ import annotations

from typing import Protocol

from hammer_code.errors import PermissionError
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import ApprovalChoice, PermissionEffect, PermissionRequest
from hammer_code.permissions.rules import RuleStore


class ApprovalPort(Protocol):
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice: ...


class PermissionService:
    def __init__(
        self, checker: PermissionChecker, approvals: ApprovalPort, rules: RuleStore | None
    ) -> None:
        self.checker, self.approvals, self.rules = checker, approvals, rules

    async def authorize(self, request: PermissionRequest) -> None:
        decision = self.checker.check(request)
        if decision.effect is PermissionEffect.ALLOW:
            return
        if decision.effect is PermissionEffect.DENY:
            raise PermissionError(decision.reason)
        choice = await self.approvals.approve(request, decision.reason)
        if choice is ApprovalChoice.DENY:
            raise PermissionError("User denied this tool call")
        if choice is ApprovalChoice.ALLOW_AND_PERSIST:
            if self.rules is None:
                raise PermissionError("Persistent approval is unavailable")
            self.rules.persist_allow(request)

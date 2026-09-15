from pathlib import Path

from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import PermissionEffect, PermissionMode, PermissionRequest
from hammer_code.permissions.service import PermissionService


def _request(root: Path, command: str) -> PermissionRequest:
    return PermissionRequest.for_command("shell", command, root, root)


def test_safe_then_danger_then_mode_order(tmp_path: Path) -> None:
    checker = PermissionChecker(PermissionMode.STRICT)
    assert checker.check(_request(tmp_path, "git status")).effect is PermissionEffect.ALLOW
    assert checker.check(_request(tmp_path, "rm -rf .")).effect is PermissionEffect.DENY
    assert checker.check(_request(tmp_path, "pytest")).effect is PermissionEffect.ASK


def test_permission_service_replaces_only_the_runtime_mode() -> None:
    checker = PermissionChecker(PermissionMode.STRICT)
    service = PermissionService(checker, approvals=None, rules=None)  # type: ignore[arg-type]

    service.set_mode(PermissionMode.ACCEPT_EDITS)

    assert service.mode is PermissionMode.ACCEPT_EDITS
    assert checker.mode is PermissionMode.ACCEPT_EDITS

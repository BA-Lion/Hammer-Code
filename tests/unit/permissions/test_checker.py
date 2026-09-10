from pathlib import Path

from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import PermissionEffect, PermissionMode, PermissionRequest


def _request(root: Path, command: str) -> PermissionRequest:
    return PermissionRequest.for_command("shell", command, root, root)


def test_safe_then_danger_then_mode_order(tmp_path: Path) -> None:
    checker = PermissionChecker(PermissionMode.STRICT)
    assert checker.check(_request(tmp_path, "git status")).effect is PermissionEffect.ALLOW
    assert checker.check(_request(tmp_path, "rm -rf .")).effect is PermissionEffect.DENY
    assert checker.check(_request(tmp_path, "pytest")).effect is PermissionEffect.ASK

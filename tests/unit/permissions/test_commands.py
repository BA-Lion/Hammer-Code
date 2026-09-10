import pytest

from hammer_code.permissions.commands import CommandPolicy
from hammer_code.permissions.models import PermissionEffect


@pytest.mark.parametrize(
    "command",
    ["git status", "git status --short --branch", "git --version", "python -V", "uv --version"],
)
def test_strict_safe_commands_are_allowed(command: str) -> None:
    assert CommandPolicy().safe_effect(command) is PermissionEffect.ALLOW


@pytest.mark.parametrize(
    "command",
    ["git status; whoami", "git status | more", "git status > result", "git status $(whoami)"],
)
def test_safe_commands_reject_composition(command: str) -> None:
    assert CommandPolicy().safe_effect(command) is None


@pytest.mark.parametrize(
    "command",
    ["rm -rf .", "Remove-Item -Recurse .", "Format-Volume -DriveLetter C", "diskpart"],
)
def test_dangerous_commands_are_hard_denied(command: str) -> None:
    assert CommandPolicy().dangerous_effect(command) is PermissionEffect.DENY

import json
from pathlib import Path

import pytest

from hammer_code.errors import RuleFileError
from hammer_code.permissions.models import PermissionEffect, PermissionRequest
from hammer_code.permissions.rules import RuleStore


def _command(root: Path, command: str = "pytest -q") -> PermissionRequest:
    return PermissionRequest.for_command("shell", command, root, root)


def test_persisted_shell_allow_is_exact_and_reloads(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    store = RuleStore(root, tmp_path / "local-app-data")
    request = _command(root)

    store.persist_allow(request)

    assert store.effect_for(request) is PermissionEffect.ALLOW
    assert store.effect_for(_command(root, "pytest -x")) is None
    reloaded = RuleStore(root, tmp_path / "local-app-data")
    assert reloaded.effect_for(request) is PermissionEffect.ALLOW


def test_deny_rule_wins_over_allow(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    base = tmp_path / "local-app-data"
    store = RuleStore(root, base)
    assert store.path is not None
    store.path.parent.mkdir(parents=True)
    store.path.write_text(
        "\n".join(
            (
                "version = 1",
                f"workspace = {json.dumps(str(root.resolve()))}",
                "[[allow]]",
                'tool = "shell"',
                'command = "pytest -q"',
                f"cwd = {json.dumps(str(root.resolve()))}",
                "[[deny]]",
                'tool = "shell"',
                'command = "pytest -q"',
                f"cwd = {json.dumps(str(root.resolve()))}",
            )
        ),
        encoding="utf-8",
    )

    assert RuleStore(root, base).effect_for(_command(root)) is PermissionEffect.DENY


@pytest.mark.parametrize(
    "contents",
    (
        "not valid TOML = [",
        "version = 2\nworkspace = $WORKSPACE",
        "version = 1\nworkspace = $WORKSPACE\nunknown = true",
        ('version = 1\nworkspace = $WORKSPACE\n[[allow]]\ntool = "shell"\ncommand_regex = "["'),
    ),
)
def test_invalid_or_mismatched_rule_files_fail_closed(tmp_path: Path, contents: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    store = RuleStore(root, tmp_path / "local-app-data")
    assert store.path is not None
    store.path.parent.mkdir(parents=True)
    store.path.write_text(
        contents.replace("$WORKSPACE", json.dumps(str(root.resolve()))), encoding="utf-8"
    )

    with pytest.raises(RuleFileError):
        RuleStore(root, tmp_path / "local-app-data")

import subprocess
import sys
from pathlib import Path

import pytest

from hammer_code.permissions.paths import PathPolicy


def test_existing_and_new_paths_must_remain_within_canonical_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    existing = root / "existing.txt"
    existing.write_text("ok", encoding="utf-8")
    policy = PathPolicy(root)

    assert policy.canonicalize(existing) == existing.resolve()
    assert policy.canonicalize(root / "new.txt") == (root / "new.txt").resolve()
    with pytest.raises(ValueError, match="outside"):
        policy.canonicalize(root / ".." / "outside.txt")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction semantics")
def test_junction_target_outside_workspace_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    junction = root / "junction"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("The current Windows test environment cannot create a junction")

    with pytest.raises(ValueError, match="outside"):
        PathPolicy(root).canonicalize(junction / "new.txt")

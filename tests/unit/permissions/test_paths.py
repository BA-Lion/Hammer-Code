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

from __future__ import annotations

import hashlib
from pathlib import Path

from hammer_code.skill.catalog import SkillCatalog


def test_hash_and_definition_come_from_identical_read(
    tmp_path: Path, skill_factory, monkeypatch
) -> None:
    definition = skill_factory("review", body="Original payload")
    path = definition.skill_dir / "SKILL.md"
    previous_bytes = path.read_bytes()
    read = Path.read_bytes
    changed = False

    def intercept(current):
        nonlocal changed
        payload = read(current)
        if current == path and not changed:
            changed = True
            path.write_bytes(payload.replace(b"Original payload", b"Changed payload"))
        return payload

    monkeypatch.setattr(Path, "read_bytes", intercept)
    catalog = SkillCatalog(tmp_path)
    snapshot = catalog.build_snapshot()
    identity = (definition.scope, definition.name)
    assert snapshot.effective["review"].prompt_template == "Original payload"
    assert snapshot.content_hashes[identity] == hashlib.sha256(previous_bytes).hexdigest()
    updated = catalog.build_snapshot(snapshot)
    assert updated.effective["review"].prompt_template == "Changed payload"
    assert catalog.build_snapshot(updated) is updated

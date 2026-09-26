"""One-way, whitelist-only development local-state schema migration."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

CURRENT_LOCAL_STATE_SCHEMA = 2


class LocalStateMigrationError(RuntimeError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        return path.is_symlink()
    except OSError as exc:
        raise LocalStateMigrationError("Local state path cannot be inspected") from exc
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _direct_child(parent: Path, child: Path) -> None:
    if child.parent != parent or (child.exists() and _is_reparse(child)):
        raise LocalStateMigrationError("Local state migration target is not a safe direct child")


def _write_marker(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=marker.parent, delete=False, newline="\n"
    )
    try:
        with handle:
            json.dump({"schema_version": CURRENT_LOCAL_STATE_SCHEMA}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, marker)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def migrate_local_state(workspace_root: Path) -> bool:
    """Clear exactly incompatible sessions and derived Skill state, once.

    The function deliberately never lists the harness directory: the two known
    targets are constructed from verified components, and Memory is never read.
    """
    workspace = workspace_root.resolve(strict=True)
    harness = workspace / ".hammer-code"
    marker = harness / "local-state-schema.json"
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalStateMigrationError("Local state schema marker is invalid") from exc
        if payload == {"schema_version": CURRENT_LOCAL_STATE_SCHEMA}:
            return False
    if harness.exists() and _is_reparse(harness):
        raise LocalStateMigrationError("Local state root is a reparse point")
    sessions = harness / "sessions"
    skill = harness / "skill"
    state = skill / "state"
    _direct_child(harness, sessions)
    _direct_child(harness, skill)
    _direct_child(skill, state)
    for target in (sessions, state):
        if target.exists():
            if not target.is_dir():
                raise LocalStateMigrationError("Local state migration target is not a directory")
            shutil.rmtree(target)
    _write_marker(marker)
    return True

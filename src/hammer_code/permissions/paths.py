"""Canonical component-aware project-root path checks."""

from __future__ import annotations

import os
from pathlib import Path


class PathPolicy:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve(strict=True)

    def canonicalize(self, value: str | Path) -> Path:
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else self.workspace_root / candidate
        if candidate.exists():
            canonical = candidate.resolve(strict=True)
        else:
            suffix: list[str] = []
            parent = candidate
            while not parent.exists():
                suffix.append(parent.name)
                parent = parent.parent
            canonical = parent.resolve(strict=True)
            for part in reversed(suffix):
                canonical /= part
        try:
            within = os.path.commonpath((str(self.workspace_root), str(canonical))) == str(
                self.workspace_root
            )
        except ValueError:
            within = False
        if not within:
            raise ValueError("Path is outside the workspace root")
        return canonical

    def verify_before_use(self, value: str | Path) -> Path:
        return self.canonicalize(value)

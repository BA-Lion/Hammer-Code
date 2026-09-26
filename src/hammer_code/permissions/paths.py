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

    @staticmethod
    def _within(root: Path, candidate: Path) -> bool:
        try:
            return os.path.commonpath(
                (os.path.normcase(str(root)), os.path.normcase(str(candidate)))
            ) == os.path.normcase(str(root))
        except ValueError:
            return False

    def _lexical(self, value: str | Path) -> Path:
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else self.workspace_root / candidate
        return Path(os.path.normpath(str(candidate)))

    def protected_kind(self, value: str | Path) -> str | None:
        """Check both user spelling and resolved target for protected-root aliases."""
        lexical = self._lexical(value)
        canonical = self.canonicalize(value)
        roots = {
            "skill": self.workspace_root / ".hammer-code" / "skill",
            "memory": self.workspace_root / ".hammer-code" / "memory",
        }
        for kind, root in roots.items():
            if self._within(root, lexical) or self._within(root.resolve(strict=False), canonical):
                return kind
        return None

    def guarded(self, value: str | Path, operation: str) -> Path:
        kind = self.protected_kind(value)
        if kind == "skill":
            raise ValueError("Skill storage is only available through controlled Skill services")
        if kind == "memory" and operation in {"edit_file", "create_file"}:
            raise ValueError("Memory storage is read-only through general file tools")
        return self.canonicalize(value)

    def verify_before_use(self, value: str | Path) -> Path:
        return self.canonicalize(value)

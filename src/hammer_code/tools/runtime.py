"""Bounded per-session storage for oversized tool results."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from uuid import uuid4


class RuntimeStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root.resolve() / ".hammer-code" / "runtime"
        self.session_id = str(uuid4())
        self.session_dir = self.root / self.session_id
        self.results_dir = self.session_dir / "tool-results"
        self.results_dir.mkdir(parents=True, exist_ok=False)

    def write_result(self, call_id: str, content: str) -> Path:
        digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()
        path = self.results_dir / f"{digest}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def cleanup_old(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if not self.root.exists():
            return
        for child in self.root.iterdir():
            if child == self.session_dir or not child.is_dir() or child.parent != self.root:
                continue
            if now - child.stat().st_mtime > 24 * 60 * 60:
                shutil.rmtree(child)

    def cleanup(self) -> None:
        if self.session_dir.exists() and self.session_dir.parent == self.root:
            shutil.rmtree(self.session_dir)

"""Bounded per-session storage for oversized tool results."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from uuid import uuid4


class RuntimeStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root.resolve() / ".hammer-code" / "tmp"
        self.session_id = str(uuid4())
        self.session_dir = self.root / self.session_id
        self.results_dir = self.session_dir / "tool-results"
        self.results_dir.mkdir(parents=True, exist_ok=False)

    def result_path(self, call_id: str) -> Path:
        path = self.results_dir / f"{hashlib.sha256(call_id.encode('utf-8')).hexdigest()}.txt"
        if path.parent != self.results_dir:
            raise ValueError("Invalid runtime result path")
        return path

    def write_result(self, call_id: str, content: str) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.result_path(call_id)
        path.write_text(content, encoding="utf-8")
        return path

    def delete_results(self, call_ids: tuple[str, ...] | set[str]) -> tuple[str, ...]:
        failed: list[str] = []
        for call_id in call_ids:
            try:
                self.result_path(call_id).unlink(missing_ok=True)
            except OSError:
                failed.append(call_id)
        return tuple(failed)

    def clear_results(self) -> bool:
        try:
            if self.results_dir.exists() and self.results_dir.parent == self.session_dir:
                shutil.rmtree(self.results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def cleanup_old(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if not self.root.exists():
            return
        for child in self.root.iterdir():
            if (
                child == self.session_dir
                or child.parent != self.root
                or not child.is_dir()
                or child.is_symlink()
            ):
                continue
            try:
                if now - child.stat().st_mtime > 24 * 60 * 60:
                    shutil.rmtree(child)
            except OSError:
                continue

    def cleanup(self) -> None:
        if (
            self.session_dir.exists()
            and self.session_dir.parent == self.root
            and not self.session_dir.is_symlink()
        ):
            shutil.rmtree(self.session_dir)

"""Creation, listing, restore selection, and conservative expiry cleanup for sessions."""

from __future__ import annotations

import secrets
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hammer_code.session.models import SessionSummary
from hammer_code.session.session import Session, SessionError


class SessionManager:
    def __init__(self, project_root: Path, *, now: Callable[[], datetime] | None = None) -> None:
        self.project_root = project_root.resolve()
        self.sessions_root = self.project_root / ".hammer-code" / "sessions"
        self._now = now or (lambda: datetime.now(UTC))
        self.warnings: list[str] = []

    def create(self, protocol: str) -> Session:
        for _ in range(100):
            value = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S") + "-" + secrets.token_hex(2)
            try:
                return Session.create(self.sessions_root, value, protocol)
            except FileExistsError:
                continue
        raise SessionError("Unable to allocate a unique session id")

    def list(self, protocol: str | None = None) -> tuple[SessionSummary, ...]:
        if not self.sessions_root.exists():
            return ()
        summaries: list[SessionSummary] = []
        for directory in self.sessions_root.iterdir():
            try:
                session = Session.open(self.sessions_root, directory)
                if protocol is None or session.meta.protocol == protocol:
                    summaries.append(
                        SessionSummary(
                            session.meta.id,
                            session.meta.title,
                            session.meta.protocol,
                            session.meta.last_active,
                        )
                    )
            except (OSError, SessionError):
                self.warnings.append("Ignored a session with invalid metadata or path identity.")
        return tuple(
            sorted(summaries, key=lambda value: (value.last_active, value.id), reverse=True)
        )

    def open(self, session_id: str, protocol: str) -> Session:
        if session_id == "latest":
            items = self.list(protocol)
            if not items:
                raise SessionError("No compatible sessions are available to resume")
            session_id = items[0].id
        session = Session.open(self.sessions_root, self.sessions_root / f"session-{session_id}")
        if session.meta.protocol != protocol:
            raise SessionError("Session protocol does not match the selected profile")
        return session

    def cleanup(self) -> None:
        if not self.sessions_root.exists():
            return
        threshold = self._now().astimezone(UTC) - timedelta(days=30)
        for directory in self.sessions_root.iterdir():
            try:
                session = Session.open(self.sessions_root, directory)
                if session.meta.last_active.astimezone(UTC) < threshold:
                    # Session.open has already resolved and identity-checked this direct child.
                    shutil.rmtree(session.directory)
            except (OSError, SessionError):
                self.warnings.append("Skipped unsafe or invalid session during cleanup.")

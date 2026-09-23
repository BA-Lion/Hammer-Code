"""One-shot Assignment-addressed file mailboxes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class MailboxMessage:
    id: str
    sender_assignment_id: str
    recipient_assignment_id: str
    created_at: str
    body: str


def write_message(run_path: Path, sender_id: str, recipient_id: str, body: str) -> MailboxMessage:
    if not 1 <= len(body) <= 16_000:
        raise ValueError("message body is invalid")
    mailbox = run_path / "assignments" / recipient_id / "mailbox"
    if not mailbox.is_dir() or mailbox.is_symlink():
        raise ValueError("recipient mailbox is unavailable")
    if len(tuple(mailbox.glob("*.json"))) >= 256:
        raise ValueError("recipient mailbox is full")
    message = MailboxMessage(
        str(uuid4()), sender_id, recipient_id, datetime.now(UTC).isoformat(), body
    )
    temporary = mailbox / f".{uuid4().hex}.tmp"
    target = mailbox / f"{message.id}.json"
    with open(_native_path(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(message.__dict__, ensure_ascii=False, separators=(",", ":")))
    os.replace(_native_path(temporary), _native_path(target))
    return message


def receive_messages(run_path: Path, recipient_id: str) -> tuple[MailboxMessage, ...]:
    mailbox = run_path / "assignments" / recipient_id / "mailbox"
    if not mailbox.is_dir() or mailbox.is_symlink():
        raise ValueError("recipient mailbox is unavailable")
    values: list[MailboxMessage] = []
    for path in sorted(mailbox.glob("*.json"), key=lambda item: item.name):
        try:
            with open(_native_path(path), encoding="utf-8") as handle:
                raw = json.loads(handle.read())
            value = MailboxMessage(**raw)
            if value.recipient_assignment_id != recipient_id or not 1 <= len(value.body) <= 16_000:
                raise ValueError("message is invalid")
        except Exception as exc:
            raise ValueError("mailbox contains an invalid message") from exc
        os.unlink(_native_path(path))
        values.append(value)
    return tuple(values)


def _native_path(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" and not value.startswith("\\\\?\\") else value

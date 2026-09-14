"""One durable session and the application-facing persistence coordinator."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.messages import Message, Role, TextBlock, ToolResultBlock
from hammer_code.errors import HammerCodeError
from hammer_code.session.models import RecordType, RestoreResult, SessionMeta
from hammer_code.session.serialization import encode_record, records_for_turn, scan_records

_SESSION_ID = re.compile(r"\A\d{4}-\d{2}-\d{2}-\d{6}-[a-z0-9]{4}\Z")


class SessionError(HammerCodeError):
    pass


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file using a sibling temporary file."""
    if path.exists() and (not path.is_file() or _is_reparse(path)):
        raise SessionError("Refusing to replace a non-regular session file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def validate_session_directory(sessions_root: Path, directory: Path, meta: SessionMeta) -> Path:
    """Return a verified direct child session directory or reject it safely."""
    root = sessions_root.resolve(strict=True)
    if _is_reparse(directory) or not directory.is_dir():
        raise SessionError("Session directory has an unsafe identity")
    resolved = directory.resolve(strict=True)
    if (
        resolved.parent != root
        or directory.name != f"session-{meta.id}"
        or not _SESSION_ID.fullmatch(meta.id)
    ):
        raise SessionError("Session directory identity is invalid")
    return resolved


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class Session:
    def __init__(self, directory: Path, meta: SessionMeta) -> None:
        self.directory = directory
        self.meta = meta
        self.persistence_degraded = False

    @property
    def messages_path(self) -> Path:
        return self.directory / "messages.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    @classmethod
    def create(cls, sessions_root: Path, session_id: str, protocol: str) -> Session:
        if not _SESSION_ID.fullmatch(session_id):
            raise SessionError("Session id has an invalid format")
        root = sessions_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / f"session-{session_id}"
        if directory.exists():
            raise FileExistsError(directory)
        now = datetime.now(UTC)
        meta = SessionMeta(
            id=session_id,
            protocol=protocol,
            created_at=now,
            last_active=now,
        )
        try:
            directory.mkdir()
            atomic_write_text(directory / "messages.jsonl", "")
            session = cls(directory, meta)
            session._write_meta(meta)
            return session
        except Exception:
            # Do not recursively delete an unknown or partially replaced target.
            raise

    @classmethod
    def open(cls, sessions_root: Path, directory: Path) -> Session:
        try:
            raw = (directory / "meta.json").read_text(encoding="utf-8")
            meta = SessionMeta.model_validate_json(raw)
            resolved = validate_session_directory(sessions_root, directory, meta)
        except Exception as exc:
            raise SessionError("Session metadata or directory identity is invalid") from exc
        return cls(resolved, meta)

    def _write_meta(self, meta: SessionMeta) -> None:
        atomic_write_text(
            self.meta_path, meta.model_dump_json(exclude_none=False, indent=None) + "\n"
        )
        self.meta = meta

    def load(self) -> RestoreResult:
        try:
            data = self.messages_path.read_bytes()
        except OSError as exc:
            raise SessionError("Unable to read session messages") from exc
        result = scan_records(data)
        repaired = result.safe_byte_boundary != len(data)
        degraded = False
        if repaired:
            try:
                atomic_write_text(
                    self.messages_path,
                    data[: result.safe_byte_boundary].decode("utf-8"),
                )
            except (OSError, UnicodeDecodeError, SessionError):
                degraded = True
                self.persistence_degraded = True
        cursor = min(self.meta.memory_cursor, result.latest_durable_turn)
        if cursor != self.meta.memory_cursor:
            try:
                self._write_meta(self.meta.model_copy(update={"memory_cursor": cursor}))
            except OSError:
                degraded = True
                self.persistence_degraded = True
        return RestoreResult(
            result.messages,
            result.records,
            result.latest_durable_turn,
            result.completed_turns,
            result.safe_byte_boundary,
            repaired,
            degraded,
        )

    def append(self, records: tuple, *, total_tokens: int, title: str | None = None) -> None:
        payload = b"".join(encode_record(record) for record in records)
        try:
            with self.messages_path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            count = self.meta.message_count + len(records)
            changes: dict[str, object] = {
                "message_count": count,
                "last_active": datetime.now(UTC),
                "total_tokens": total_tokens,
            }
            if title is not None:
                changes["title"] = title
            self._write_meta(self.meta.model_copy(update=changes))
        except Exception:
            self.persistence_degraded = True
            raise

    def rewrite(
        self, records: tuple, *, memory_cursor: int | None = None, total_tokens: int | None = None
    ) -> None:
        payload = b"".join(encode_record(record) for record in records).decode("utf-8")
        try:
            atomic_write_text(self.messages_path, payload)
            changes: dict[str, object] = {
                "message_count": len(records),
                "last_active": datetime.now(UTC),
            }
            if memory_cursor is not None:
                changes["memory_cursor"] = memory_cursor
            if total_tokens is not None:
                changes["total_tokens"] = total_tokens
            self._write_meta(self.meta.model_copy(update=changes))
            self.persistence_degraded = False
        except Exception:
            self.persistence_degraded = True
            raise

    def compare_replace_meta(self, expected_cursor: int, new_cursor: int) -> bool:
        if self.meta.memory_cursor != expected_cursor:
            return False
        try:
            current = SessionMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))
            if current.memory_cursor != expected_cursor:
                return False
            self._write_meta(current.model_copy(update={"memory_cursor": new_cursor}))
            return True
        except Exception:
            self.persistence_degraded = True
            return False

    def touch(self) -> None:
        try:
            self._write_meta(self.meta.model_copy(update={"last_active": datetime.now(UTC)}))
        except Exception:
            self.persistence_degraded = True


class SessionCoordinator:
    """Maps committed Conversation history to one session without owning conversation state."""

    def __init__(self, session: Session, manager: ConversationManager) -> None:
        self.session = session
        self.manager = manager
        self._turn_positions: list[tuple[Message, ...]] = []
        self._records: list = []
        self._next_turn_index = 1

    @property
    def persistence_degraded(self) -> bool:
        return self.session.persistence_degraded

    @classmethod
    def restored(
        cls, session: Session, manager: ConversationManager, result: RestoreResult
    ) -> SessionCoordinator:
        value = cls(session, manager)
        value._records = list(result.records)
        value._rebuild_positions(result.messages)
        value._next_turn_index = result.latest_durable_turn + 1
        return value

    async def append_turn(self, messages: tuple[Message, ...], *, completed: bool) -> None:
        if not messages:
            return
        position = self._next_turn_index
        records = records_for_turn(messages, position)
        try:
            title = self._title_for(messages) if not self.session.meta.title else None
            await asyncio.to_thread(
                self.session.append,
                records,
                total_tokens=self._total_tokens(),
                title=title,
            )
            self._records.extend(records)
            self._turn_positions.append(messages)
            self._next_turn_index += 1
            if completed:
                self._sync_total()
        except Exception:
            self.session.persistence_degraded = True

    async def rewrite_after_compaction(
        self,
        before: tuple[Message, ...],
        after: tuple[Message, ...],
        *,
        summary: str,
        summarized_turn_positions: tuple[int, ...],
    ) -> None:
        del before
        if not summary.strip() or not summarized_turn_positions:
            self.session.persistence_degraded = True
            return
        summary_turn = max(summarized_turn_positions)
        from hammer_code.session.models import SessionRecord

        compression = SessionRecord(
            type=RecordType.COMPRESSION,
            content=[{"type": "text", "text": summary.strip()}],
            timestamp=datetime.now(UTC),
            turn_index=summary_turn,
            tool_use_id=None,
            is_error=False,
        )
        # Reprojecting `after` preserves only the new committed history; the compression record
        # replaces the summarized range and its turn position remains monotonic.
        try:
            positions = self._split_turns(after)
            if len(positions) != 3:
                raise SessionError("Compaction projection has an invalid turn structure")
            records: list = []
            records.extend(records_for_turn(positions[0], 1))
            records.append(compression)
            records.extend(records_for_turn(positions[2], self._next_turn_index - 1))
            await asyncio.to_thread(
                self.session.rewrite, tuple(records), total_tokens=self._total_tokens()
            )
            self._records = records
            self._turn_positions = positions
            self._next_turn_index = max(record.turn_index for record in records) + 1
        except Exception:
            self.session.persistence_degraded = True

    async def clear_history(self) -> None:
        try:
            await asyncio.to_thread(self.session.rewrite, (), memory_cursor=0, total_tokens=0)
            self._records.clear()
            self._turn_positions.clear()
            self._next_turn_index = 1
        except Exception:
            self.session.persistence_degraded = True

    async def retry_full_rewrite(self) -> None:
        if self.manager.conversation is None:
            return
        try:
            history = self.manager.snapshot_committed()
            positions = self._split_turns(history)
            # When the current Conversation still matches the coordinator projection, reuse the
            # durable records verbatim. Reprojecting a compacted summary would turn its COMPRESSION
            # record into ordinary USER/ASSISTANT records and renumber the durable turn indices.
            records = (
                tuple(self._records)
                if positions == self._turn_positions
                else self._records_for_history(history)
            )
            await asyncio.to_thread(
                self.session.rewrite, records, total_tokens=self._total_tokens()
            )
            self._records = list(records)
            self._turn_positions = positions
            self._next_turn_index = max((record.turn_index for record in records), default=0) + 1
        except Exception:
            self.session.persistence_degraded = True

    async def close(self) -> None:
        await self.retry_full_rewrite()

    def _sync_total(self) -> None:
        # A later full rewrite carries usage totals. Append does not reopen Meta after success.
        return None

    @staticmethod
    def _title_for(messages: tuple[Message, ...]) -> str:
        for message in messages:
            if message.role is Role.USER:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        return " ".join(block.text.split())[:50]
        return ""

    def _total_tokens(self) -> int:
        if self.manager.conversation is None:
            return self.session.meta.total_tokens
        return self.manager.conversation.usage_ledger.for_conversation().total_tokens or 0

    def _records_for_history(self, messages: tuple[Message, ...]) -> tuple:
        return tuple(
            record
            for position, turn in enumerate(self._split_turns(messages), start=1)
            for record in records_for_turn(turn, position)
        )

    def _rebuild_positions(self, messages: tuple[Message, ...]) -> None:
        self._turn_positions = self._split_turns(messages)

    @staticmethod
    def _split_turns(messages: tuple[Message, ...]) -> list[tuple[Message, ...]]:
        turns: list[list[Message]] = []
        for message in messages:
            starts = message.role is Role.USER and any(
                not isinstance(block, ToolResultBlock) for block in message.content
            )
            if starts:
                turns.append([message])
            elif turns:
                turns[-1].append(message)
        return [tuple(turn) for turn in turns]

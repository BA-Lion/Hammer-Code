from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.session.manager import SessionManager
from hammer_code.session.models import RecordType
from hammer_code.session.serialization import encode_record, records_for_turn, scan_records


def test_records_round_trip_a_completed_tool_turn() -> None:
    messages = (
        Message(Role.USER, (TextBlock("inspect this"),)),
        Message(
            Role.ASSISTANT,
            (ToolCallBlock("call-1", "read_file", {"path": "a.py"}, '{"path":"a.py"}'),),
        ),
        Message(Role.USER, (ToolResultBlock("call-1", (TextBlock("content"),), False),)),
        Message(Role.ASSISTANT, (TextBlock("done"),)),
    )
    records = records_for_turn(messages, 1, datetime(2026, 9, 14, tzinfo=UTC))
    restored = scan_records(b"".join(encode_record(record) for record in records))

    assert [record.type for record in records] == [
        RecordType.USER,
        RecordType.ASSISTANT,
        RecordType.TOOL_RESULT,
        RecordType.ASSISTANT,
    ]
    assert restored.messages == messages
    assert restored.completed_turns == frozenset({1})


def test_scan_discards_incomplete_tool_chain() -> None:
    messages = (
        Message(Role.USER, (TextBlock("inspect this"),)),
        Message(
            Role.ASSISTANT,
            (ToolCallBlock("call-1", "read_file", {"path": "a.py"}, '{"path":"a.py"}'),),
        ),
    )
    data = b"".join(encode_record(record) for record in records_for_turn(messages, 1))
    restored = scan_records(data)

    assert restored.messages == ()
    assert restored.records == ()
    assert restored.safe_byte_boundary == 0


def test_session_manager_cleanup_is_strict_at_thirty_day_boundary(tmp_path) -> None:
    now = datetime(2026, 9, 14, tzinfo=UTC)
    manager = SessionManager(tmp_path, now=lambda: now)
    keep = manager.create("openai")
    remove = manager.create("openai")
    keep._write_meta(keep.meta.model_copy(update={"last_active": now - timedelta(days=30)}))
    remove._write_meta(
        remove.meta.model_copy(update={"last_active": now - timedelta(days=30, seconds=1)})
    )

    manager.cleanup()

    assert keep.directory.exists()
    assert not remove.directory.exists()

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.prompts import COMPACT_BOUNDARY_MESSAGE
from hammer_code.session.manager import SessionManager
from hammer_code.session.models import RecordType
from hammer_code.session.serialization import encode_record, records_for_turn, scan_records
from hammer_code.session.session import SessionCoordinator


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


@pytest.mark.asyncio
async def test_compaction_record_survives_final_full_rewrite(tmp_path) -> None:
    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_chat_completions",
                    "model": "model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "KEY",
                    "max_output_tokens": 100,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    resolved = resolve_profile(config, None, {"KEY": "secret"})
    conversation = ConversationManager()
    conversation.create(resolved)
    session = SessionManager(tmp_path).create(resolved.profile.protocol)
    coordinator = SessionCoordinator(session, conversation)
    for index in range(1, 6):
        turn = conversation.begin_turn(f"user {index}")
        committed = conversation.commit(
            turn, Message(Role.ASSISTANT, (TextBlock(f"assistant {index}"),))
        )
        await coordinator.append_turn(committed, completed=True)
    assert session.compare_replace_meta(0, 4)

    before = conversation.snapshot_committed()
    summary = "Durable synthetic summary"
    after = (
        *before[:2],
        Message(Role.USER, (TextBlock(f"[Conversation summary]\n{summary}"),)),
        Message(Role.ASSISTANT, (TextBlock(COMPACT_BOUNDARY_MESSAGE),)),
        *before[-2:],
    )
    conversation.replace_committed_history(before, after)
    await coordinator.rewrite_after_compaction(
        before,
        after,
        summary=summary,
        summarized_turn_positions=(2, 3, 4),
    )

    compacted = session.load()
    assert [record.type for record in compacted.records] == [
        RecordType.USER,
        RecordType.ASSISTANT,
        RecordType.COMPRESSION,
        RecordType.USER,
        RecordType.ASSISTANT,
    ]
    assert [record.turn_index for record in compacted.records] == [1, 1, 4, 5, 5]
    assert compacted.messages == after

    await coordinator.close()

    reopened = session.load()
    assert [record.type for record in reopened.records] == [
        RecordType.USER,
        RecordType.ASSISTANT,
        RecordType.COMPRESSION,
        RecordType.USER,
        RecordType.ASSISTANT,
    ]
    assert [record.turn_index for record in reopened.records] == [1, 1, 4, 5, 5]
    assert reopened.messages == after
    assert session.meta.memory_cursor == 4

"""Pure strict conversion between domain messages and session JSONL records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from hammer_code.domain.messages import (
    Message,
    ProviderStateBlock,
    ReasoningBlock,
    ReasoningVisibility,
    RefusalBlock,
    Role,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from hammer_code.prompts import COMPACT_BOUNDARY_MESSAGE
from hammer_code.session.models import RecordType, RestoreResult, SessionRecord


class SessionSerializationError(ValueError):
    """Malformed persisted data or an invalid Message-to-record projection."""


def _json_value(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionSerializationError("Block JSON data must be JSON values") from exc
    return value


def encode_block(block: object) -> dict[str, object]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ReasoningBlock):
        return {"type": "reasoning", "text": block.text, "visibility": block.visibility.value}
    if isinstance(block, ProviderStateBlock):
        return {
            "type": "provider_state",
            "protocol": block.protocol,
            "kind": block.kind,
            "data": _json_value(dict(block.data)),
        }
    if isinstance(block, RefusalBlock):
        return {"type": "refusal", "reason": block.reason}
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_call",
            "call_id": block.call_id,
            "name": block.name,
            "arguments": _json_value(dict(block.arguments)),
            "raw_arguments": block.raw_arguments,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "call_id": block.call_id,
            "content": [encode_block(item) for item in block.content],
            "is_error": block.is_error,
        }
    raise SessionSerializationError("Unknown content block type")


def _object(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionSerializationError("Serialized content block must be an object")
    return value


def _only(value: Mapping[str, Any], *names: str) -> None:
    if set(value) != set(names):
        raise SessionSerializationError("Serialized content block has unknown or missing fields")


def decode_block(value: object) -> object:
    data = _object(value)
    block_type = data.get("type")
    try:
        if block_type == "text":
            _only(data, "type", "text")
            return TextBlock(_string(data, "text"))
        if block_type == "reasoning":
            _only(data, "type", "text", "visibility")
            return ReasoningBlock(
                _string(data, "text"), ReasoningVisibility(_string(data, "visibility"))
            )
        if block_type == "provider_state":
            _only(data, "type", "protocol", "kind", "data")
            raw = data.get("data")
            if not isinstance(raw, Mapping):
                raise SessionSerializationError("Provider state data must be an object")
            _json_value(dict(raw))
            return ProviderStateBlock(_string(data, "protocol"), _string(data, "kind"), dict(raw))
        if block_type == "refusal":
            _only(data, "type", "reason")
            return RefusalBlock(_string(data, "reason"))
        if block_type == "tool_call":
            _only(data, "type", "call_id", "name", "arguments", "raw_arguments")
            raw = data.get("arguments")
            if not isinstance(raw, Mapping):
                raise SessionSerializationError("Tool arguments must be an object")
            _json_value(dict(raw))
            return ToolCallBlock(
                _string(data, "call_id"),
                _string(data, "name"),
                dict(raw),
                _string(data, "raw_arguments"),
            )
        if block_type == "tool_result":
            _only(data, "type", "call_id", "content", "is_error")
            raw = data.get("content")
            if not isinstance(raw, list) or not raw or not isinstance(data.get("is_error"), bool):
                raise SessionSerializationError(
                    "Tool result needs non-empty text content and is_error"
                )
            content = tuple(decode_block(item) for item in raw)
            if not all(isinstance(item, TextBlock) for item in content):
                raise SessionSerializationError("Tool result content may only contain text blocks")
            return ToolResultBlock(
                _string(data, "call_id"), cast(tuple[TextBlock, ...], content), data["is_error"]
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SessionSerializationError):
            raise
        raise SessionSerializationError("Invalid serialized content block") from exc
    raise SessionSerializationError("Unknown serialized content block type")


def _string(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str):
        raise SessionSerializationError(f"{name} must be a string")
    return value


def _validate_record(record: SessionRecord) -> tuple[object, ...]:
    try:
        blocks = tuple(decode_block(item) for item in record.content)
    except (TypeError, ValueError) as exc:
        raise SessionSerializationError("Record contains invalid blocks") from exc
    if not blocks:
        raise SessionSerializationError("Record content must not be empty")
    if record.type is RecordType.USER:
        if (
            any(isinstance(item, ToolResultBlock) for item in blocks)
            or record.tool_use_id is not None
            or record.is_error
        ):
            raise SessionSerializationError("Invalid user record fields")
    elif record.type is RecordType.ASSISTANT:
        calls = [item for item in blocks if isinstance(item, ToolCallBlock)]
        if any(isinstance(item, ToolResultBlock) for item in blocks) or len(
            {item.call_id for item in calls}
        ) != len(calls):
            raise SessionSerializationError("Invalid assistant record blocks")
        if record.tool_use_id is not None or record.is_error:
            raise SessionSerializationError("Invalid assistant record fields")
    elif record.type is RecordType.TOOL_RESULT:
        if len(blocks) != 1 or not isinstance(blocks[0], ToolResultBlock):
            raise SessionSerializationError("Tool result record requires one ToolResultBlock")
        if record.tool_use_id != blocks[0].call_id or record.is_error != blocks[0].is_error:
            raise SessionSerializationError("Tool result redundant fields disagree")
    elif record.type is RecordType.COMPRESSION:
        if len(blocks) != 1 or not isinstance(blocks[0], TextBlock):
            raise SessionSerializationError("Compression record requires one text block")
        if "<analysis>" in blocks[0].text or "<summary>" in blocks[0].text:
            raise SessionSerializationError("Compression text must be summary body only")
        if record.tool_use_id is not None or record.is_error:
            raise SessionSerializationError("Invalid compression record fields")
    return blocks


def records_for_turn(
    messages: tuple[Message, ...], turn_index: int, timestamp: datetime | None = None
) -> tuple[SessionRecord, ...]:
    if turn_index < 1:
        raise SessionSerializationError("turn_index must be positive")
    now = timestamp or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise SessionSerializationError("Session timestamps must include a timezone")
    records: list[SessionRecord] = []
    for message in messages:
        if message.role is Role.USER:
            tools = [block for block in message.content if isinstance(block, ToolResultBlock)]
            if tools:
                if len(tools) != len(message.content):
                    raise SessionSerializationError(
                        "User messages cannot mix tool results with other blocks"
                    )
                records.extend(
                    SessionRecord(
                        type=RecordType.TOOL_RESULT,
                        content=[encode_block(block)],
                        timestamp=now,
                        turn_index=turn_index,
                        tool_use_id=block.call_id,
                        is_error=block.is_error,
                    )
                    for block in tools
                )
            else:
                records.append(
                    SessionRecord(
                        type=RecordType.USER,
                        content=[encode_block(block) for block in message.content],
                        timestamp=now,
                        turn_index=turn_index,
                    )
                )
        elif message.role is Role.ASSISTANT:
            records.append(
                SessionRecord(
                    type=RecordType.ASSISTANT,
                    content=[encode_block(block) for block in message.content],
                    timestamp=now,
                    turn_index=turn_index,
                )
            )
        else:
            raise SessionSerializationError("Unknown message role")
    for record in records:
        _validate_record(record)
    return tuple(records)


def _parse_line(line: bytes) -> SessionRecord:
    try:
        raw = json.loads(line.decode("utf-8"))
        return SessionRecord.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise SessionSerializationError("Invalid session JSONL record") from exc


def scan_records(data: bytes) -> RestoreResult:
    """Return only the complete, structurally safe JSONL prefix without accessing disk."""
    records: list[SessionRecord] = []
    messages: list[Message] = []
    completed: set[int] = set()
    boundary = 0
    previous_turn = 0
    pending_calls: set[str] | None = None
    pending_results: list[ToolResultBlock] = []
    pending_turn = 0
    offset = 0
    for raw in data.splitlines(keepends=True):
        end = offset + len(raw)
        if not raw.endswith(b"\n"):
            break
        try:
            record = _parse_line(raw[:-1])
            blocks = _validate_record(record)
            if record.turn_index < previous_turn:
                raise SessionSerializationError("turn_index must not decrease")
            if pending_calls is not None and record.type is not RecordType.TOOL_RESULT:
                if set(item.call_id for item in pending_results) != pending_calls:
                    raise SessionSerializationError("Tool calls are not fully matched")
                messages.append(Message(Role.USER, tuple(pending_results)))
                pending_calls = None
                pending_results = []
                completed.add(pending_turn)
            if record.type is RecordType.TOOL_RESULT:
                if pending_calls is None or record.turn_index != pending_turn:
                    raise SessionSerializationError("Unexpected tool result")
                item = blocks[0]
                assert isinstance(item, ToolResultBlock)
                if item.call_id not in pending_calls or any(
                    x.call_id == item.call_id for x in pending_results
                ):
                    raise SessionSerializationError("Invalid or duplicate tool result")
                pending_results.append(item)
            else:
                if pending_calls is not None:
                    raise SessionSerializationError("Unclosed tool calls")
                if record.type is RecordType.COMPRESSION:
                    item = blocks[0]
                    assert isinstance(item, TextBlock)
                    messages.extend(
                        (
                            Message(
                                Role.USER, (TextBlock(f"[Conversation summary]\n{item.text}"),)
                            ),
                            Message(Role.ASSISTANT, (TextBlock(COMPACT_BOUNDARY_MESSAGE),)),
                        )
                    )
                    completed.add(record.turn_index)
                else:
                    role = Role.USER if record.type is RecordType.USER else Role.ASSISTANT
                    messages.append(Message(role, cast(tuple, blocks)))
                    calls = {item.call_id for item in blocks if isinstance(item, ToolCallBlock)}
                    if calls:
                        pending_calls, pending_turn = calls, record.turn_index
                    elif record.type is RecordType.ASSISTANT:
                        completed.add(record.turn_index)
            records.append(record)
            previous_turn = record.turn_index
            boundary = end
            offset = end
        except SessionSerializationError:
            break
    if pending_calls is not None:
        # Exclude the unmatched assistant record and every prior record in its incomplete suffix.
        while records and records[-1].turn_index == pending_turn:
            records.pop()
        messages = _messages_for_closed_records(tuple(records))
        completed.discard(pending_turn)
        boundary = sum(_line_length(record) for record in records)
    latest = max((record.turn_index for record in records), default=0)
    return RestoreResult(
        tuple(messages),
        tuple(records),
        latest,
        frozenset(completed),
        boundary,
        boundary != len(data),
        False,
    )


def _messages_for_closed_records(records: tuple[SessionRecord, ...]) -> list[Message]:
    """Small recursive-free replay used when an EOF leaves an open tool chain."""
    payload = b"".join(_encode_record(record) for record in records)
    result = scan_records(payload)
    return list(result.messages)


def _encode_record(record: SessionRecord) -> bytes:
    return (
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _line_length(record: SessionRecord) -> int:
    return len(_encode_record(record))


def records_to_messages(records: tuple[SessionRecord, ...]) -> tuple[Message, ...]:
    return scan_records(b"".join(_encode_record(record) for record in records)).messages


def encode_record(record: SessionRecord) -> bytes:
    _validate_record(record)
    return _encode_record(record)

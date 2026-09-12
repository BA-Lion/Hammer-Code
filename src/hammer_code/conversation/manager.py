"""Atomic in-memory conversation history and usage management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.domain.usage import TokenUsage, UsageLedger
from hammer_code.errors import ConversationBusyError, ConversationError, InvalidTurnStateError


class TurnStatus(StrEnum):
    OPEN = "open"
    COMMITTED = "committed"
    ABORTED = "aborted"
    INTERRUPTED = "interrupted"


@dataclass
class TurnTransaction:
    id: str
    user_message: Message
    request_ids: list[str] = field(default_factory=list)
    staged_messages: list[Message] = field(default_factory=list)
    status: TurnStatus = TurnStatus.OPEN


@dataclass
class Conversation:
    id: str
    profile_name: str
    protocol: str
    model: str
    messages: list[Message] = field(default_factory=list)
    usage_ledger: UsageLedger = field(default_factory=UsageLedger)


class ConversationManager:
    def __init__(self) -> None:
        self.conversation: Conversation | None = None
        self._active_turn: TurnTransaction | None = None

    def create(self, profile_identity: object) -> Conversation:
        if self._active_turn is not None:
            raise ConversationBusyError("Cannot replace a conversation during an active turn")
        name, protocol, model = self._identity(profile_identity)
        self.conversation = Conversation(str(uuid4()), name, protocol, model)
        return self.conversation

    def begin_turn(self, user_text: str) -> TurnTransaction:
        if not user_text.strip():
            raise ConversationError("Message must not be empty")
        if self.conversation is None:
            raise ConversationError("Create a conversation before beginning a turn")
        if self._active_turn is not None:
            raise ConversationBusyError("A conversation turn is already active")
        user_message = Message(Role.USER, (TextBlock(user_text),))
        turn = TurnTransaction(str(uuid4()), user_message, staged_messages=[user_message])
        self._active_turn = turn
        return turn

    def snapshot_for_request(self, turn: TurnTransaction) -> tuple[Message, ...]:
        self._assert_active(turn)
        assert self.conversation is not None
        return tuple(self.conversation.messages) + tuple(turn.staged_messages)

    def snapshot_committed(self) -> tuple[Message, ...]:
        if self.conversation is None:
            raise ConversationError("Create a conversation before reading its history")
        return tuple(self.conversation.messages)

    def snapshot_staged(self, turn: TurnTransaction) -> tuple[Message, ...]:
        self._assert_active(turn)
        return tuple(turn.staged_messages)

    def replace_committed_history(
        self, expected: tuple[Message, ...], replacement: tuple[Message, ...]
    ) -> None:
        if self.conversation is None:
            raise ConversationError("Create a conversation before replacing its history")
        if tuple(self.conversation.messages) != expected:
            raise ConversationBusyError("Conversation history changed during context compaction")
        if any(not isinstance(message, Message) or not message.content for message in replacement):
            raise ConversationError("Replacement history must contain non-empty messages")
        self.conversation.messages[:] = replacement

    def record_maintenance_usage(
        self, operation_id: str, request_id: str, usage: TokenUsage
    ) -> None:
        if self.conversation is None:
            raise ConversationError("Create a conversation before recording usage")
        self.conversation.usage_ledger.upsert(request_id, operation_id, usage)

    def record_usage(self, turn: TurnTransaction, request_id: str, usage: TokenUsage) -> None:
        self._assert_active(turn)
        assert self.conversation is not None
        if request_id not in turn.request_ids:
            turn.request_ids.append(request_id)
        self.conversation.usage_ledger.upsert(request_id, turn.id, usage)

    def commit(self, turn: TurnTransaction, assistant_message: Message) -> None:
        self._assert_active(turn)
        if assistant_message.role is not Role.ASSISTANT:
            raise ConversationError("Only an assistant message can be committed")
        assert self.conversation is not None
        self.conversation.messages.extend((*turn.staged_messages, assistant_message))
        turn.status = TurnStatus.COMMITTED
        self._active_turn = None

    def abort(self, turn: TurnTransaction) -> None:
        self._assert_active(turn)
        turn.status = TurnStatus.ABORTED
        self._active_turn = None

    def stage_tool_call(self, turn: TurnTransaction, assistant_message: Message) -> None:
        """Stage a completed tool-call response without exposing partial model output."""
        self._assert_active(turn)
        if assistant_message.role is not Role.ASSISTANT:
            raise ConversationError("Tool calls must be staged in an assistant message")
        calls = [block for block in assistant_message.content if isinstance(block, ToolCallBlock)]
        if not calls or len({call.call_id for call in calls}) != len(calls):
            raise ConversationError("Tool call message needs unique tool call ids")
        turn.staged_messages.append(assistant_message)

    def stage_tool_results(self, turn: TurnTransaction, user_message: Message) -> None:
        """Stage exactly one result per call from the immediately preceding tool response."""
        self._assert_active(turn)
        if user_message.role is not Role.USER or not turn.staged_messages:
            raise ConversationError("Tool results must follow a tool-call assistant message")
        preceding = turn.staged_messages[-1]
        call_ids = {
            block.call_id for block in preceding.content if isinstance(block, ToolCallBlock)
        }
        results = [block for block in user_message.content if isinstance(block, ToolResultBlock)]
        if (
            not call_ids
            or len(results) != len(user_message.content)
            or {item.call_id for item in results} != call_ids
        ):
            raise ConversationError("Tool results must exactly match the preceding tool call ids")
        turn.staged_messages.append(user_message)

    def interrupt(self, turn: TurnTransaction) -> None:
        """Commit complete tool exchanges after an interrupted later model request."""
        self._assert_active(turn)
        has_exchange = any(
            previous.role is Role.ASSISTANT
            and any(isinstance(block, ToolCallBlock) for block in previous.content)
            and following.role is Role.USER
            and all(isinstance(block, ToolResultBlock) for block in following.content)
            for previous, following in zip(
                turn.staged_messages, turn.staged_messages[1:], strict=False
            )
        )
        if not has_exchange:
            self.abort(turn)
            return
        assert self.conversation is not None
        self.conversation.messages.extend(turn.staged_messages)
        turn.status = TurnStatus.INTERRUPTED
        self._active_turn = None

    def clear(self) -> None:
        if self._active_turn is not None:
            raise ConversationBusyError("Cannot clear a conversation during an active turn")
        if self.conversation is not None:
            self.conversation.messages.clear()
            self.conversation.usage_ledger.clear()

    def _assert_active(self, turn: TurnTransaction) -> None:
        if self._active_turn is not turn:
            raise InvalidTurnStateError("Turn is not the active transaction")
        if turn.status is not TurnStatus.OPEN:
            raise InvalidTurnStateError(f"Turn is already {turn.status.value}")

    @staticmethod
    def _identity(value: object) -> tuple[str, str, str]:
        try:
            return value.name, value.profile.protocol, value.profile.model  # type: ignore[union-attr]
        except AttributeError:
            try:
                return value.name, value.protocol, value.model  # type: ignore[union-attr]
            except AttributeError as exc:
                raise ConversationError(
                    "Profile identity must provide name, protocol and model"
                ) from exc

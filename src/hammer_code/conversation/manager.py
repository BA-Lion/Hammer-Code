"""Atomic in-memory conversation history and usage management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.domain.usage import TokenUsage, UsageLedger
from hammer_code.errors import ConversationBusyError, ConversationError, InvalidTurnStateError


class TurnStatus(StrEnum):
    OPEN = "open"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class TurnTransaction:
    id: str
    user_message: Message
    request_ids: list[str] = field(default_factory=list)
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
        turn = TurnTransaction(str(uuid4()), Message(Role.USER, (TextBlock(user_text),)))
        self._active_turn = turn
        return turn

    def snapshot_for_request(self, turn: TurnTransaction) -> tuple[Message, ...]:
        self._assert_active(turn)
        assert self.conversation is not None
        return tuple(self.conversation.messages) + (turn.user_message,)

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
        self.conversation.messages.extend((turn.user_message, assistant_message))
        turn.status = TurnStatus.COMMITTED
        self._active_turn = None

    def abort(self, turn: TurnTransaction) -> None:
        self._assert_active(turn)
        turn.status = TurnStatus.ABORTED
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

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.errors import ConversationBusyError, InvalidTurnStateError


def _resolved():
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_chat_completions",
                    "model": "m",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "K",
                    "max_output_tokens": 2,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    return resolve_profile(config, None, {"K": "secret"})


def test_commit_abort_and_usage_are_transactional() -> None:
    manager = ConversationManager()
    manager.create(_resolved())
    turn = manager.begin_turn("hi")
    assert len(manager.snapshot_for_request(turn)) == 1
    manager.record_usage(turn, "req", TokenUsage(1, 2))
    manager.abort(turn)
    assert manager.conversation and manager.conversation.messages == []
    assert manager.conversation.usage_ledger.for_conversation().usage.total_tokens == 3
    next_turn = manager.begin_turn("again")
    manager.commit(next_turn, Message(Role.ASSISTANT, (TextBlock("ok"),)))
    assert len(manager.conversation.messages) == 2
    with pytest.raises(InvalidTurnStateError):
        manager.commit(next_turn, Message(Role.ASSISTANT, (TextBlock("no"),)))


def test_single_active_turn_and_clear() -> None:
    manager = ConversationManager()
    manager.create(_resolved())
    turn = manager.begin_turn("one")
    with pytest.raises(ConversationBusyError):
        manager.begin_turn("two")
    with pytest.raises(ConversationBusyError):
        manager.clear()
    manager.abort(turn)
    manager.clear()


def test_interrupted_turn_keeps_complete_tool_exchange() -> None:
    manager = ConversationManager()
    manager.create(_resolved())
    turn = manager.begin_turn("edit it")
    call = ToolCallBlock("call-1", "read_file", {"path": "x"}, '{"path":"x"}')
    manager.stage_tool_call(turn, Message(Role.ASSISTANT, (call,)))
    manager.stage_tool_results(
        turn, Message(Role.USER, (ToolResultBlock("call-1", (TextBlock("contents"),), False),))
    )
    manager.interrupt(turn)
    assert manager.conversation is not None
    assert [message.role for message in manager.conversation.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]

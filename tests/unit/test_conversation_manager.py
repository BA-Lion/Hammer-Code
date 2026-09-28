import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.errors import ConversationBusyError, InvalidTurnStateError
from hammer_code.skill.history import project_durable_skill_messages
from hammer_code.skill.models import SkillLoadOutcome, SkillRef, SkillScope


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


def test_runtime_turn_appends_one_complete_text_pair_without_an_active_turn() -> None:
    manager = ConversationManager()
    manager.create(_resolved())

    user, assistant = manager.append_completed_runtime_turn("runtime result", "recorded")

    assert user.role is Role.USER
    assert assistant.role is Role.ASSISTANT
    assert manager.snapshot_committed() == (user, assistant)
    turn = manager.begin_turn("next")
    with pytest.raises(ConversationBusyError):
        manager.append_completed_runtime_turn("other", "other")
    manager.abort(turn)


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


def test_durable_skill_projection_keeps_live_body_and_replaces_only_known_result() -> None:
    manager = ConversationManager()
    manager.create(_resolved())
    turn = manager.begin_turn("help")
    call = ToolCallBlock("skill-call", "use_skill", {"name": "review"}, '{"name":"review"}')
    manager.stage_tool_call(turn, Message(Role.ASSISTANT, (call,)))
    result = ToolResultBlock("skill-call", (TextBlock("private Skill body"),), False)
    manager.stage_tool_results(turn, Message(Role.USER, (result,)))
    outcome = manager.commit(
        turn,
        Message(Role.ASSISTANT, (TextBlock("done"),)),
        lambda live: project_durable_skill_messages(
            live,
            {"skill-call": SkillLoadOutcome(SkillRef(SkillScope.PROJECT, "review", "1.0.0"))},
        ),
    )
    live_result = outcome.live_messages[2].content[0]
    durable_result = outcome.durable_messages[2].content[0]
    assert isinstance(live_result, ToolResultBlock)
    assert isinstance(durable_result, ToolResultBlock)
    assert isinstance(live_result.content[0], TextBlock)
    assert isinstance(durable_result.content[0], TextBlock)
    assert live_result.content[0].text == "private Skill body"
    assert "private Skill body" not in durable_result.content[0].text


def test_direct_skill_projection_preserves_exact_original_without_text_search() -> None:
    manager = ConversationManager()
    manager.create(_resolved())
    original = "Question containing --- and the same Skill body text."
    turn = manager.begin_turn(original)
    suffix = (
        "---\nname: review\n---\nSkill body text.\n\n<retrieved-skills>metadata</retrieved-skills>"
    )
    manager.append_live_user_context(turn, suffix)
    request = manager.snapshot_for_request(turn)
    live_text = "".join(block.text for block in request[0].content if isinstance(block, TextBlock))
    assert live_text == original + "\n\n" + suffix
    outcome = manager.commit(
        turn,
        Message(Role.ASSISTANT, (TextBlock("done"),)),
        lambda live: project_durable_skill_messages(
            live, {}, original_user_message=turn.user_message
        ),
    )
    assert outcome.live_messages[0] == request[0]
    assert outcome.durable_messages[0] == Message(Role.USER, (TextBlock(original),))
    assert manager.snapshot_committed() == outcome.durable_messages

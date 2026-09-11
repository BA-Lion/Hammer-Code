from hammer_code.domain.events import ModelRequest
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from hammer_code.errors import TransportError
from hammer_code.llm._common import as_anthropic_messages, as_openai_chat_messages, map_exception
from hammer_code.llm.openai_responses import OpenAIResponsesClient


class APIConnectionError(Exception):
    pass


class ConnectError(Exception):
    pass


class ProxyError(Exception):
    pass


def test_connection_errors_are_actionable_and_do_not_leak_raw_details() -> None:
    secret = "token=secret https://example.test/path?key=secret"
    for exception_type in (APIConnectionError, ConnectionError, ConnectError, ProxyError):
        mapped = map_exception(exception_type(secret))
        assert isinstance(mapped, TransportError)
        assert str(mapped) == "Model connection failed; check network, proxy, DNS and TLS settings"
        assert secret not in str(mapped)


def test_three_protocol_history_mappers_preserve_multiple_tool_pairs() -> None:
    calls = Message(
        Role.ASSISTANT,
        (
            ToolCallBlock("one", "read_file", {"path": "a.txt"}, '{"path":"a.txt"}'),
            ToolCallBlock("two", "grep", {"path": ".", "query": "TODO"}, '{"query":"TODO"}'),
        ),
    )
    results = Message(
        Role.USER,
        (
            ToolResultBlock("one", (TextBlock("file contents"),), False),
            ToolResultBlock("two", (TextBlock("not permitted"),), True),
        ),
    )
    messages = (Message(Role.USER, (TextBlock("inspect"),)), calls, results)

    responses = OpenAIResponsesClient._input_messages(
        ModelRequest("request", "turn", "", messages, (), 32)
    )
    assert responses[-4:] == [
        {
            "type": "function_call",
            "call_id": "one",
            "name": "read_file",
            "arguments": '{"path":"a.txt"}',
        },
        {
            "type": "function_call",
            "call_id": "two",
            "name": "grep",
            "arguments": '{"query":"TODO"}',
        },
        {"type": "function_call_output", "call_id": "one", "output": "file contents"},
        {"type": "function_call_output", "call_id": "two", "output": "not permitted"},
    ]

    chat = as_openai_chat_messages(messages)
    assert chat[1]["role"] == "assistant"
    chat_calls = chat[1]["tool_calls"]
    assert isinstance(chat_calls, list)
    assert len(chat_calls) == 2
    assert chat[3] == {"role": "tool", "tool_call_id": "two", "content": "Error: not permitted"}

    anthropic = as_anthropic_messages(messages)
    anthropic_calls = anthropic[1]["content"]
    assert isinstance(anthropic_calls, list)
    assert len(anthropic_calls) == 2
    assert anthropic[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "one",
            "content": "file contents",
            "is_error": False,
        },
        {
            "type": "tool_result",
            "tool_use_id": "two",
            "content": "not permitted",
            "is_error": True,
        },
    ]

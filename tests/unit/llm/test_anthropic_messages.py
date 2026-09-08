from types import SimpleNamespace as NS

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.domain.events import ModelRequest, ResponseCompleted
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.llm.anthropic_messages import AnthropicMessagesClient
from tests.unit.llm.test_openai_chat import Stream


@pytest.mark.asyncio
async def test_anthropic_adapter_accumulates_text() -> None:
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "anthropic_messages",
                    "model": "m",
                    "base_url": "https://api.anthropic.com",
                    "api_key_env": "K",
                    "max_output_tokens": 2,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    stream = Stream(
        [
            NS(
                type="message_start",
                message=NS(id="r", model="m", usage=NS(input_tokens=1, output_tokens=0)),
            ),
            NS(type="content_block_start", index=0, content_block=NS(type="text")),
            NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="ok")),
            NS(type="content_block_stop", index=0),
            NS(
                type="message_delta",
                delta=NS(stop_reason="end_turn"),
                usage=NS(input_tokens=1, output_tokens=1),
            ),
            NS(type="message_stop"),
        ]
    )
    sdk = NS(messages=NS(create=lambda **_: stream), close=lambda: None)
    client = AnthropicMessagesClient(resolve_profile(config, None, {"K": "x"}), sdk)
    events = [
        event
        async for event in client.stream(
            ModelRequest("q", "t", "", (Message(Role.USER, (TextBlock("x"),)),), (), 2)
        )
    ]
    assert isinstance(events[-1], ResponseCompleted)

from types import SimpleNamespace as NS

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.domain.events import ModelRequest, ResponseCompleted, TextDelta
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.llm.openai_chat import OpenAIChatCompletionsClient


class Stream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __aiter__(self):
        async def values():
            for value in self.events:
                yield value

        return values()

    async def aclose(self):
        self.closed = True


class SDK:
    def __init__(self, stream):
        self.chat = NS(completions=NS(create=lambda **_: stream))
        self.stream = stream

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_chat_adapter_maps_text_usage_and_finish() -> None:
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
    stream = Stream(
        [
            NS(
                id="r",
                model="m",
                choices=[NS(index=0, delta=NS(content="hi", tool_calls=[]), finish_reason=None)],
                usage=None,
            ),
            NS(
                id="r",
                model="m",
                choices=[NS(index=0, delta=NS(content=None, tool_calls=[]), finish_reason="stop")],
                usage=NS(prompt_tokens=2, completion_tokens=1),
            ),
        ]
    )
    client = OpenAIChatCompletionsClient(resolve_profile(config, None, {"K": "x"}), SDK(stream))
    request = ModelRequest("q", "t", "", (Message(Role.USER, (TextBlock("x"),)),), (), 2)
    events = [event async for event in client.stream(request)]
    assert any(isinstance(event, TextDelta) and event.text == "hi" for event in events)
    assert isinstance(events[-1], ResponseCompleted)
    assert stream.closed

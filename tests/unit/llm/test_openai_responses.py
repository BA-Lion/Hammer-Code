from types import SimpleNamespace as NS

import pytest

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.domain.events import ModelRequest, ResponseCompleted
from hammer_code.domain.messages import Message, Role, TextBlock
from hammer_code.llm.openai_responses import OpenAIResponsesClient
from tests.unit.llm.test_openai_chat import Stream


@pytest.mark.asyncio
async def test_responses_adapter_completes() -> None:
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_responses",
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
    response = NS(
        id="r",
        model="m",
        usage=NS(
            input_tokens=1,
            output_tokens=2,
            input_tokens_details=NS(cached_tokens=0),
            output_tokens_details=NS(reasoning_tokens=0),
        ),
    )
    stream = Stream(
        [
            NS(type="response.created", response=response),
            NS(type="response.output_text.delta", delta="ok", content_index=0),
            NS(type="response.completed", response=response),
        ]
    )
    sdk = NS(responses=NS(create=lambda **_: stream), close=lambda: None)
    client = OpenAIResponsesClient(resolve_profile(config, None, {"K": "x"}), sdk)
    request = ModelRequest("q", "t", "", (Message(Role.USER, (TextBlock("x"),)),), (), 2)
    events = [event async for event in client.stream(request)]
    assert isinstance(events[-1], ResponseCompleted)
    assert events[-1].response.message.content[0] == TextBlock("ok")

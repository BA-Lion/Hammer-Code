"""Exact protocol discriminator factory."""

from __future__ import annotations

from hammer_code.config import (
    AnthropicMessagesProfile,
    OpenAIChatProfile,
    OpenAIResponsesProfile,
    ResolvedProfile,
)
from hammer_code.llm.anthropic_messages import AnthropicMessagesClient
from hammer_code.llm.client import ModelClient
from hammer_code.llm.openai_chat import OpenAIChatCompletionsClient
from hammer_code.llm.openai_responses import OpenAIResponsesClient


def create_model_client(resolved_profile: ResolvedProfile) -> ModelClient:
    profile = resolved_profile.profile
    if isinstance(profile, OpenAIResponsesProfile):
        return OpenAIResponsesClient(resolved_profile)
    if isinstance(profile, OpenAIChatProfile):
        return OpenAIChatCompletionsClient(resolved_profile)
    if isinstance(profile, AnthropicMessagesProfile):
        return AnthropicMessagesClient(resolved_profile)
    raise AssertionError(f"Unsupported protocol {profile.protocol}")

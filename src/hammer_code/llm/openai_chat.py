"""OpenAI Chat Completions / compatible protocol adapter."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import cast

from hammer_code.config import OpenAIChatProfile, ResolvedProfile
from hammer_code.domain.events import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    ResponseStarted,
    StopReason,
    TextDelta,
    ToolCallCompleted,
    UsageUpdated,
)
from hammer_code.domain.messages import ContentBlock, ReasoningBlock, ReasoningVisibility, TextBlock
from hammer_code.domain.usage import UsageStatus
from hammer_code.errors import StreamInterruptedError, StreamProtocolError
from hammer_code.llm._common import (
    as_openai_chat_messages,
    blocks_from_parts,
    get,
    make_tool_call,
    map_exception,
    normalize_openai_usage,
)
from hammer_code.llm.client import ClientCapabilities, ModelClient


class OpenAIChatCompletionsClient(ModelClient):
    def __init__(self, resolved: ResolvedProfile, sdk_client: object | None = None) -> None:
        self.resolved, self.profile = resolved, cast(OpenAIChatProfile, resolved.profile)
        if sdk_client is None:
            from openai import AsyncOpenAI

            sdk_client = AsyncOpenAI(
                api_key=resolved.api_key.get_secret_value(),
                base_url=self.profile.base_url,
                timeout=self.profile.timeout_seconds,
                max_retries=0,
            )
        self._sdk = sdk_client

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities(
            "openai_chat_completions", True, True, True, self.profile.stream_include_usage
        )

    async def aclose(self) -> None:
        result = self._sdk.close()  # type: ignore[attr-defined]
        if inspect.isawaitable(result):
            await result

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        payload: dict[str, object] = {
            "model": self.profile.model,
            "messages": as_openai_chat_messages(request.messages),
            "max_completion_tokens": request.max_output_tokens,
            "stream": True,
            "n": 1,
        }
        if request.system_prompt:
            payload["messages"] = [{"role": "system", "content": request.system_prompt}] + payload[
                "messages"
            ]  # type: ignore[operator]
        if self.profile.stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": False,
                    },
                }
                for tool in request.tools
            ]
        stream: object | None = None
        started = False
        finished = False
        parts: list[ContentBlock] = []
        tools: dict[int, dict[str, str]] = {}
        usage_seen = False
        try:
            result = self._sdk.chat.completions.create(**payload)  # type: ignore[attr-defined]
            stream = await result if inspect.isawaitable(result) else result
            async for chunk in stream:  # type: ignore[union-attr]
                choices = get(chunk, "choices", []) or []
                usage = get(chunk, "usage")
                if usage is not None:
                    usage_seen = True
                    yield UsageUpdated(
                        request.request_id, normalize_openai_usage(usage, UsageStatus.FINAL)
                    )
                if not choices:
                    continue
                if len(choices) != 1:
                    raise StreamProtocolError("Chat Completions returned multiple choices")
                choice = choices[0]
                index = get(choice, "index", 0)
                if index != 0:
                    raise StreamProtocolError("Chat Completions choice index must be zero")
                if not started:
                    started = True
                    yield ResponseStarted(
                        request.request_id,
                        str(get(chunk, "id", "unknown") or "unknown"),
                        str(get(chunk, "model", self.profile.model) or self.profile.model),
                    )
                delta = get(choice, "delta")
                content = get(delta, "content", "") if delta else ""
                if content:
                    text = str(content)
                    parts.append(TextBlock(text))
                    yield TextDelta(request.request_id, len(parts) - 1, text)
                reasoning = (
                    get(delta, "reasoning_content", get(delta, "reasoning", "")) if delta else ""
                )
                if reasoning:
                    from hammer_code.domain.events import ReasoningDelta

                    text = str(reasoning)
                    parts.append(ReasoningBlock(text, ReasoningVisibility.VISIBLE))
                    yield ReasoningDelta(
                        request.request_id, len(parts) - 1, text, ReasoningVisibility.VISIBLE
                    )
                for tool in get(delta, "tool_calls", []) or []:
                    tool_index = int(get(tool, "index", 0) or 0)
                    state = tools.setdefault(tool_index, {"id": "", "name": "", "raw": ""})
                    state["id"] += str(get(tool, "id", "") or "")
                    func = get(tool, "function")
                    state["name"] += str(get(func, "name", "") or "")
                    state["raw"] += str(get(func, "arguments", "") or "")
                finish = get(choice, "finish_reason")
                if finish:
                    if finished:
                        raise StreamProtocolError("Chat Completions repeated finish reason")
                    for tool_index in sorted(tools):
                        call = make_tool_call(
                            tools[tool_index]["id"],
                            tools[tool_index]["name"],
                            tools[tool_index]["raw"],
                        )
                        parts.append(call)
                        yield ToolCallCompleted(request.request_id, len(parts) - 1, call)
                    stop = {
                        "stop": StopReason.END_TURN,
                        "tool_calls": StopReason.TOOL_CALL,
                        "length": StopReason.MAX_TOKENS,
                        "content_filter": StopReason.CONTENT_FILTER,
                    }.get(str(finish), StopReason.OTHER)
                    usage_value = (
                        normalize_openai_usage(usage, UsageStatus.FINAL)
                        if usage is not None
                        else normalize_openai_usage(None, UsageStatus.UNAVAILABLE)
                    )
                    if not usage_seen:
                        yield UsageUpdated(request.request_id, usage_value)
                    yield ResponseCompleted(
                        request.request_id,
                        ModelResponse(
                            str(get(chunk, "id", "unknown") or "unknown"),
                            str(get(chunk, "model", self.profile.model) or self.profile.model),
                            blocks_from_parts(parts),
                            stop,
                            usage_value,
                        ),
                    )
                    finished = True
            if not finished:
                raise StreamInterruptedError("Chat Completions stream ended before a finish reason")
        except asyncio.CancelledError:
            raise
        except (StreamProtocolError, StreamInterruptedError):
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        finally:
            if stream is not None:
                close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
                if close:
                    outcome = close()
                    if inspect.isawaitable(outcome):
                        await outcome

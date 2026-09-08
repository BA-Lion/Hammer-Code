"""Anthropic Messages protocol adapter with independent block accumulation."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import cast

from hammer_code.config import AnthropicMessagesProfile, ResolvedProfile
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
from hammer_code.domain.messages import (
    ContentBlock,
    ProviderStateBlock,
    ReasoningBlock,
    ReasoningVisibility,
    TextBlock,
)
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.errors import StreamInterruptedError, StreamProtocolError
from hammer_code.llm._common import (
    as_anthropic_messages,
    blocks_from_parts,
    get,
    make_tool_call,
    map_exception,
)
from hammer_code.llm.client import ClientCapabilities, ModelClient


def _usage(raw: object | None, status: UsageStatus) -> TokenUsage:
    if raw is None:
        return TokenUsage(None, None, status=UsageStatus.UNAVAILABLE)
    read = int(get(raw, "cache_read_input_tokens", 0) or 0)
    write = int(get(raw, "cache_creation_input_tokens", 0) or 0)
    input_tokens = get(raw, "input_tokens")
    return TokenUsage(
        (int(input_tokens) if input_tokens is not None else 0) + read + write,
        get(raw, "output_tokens"),
        read,
        write,
        0,
        status,
    )


class AnthropicMessagesClient(ModelClient):
    def __init__(self, resolved: ResolvedProfile, sdk_client: object | None = None) -> None:
        self.resolved, self.profile = resolved, cast(AnthropicMessagesProfile, resolved.profile)
        if sdk_client is None:
            from anthropic import AsyncAnthropic

            sdk_client = AsyncAnthropic(
                api_key=resolved.api_key.get_secret_value(),
                base_url=self.profile.base_url,
                timeout=self.profile.timeout_seconds,
                max_retries=0,
            )
        self._sdk = sdk_client

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities(
            "anthropic_messages", True, True, self.profile.thinking_mode != "disabled", True
        )

    async def aclose(self) -> None:
        result = self._sdk.close()  # type: ignore[attr-defined]
        if inspect.isawaitable(result):
            await result

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        payload: dict[str, object] = {
            "model": self.profile.model,
            "max_tokens": request.max_output_tokens,
            "messages": as_anthropic_messages(request.messages),
            "system": request.system_prompt,
            "stream": True,
        }
        if self.profile.thinking_mode == "enabled":
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.profile.thinking_budget}
        elif self.profile.thinking_mode == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        stream: object | None = None
        started = False
        completed = False
        blocks: dict[int, dict[str, object]] = {}
        parts: list[ContentBlock] = []
        final_usage = TokenUsage(None, None, status=UsageStatus.UNAVAILABLE)
        stop = StopReason.OTHER
        response_id = "unknown"
        model = self.profile.model
        try:
            result = self._sdk.messages.create(**payload)  # type: ignore[attr-defined]
            stream = await result if inspect.isawaitable(result) else result
            async for event in stream:  # type: ignore[union-attr]
                typ = str(get(event, "type", "") or "")
                if typ == "message_start":
                    if started:
                        raise StreamProtocolError("Anthropic stream started twice")
                    started = True
                    message = get(event, "message")
                    response_id = str(get(message, "id", "unknown") or "unknown")
                    model = str(get(message, "model", model) or model)
                    initial = _usage(get(message, "usage"), UsageStatus.PARTIAL)
                    final_usage = initial
                    yield ResponseStarted(request.request_id, response_id, model)
                    yield UsageUpdated(request.request_id, initial)
                elif typ == "content_block_start":
                    index = int(get(event, "index", 0) or 0)
                    content = get(event, "content_block")
                    kind = str(get(content, "type", "") or "")
                    blocks[index] = {
                        "kind": kind,
                        "text": "",
                        "id": str(get(content, "id", "") or ""),
                        "name": str(get(content, "name", "") or ""),
                        "input": "",
                        "signature": "",
                        "redacted": get(content, "data"),
                    }
                elif typ == "content_block_delta":
                    index = int(get(event, "index", 0) or 0)
                    if index not in blocks:
                        raise StreamProtocolError("Anthropic delta has no open block")
                    delta = get(event, "delta")
                    kind = str(get(delta, "type", "") or "")
                    state = blocks[index]
                    if kind == "text_delta":
                        text = str(get(delta, "text", "") or "")
                        state["text"] = str(state["text"]) + text
                        yield TextDelta(request.request_id, index, text)
                    elif kind == "thinking_delta":
                        text = str(get(delta, "thinking", "") or "")
                        state["text"] = str(state["text"]) + text
                        from hammer_code.domain.events import ReasoningDelta

                        yield ReasoningDelta(
                            request.request_id, index, text, ReasoningVisibility.VISIBLE
                        )
                    elif kind == "signature_delta":
                        state["signature"] = str(state["signature"]) + str(
                            get(delta, "signature", "") or ""
                        )
                    elif kind == "input_json_delta":
                        state["input"] = str(state["input"]) + str(
                            get(delta, "partial_json", "") or ""
                        )
                elif typ == "content_block_stop":
                    index = int(get(event, "index", 0) or 0)
                    state = blocks.pop(index, None)
                    if state is None:
                        raise StreamProtocolError("Anthropic stopped an unknown block")
                    kind = str(state["kind"])
                    if kind == "text":
                        parts.append(TextBlock(str(state["text"])))
                    elif kind == "thinking":
                        parts.append(
                            ReasoningBlock(str(state["text"]), ReasoningVisibility.VISIBLE)
                        )
                        if state["signature"]:
                            parts.append(
                                ProviderStateBlock(
                                    "anthropic_messages",
                                    "thinking_signature",
                                    {"signature": str(state["signature"])},
                                )
                            )
                    elif kind == "redacted_thinking":
                        parts.append(
                            ProviderStateBlock(
                                "anthropic_messages",
                                "redacted_thinking",
                                {"data": str(state["redacted"] or "")},
                            )
                        )
                    elif kind == "tool_use":
                        call = make_tool_call(
                            str(state["id"]), str(state["name"]), str(state["input"])
                        )
                        parts.append(call)
                        yield ToolCallCompleted(request.request_id, index, call)
                elif typ == "message_delta":
                    delta = get(event, "delta")
                    raw_stop = str(get(delta, "stop_reason", "") or "")
                    stop = {
                        "end_turn": StopReason.END_TURN,
                        "tool_use": StopReason.TOOL_CALL,
                        "max_tokens": StopReason.MAX_TOKENS,
                        "refusal": StopReason.REFUSAL,
                    }.get(raw_stop, StopReason.OTHER)
                    final_usage = _usage(get(event, "usage"), UsageStatus.PARTIAL)
                    yield UsageUpdated(request.request_id, final_usage)
                elif typ == "message_stop":
                    if not started or completed or blocks:
                        raise StreamProtocolError("Anthropic stream is incomplete")
                    final_usage = TokenUsage(
                        final_usage.input_tokens,
                        final_usage.output_tokens,
                        final_usage.cache_read_tokens,
                        final_usage.cache_write_tokens,
                        final_usage.reasoning_tokens,
                        UsageStatus.FINAL,
                    )
                    yield UsageUpdated(request.request_id, final_usage)
                    yield ResponseCompleted(
                        request.request_id,
                        ModelResponse(
                            response_id, model, blocks_from_parts(parts), stop, final_usage
                        ),
                    )
                    completed = True
                elif typ in {"ping", ""}:
                    continue
                elif typ == "error":
                    raise StreamProtocolError("Anthropic provider reported a stream error")
            if not completed:
                raise StreamInterruptedError("Anthropic stream ended before message_stop")
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

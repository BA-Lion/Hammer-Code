"""OpenAI Responses protocol adapter with its own event state machine."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import cast

from hammer_code.config import OpenAIResponsesProfile, ResolvedProfile
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
    RefusalBlock,
    TextBlock,
)
from hammer_code.domain.usage import UsageStatus
from hammer_code.errors import StreamInterruptedError, StreamProtocolError
from hammer_code.llm._common import (
    blocks_from_parts,
    get,
    make_tool_call,
    map_exception,
    normalize_openai_usage,
)
from hammer_code.llm.client import ClientCapabilities, ModelClient


class OpenAIResponsesClient(ModelClient):
    def __init__(self, resolved: ResolvedProfile, sdk_client: object | None = None) -> None:
        self.resolved = resolved
        self.profile = cast(OpenAIResponsesProfile, resolved.profile)
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
        return ClientCapabilities("openai_responses", True, True, True, True)

    async def _create_stream(self, payload: dict[str, object]) -> object:
        result = self._sdk.responses.create(**payload)  # type: ignore[attr-defined]
        return await result if inspect.isawaitable(result) else result

    async def _close_stream(self, stream: object) -> None:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def aclose(self) -> None:
        close = getattr(self._sdk, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        payload: dict[str, object] = {
            "model": self.profile.model,
            "input": self._input_messages(request),
            "instructions": request.system_prompt,
            "max_output_tokens": request.max_output_tokens,
            "stream": True,
            "store": False,
        }
        if self.profile.reasoning_effort or self.profile.reasoning_summary:
            reasoning: dict[str, object] = {}
            if self.profile.reasoning_effort:
                reasoning["effort"] = self.profile.reasoning_effort
            if self.profile.reasoning_summary:
                reasoning["summary"] = self.profile.reasoning_summary
            payload["reasoning"] = reasoning
        event_stream: object | None = None
        started = False
        completed = False
        parts: list[ContentBlock] = []
        tool_raw: dict[str, dict[str, str]] = {}
        try:
            event_stream = await self._create_stream(payload)
            async for event in event_stream:  # type: ignore[union-attr]
                event_type = get(event, "type", "") or ""
                response = get(event, "response")
                if event_type in {"response.created", "response.in_progress"}:
                    if not started:
                        response_id = (
                            get(response, "id", get(event, "response_id", "unknown")) or "unknown"
                        )
                        model = get(response, "model", self.profile.model) or self.profile.model
                        started = True
                        yield ResponseStarted(request.request_id, str(response_id), str(model))
                elif event_type == "response.output_text.delta":
                    text = get(event, "delta", "") or ""
                    if text:
                        index = int(get(event, "content_index", len(parts)) or len(parts))
                        parts.append(TextBlock(str(text)))
                        yield TextDelta(request.request_id, index, str(text))
                elif event_type in {"response.refusal.delta", "response.refusal.done"}:
                    delta = str(get(event, "delta", get(event, "refusal", "")) or "")
                    if delta:
                        parts.append(RefusalBlock(delta))
                elif event_type == "response.reasoning_summary_text.delta":
                    delta = str(get(event, "delta", "") or "")
                    if delta:
                        index = int(get(event, "summary_index", len(parts)) or 0)
                        parts.append(ReasoningBlock(delta, ReasoningVisibility.SUMMARY))
                        from hammer_code.domain.events import ReasoningDelta

                        yield ReasoningDelta(
                            request.request_id, index, delta, ReasoningVisibility.SUMMARY
                        )
                elif event_type in {
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                }:
                    item_id = str(get(event, "item_id", get(event, "call_id", "")) or "")
                    state = tool_raw.setdefault(
                        item_id,
                        {
                            "id": str(get(event, "call_id", "") or ""),
                            "name": str(get(event, "name", "") or ""),
                            "raw": "",
                        },
                    )
                    state["id"] = state["id"] or str(get(event, "call_id", "") or "")
                    state["name"] = state["name"] or str(get(event, "name", "") or "")
                    state["raw"] += str(get(event, "delta", "") or "")
                    if event_type.endswith(".done"):
                        state["raw"] = str(get(event, "arguments", state["raw"]) or state["raw"])
                        call = make_tool_call(state["id"], state["name"], state["raw"])
                        index = int(get(event, "output_index", len(parts)) or len(parts))
                        parts.append(call)
                        yield ToolCallCompleted(request.request_id, index, call)
                elif event_type in {"response.completed", "response.incomplete"}:
                    if completed or not started:
                        raise StreamProtocolError("Responses stream has invalid termination")
                    final = response or get(event, "response")
                    usage = normalize_openai_usage(get(final, "usage"), UsageStatus.FINAL)
                    yield UsageUpdated(request.request_id, usage)
                    incomplete = get(final, "incomplete_details") or get(
                        event, "incomplete_details"
                    )
                    reason = str(get(incomplete, "reason", "") or "")
                    stop = (
                        StopReason.MAX_TOKENS if reason == "max_output_tokens" else StopReason.OTHER
                    )
                    message = blocks_from_parts(parts)
                    response_id = str(get(final, "id", "unknown") or "unknown")
                    model = str(get(final, "model", self.profile.model) or self.profile.model)
                    yield ResponseCompleted(
                        request.request_id, ModelResponse(response_id, model, message, stop, usage)
                    )
                    completed = True
                elif event_type in {"response.failed", "error"}:
                    raise StreamProtocolError("Responses provider reported a failed stream")
            if not completed:
                raise StreamInterruptedError("Responses stream ended before a completion event")
        except asyncio.CancelledError:
            raise
        except (StreamProtocolError, StreamInterruptedError):
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        finally:
            if event_stream is not None:
                await self._close_stream(event_stream)

    @staticmethod
    def _input_messages(request: ModelRequest) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for message in request.messages:
            blocks: list[dict[str, object]] = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    blocks.append(
                        {
                            "type": "input_text" if message.role.value == "user" else "output_text",
                            "text": block.text,
                        }
                    )
                elif isinstance(block, ProviderStateBlock) and block.protocol == "openai_responses":
                    blocks.append(dict(block.data))
            if blocks:
                result.append({"role": message.role.value, "content": blocks})
        return result

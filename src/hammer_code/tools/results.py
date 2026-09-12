"""Bounded ToolResult content with deterministic local token estimates."""

from __future__ import annotations

from dataclasses import dataclass

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import ContextConfig
from hammer_code.domain.messages import Message, Role, TextBlock, ToolResultBlock
from hammer_code.tools.runtime import RuntimeStore

INLINE_BYTES = 32 * 1024
INLINE_LINES = 1000
CAPTURE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class BoundedResult:
    content: str
    stored: bool


def clip_prefix(content: str, token_limit: int, estimator: TokenEstimator) -> str:
    if not content:
        return "(empty output)"
    encoded = content.encode("utf-8")
    low, high = 0, len(encoded)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = encoded[:mid].decode("utf-8", errors="ignore")
        if estimator.estimate_text(candidate) <= token_limit:
            low = mid
        else:
            high = mid - 1
    return encoded[:low].decode("utf-8", errors="ignore") or "(output omitted)"


def _capture(content: str) -> str:
    if len(content.encode("utf-8")) <= CAPTURE_BYTES:
        return content
    return (
        _clip_bytes(content, CAPTURE_BYTES, 0.7) + "\n[Capture truncated; not a complete capture.]"
    )


def _preview(
    content: str, call_id: str, runtime: RuntimeStore, token_limit: int, estimator: TokenEstimator
) -> str:
    path = runtime.write_result(call_id, _capture(content))
    address = path.relative_to(runtime.root.parent.parent).as_posix()
    return (
        f"{address}\n{clip_prefix(content, token_limit, estimator)}\n"
        f"[Tool output truncated; original estimate ~{estimator.estimate_text(content)} tokens.]"
    )


def bound_result(
    content: str,
    call_id: str,
    runtime: RuntimeStore,
    context_config: ContextConfig | None = None,
    estimator: TokenEstimator | None = None,
) -> BoundedResult:
    config = context_config or ContextConfig()
    token_estimator = estimator or TokenEstimator()
    normalized = content or "(empty output)"
    if (
        token_estimator.estimate_text(normalized) <= config.tool_result_overflow_tokens
        and len(normalized.encode("utf-8")) <= INLINE_BYTES
        and len(normalized.splitlines()) <= INLINE_LINES
    ):
        return BoundedResult(normalized, False)
    return BoundedResult(
        _preview(normalized, call_id, runtime, config.tool_result_preview_tokens, token_estimator),
        True,
    )


def bound_tool_results(
    results: tuple[ToolResultBlock, ...],
    runtime: RuntimeStore,
    context_config: ContextConfig,
    estimator: TokenEstimator,
) -> tuple[ToolResultBlock, ...]:
    bounded = [
        ToolResultBlock(
            item.call_id,
            (
                TextBlock(
                    bound_result(
                        "\n".join(part.text for part in item.content),
                        item.call_id,
                        runtime,
                        context_config,
                        estimator,
                    ).content
                ),
            ),
            item.is_error,
        )
        for item in results
    ]

    def total() -> int:
        return estimator.estimate_messages((Message(Role.USER, tuple(bounded)),))

    while total() > context_config.tool_batch_tokens:
        changed = False
        for index in sorted(
            range(len(bounded)),
            key=lambda i: estimator.estimate_messages((Message(Role.USER, (bounded[i],)),)),
            reverse=True,
        ):
            item = bounded[index]
            current = item.content[0].text
            if current.startswith(".hammer-code/"):
                address, _, rest = current.partition("\n")
                preview = clip_prefix(
                    rest,
                    max(1, context_config.tool_result_preview_tokens // 2),
                    estimator,
                )
                replacement = (
                    f"{address}\n{preview}\n[Tool output truncated further for batch budget.]"
                )
            else:
                replacement = _preview(
                    current,
                    item.call_id,
                    runtime,
                    context_config.tool_result_preview_tokens,
                    estimator,
                )
            if replacement != current:
                bounded[index] = ToolResultBlock(
                    item.call_id, (TextBlock(replacement),), item.is_error
                )
                changed = True
            if total() <= context_config.tool_batch_tokens:
                break
        if not changed:
            break
    return tuple(bounded)


def _clip_bytes(content: str, limit: int, head_fraction: float) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= limit:
        return content
    head_size = int(limit * head_fraction)
    tail_size = limit - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    return f"{head}\n[... output omitted ...]\n{tail}"

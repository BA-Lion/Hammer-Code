"""Tool result clipping with a bounded optional complete capture."""

from __future__ import annotations

from dataclasses import dataclass

from hammer_code.tools.runtime import RuntimeStore

INLINE_BYTES = 32 * 1024
INLINE_LINES = 1000
CAPTURE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class BoundedResult:
    content: str
    stored: bool


def bound_result(content: str, call_id: str, runtime: RuntimeStore) -> BoundedResult:
    encoded = content.encode("utf-8")
    lines = content.splitlines()
    if len(encoded) <= INLINE_BYTES and len(lines) <= INLINE_LINES:
        return BoundedResult(content, False)
    stored_content = _clip_bytes(content, CAPTURE_BYTES, 0.7)
    result_path = runtime.write_result(call_id, stored_content)
    clipped = _clip_bytes(content, INLINE_BYTES, 0.7)
    omitted = len(encoded) - len(clipped.encode("utf-8"))
    return BoundedResult(
        f"{clipped}\n\n[Tool output truncated; omitted {max(0, omitted)} bytes. "
        f"Complete capture: {result_path.relative_to(runtime.root.parent.parent)}]",
        True,
    )


def _clip_bytes(content: str, limit: int, head_fraction: float) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= limit:
        return content
    head_size = int(limit * head_fraction)
    tail_size = limit - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    return f"{head}\n[... output omitted ...]\n{tail}"

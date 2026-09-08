"""Provider-reported token snapshots and their conservative aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class UsageStatus(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    status: UsageStatus = UsageStatus.PARTIAL

    def __post_init__(self) -> None:
        for value in (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
        ):
            if value is not None and value < 0:
                raise ValueError("Token counts cannot be negative")
        if self.status is UsageStatus.UNAVAILABLE and (
            self.input_tokens is not None or self.output_tokens is not None
        ):
            raise ValueError("Unavailable usage cannot contain token totals")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class UsageSummary:
    usage: TokenUsage
    final_requests: int
    partial_requests: int
    unavailable_requests: int


class UsageLedger:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, TokenUsage]] = {}

    def upsert(self, request_id: str, turn_id: str, usage: TokenUsage) -> None:
        old = self._entries.get(request_id)
        if (
            old is not None
            and old[1].status is UsageStatus.FINAL
            and usage.status is UsageStatus.PARTIAL
        ):
            return
        self._entries[request_id] = (turn_id, usage)

    def mark_unavailable(self, request_id: str, turn_id: str) -> None:
        if request_id not in self._entries:
            self._entries[request_id] = (
                turn_id,
                TokenUsage(None, None, status=UsageStatus.UNAVAILABLE),
            )

    def for_turn(self, turn_id: str) -> UsageSummary:
        return self._summarize(item[1] for item in self._entries.values() if item[0] == turn_id)

    def for_conversation(self) -> UsageSummary:
        return self._summarize(item[1] for item in self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def _summarize(usages: Iterable[TokenUsage]) -> UsageSummary:
        entries = list(usages)
        known_input = [u.input_tokens for u in entries if u.input_tokens is not None]
        known_output = [u.output_tokens for u in entries if u.output_tokens is not None]
        status_counts = {status: sum(u.status is status for u in entries) for status in UsageStatus}
        usage = TokenUsage(
            sum(known_input) if len(known_input) == len(entries) else None,
            sum(known_output) if len(known_output) == len(entries) else None,
            sum(u.cache_read_tokens for u in entries),
            sum(u.cache_write_tokens for u in entries),
            sum(u.reasoning_tokens for u in entries),
            UsageStatus.FINAL
            if entries and status_counts[UsageStatus.FINAL] == len(entries)
            else (
                UsageStatus.UNAVAILABLE
                if not entries or status_counts[UsageStatus.UNAVAILABLE]
                else UsageStatus.PARTIAL
            ),
        )
        return UsageSummary(
            usage,
            status_counts[UsageStatus.FINAL],
            status_counts[UsageStatus.PARTIAL],
            status_counts[UsageStatus.UNAVAILABLE],
        )

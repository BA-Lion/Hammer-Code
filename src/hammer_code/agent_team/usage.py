"""Atomic TeamRun-wide token admission for concurrent model requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.subagent.usage import SubagentUsageSnapshot


@dataclass
class _RequestAccount:
    estimated_input: int
    allowed_output: int
    usage: TokenUsage

    @property
    def charge(self) -> int:
        if self.usage.status is UsageStatus.FINAL and self.usage.total_tokens is not None:
            return self.usage.total_tokens
        return self.estimated_input + self.allowed_output


class TeamUsageTracker:
    """One lock protects both reservation and provider-usage correction.

    Reservations deliberately remain conservative until a provider FINAL result is
    observed, so concurrently starting assignments cannot spend the same budget.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Team token limit must be positive")
        self._limit = limit
        self._requests: dict[str, _RequestAccount] = {}
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    async def reserve(self, request_id: str, estimated_input: int, max_output: int) -> int | None:
        if estimated_input < 0 or max_output < 1:
            raise ValueError("Team request estimate is invalid")
        async with self._lock:
            if request_id in self._requests:
                raise ValueError("Team request id is already reserved")
            remaining = self._limit - sum(item.charge for item in self._requests.values())
            allowed = min(max_output, remaining - estimated_input)
            if allowed < 1:
                return None
            self._requests[request_id] = _RequestAccount(
                estimated_input, allowed, TokenUsage(None, None, status=UsageStatus.UNAVAILABLE)
            )
            return allowed

    async def observe(self, request_id: str, usage: TokenUsage) -> None:
        async with self._lock:
            try:
                account = self._requests[request_id]
            except KeyError as exc:
                raise ValueError("Team usage arrived for an unreserved request") from exc
            if account.usage.status is UsageStatus.FINAL and usage.status is UsageStatus.PARTIAL:
                return
            account.usage = usage

    async def snapshot(self) -> SubagentUsageSnapshot:
        async with self._lock:
            accounts = tuple(self._requests.values())
            usages = tuple(item.usage for item in accounts)
            known_input = tuple(
                item.input_tokens for item in usages if item.input_tokens is not None
            )
            known_output = tuple(
                item.output_tokens for item in usages if item.output_tokens is not None
            )
            totals = tuple(item.total_tokens for item in usages if item.total_tokens is not None)
            accounted = sum(item.charge for item in accounts)
            reported = sum(totals) if len(totals) == len(usages) and usages else None
            return SubagentUsageSnapshot(
                input_tokens=sum(known_input)
                if len(known_input) == len(usages) and usages
                else None,
                output_tokens=sum(known_output)
                if len(known_output) == len(usages) and usages
                else None,
                cache_read_tokens=sum(item.cache_read_tokens for item in usages),
                cache_write_tokens=sum(item.cache_write_tokens for item in usages),
                reasoning_tokens=sum(item.reasoning_tokens for item in usages),
                reported_total=reported,
                accounted_tokens=accounted,
                limit=self._limit,
                final_requests=sum(item.status is UsageStatus.FINAL for item in usages),
                partial_requests=sum(item.status is UsageStatus.PARTIAL for item in usages),
                unavailable_requests=sum(item.status is UsageStatus.UNAVAILABLE for item in usages),
                estimated=any(
                    item.status is not UsageStatus.FINAL or item.total_tokens is None
                    for item in usages
                ),
                exhausted=accounted >= self._limit,
                exceeded=reported is not None and reported > self._limit,
            )

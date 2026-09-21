"""Per-invocation Subagent token budgeting without persistent accounting."""

from __future__ import annotations

from dataclasses import dataclass

from hammer_code.domain.usage import TokenUsage, UsageStatus


@dataclass(frozen=True)
class SubagentUsageSnapshot:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    reported_total: int | None
    accounted_tokens: int
    limit: int
    final_requests: int
    partial_requests: int
    unavailable_requests: int
    estimated: bool
    exhausted: bool
    exceeded: bool


@dataclass
class _RequestAccount:
    estimated_input: int
    allowed_max_output: int
    usage: TokenUsage

    @property
    def conservative_charge(self) -> int:
        if self.usage.status is UsageStatus.FINAL and self.usage.total_tokens is not None:
            return self.usage.total_tokens
        return self.estimated_input + self.allowed_max_output


class SubagentUsageTracker:
    """Reserve request budgets and retain the newest provider snapshot per request."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Subagent token limit must be positive")
        self._limit = limit
        self._requests: dict[str, _RequestAccount] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def reserve(self, request_id: str, estimated_input: int, max_output: int) -> int | None:
        if request_id in self._requests:
            raise ValueError("Subagent request id is already reserved")
        if estimated_input < 0 or max_output < 1:
            raise ValueError("Subagent request estimate is invalid")
        remaining = self._limit - self.snapshot().accounted_tokens
        allowed = min(max_output, remaining - estimated_input)
        if allowed < 1:
            return None
        self._requests[request_id] = _RequestAccount(
            estimated_input,
            allowed,
            TokenUsage(None, None, status=UsageStatus.UNAVAILABLE),
        )
        return allowed

    def observe(self, request_id: str, usage: TokenUsage) -> None:
        try:
            account = self._requests[request_id]
        except KeyError as exc:
            raise ValueError("Subagent usage arrived for an unreserved request") from exc
        if account.usage.status is UsageStatus.FINAL and usage.status is UsageStatus.PARTIAL:
            return
        account.usage = usage

    def snapshot(self) -> SubagentUsageSnapshot:
        accounts = tuple(self._requests.values())
        usages = tuple(account.usage for account in accounts)
        known_input = tuple(
            usage.input_tokens for usage in usages if usage.input_tokens is not None
        )
        known_output = tuple(
            usage.output_tokens for usage in usages if usage.output_tokens is not None
        )
        totals = tuple(usage.total_tokens for usage in usages if usage.total_tokens is not None)
        input_tokens = sum(known_input) if len(known_input) == len(usages) and usages else None
        output_tokens = sum(known_output) if len(known_output) == len(usages) and usages else None
        reported_total = sum(totals) if len(totals) == len(usages) and usages else None
        accounted = sum(account.conservative_charge for account in accounts)
        final = sum(usage.status is UsageStatus.FINAL for usage in usages)
        partial = sum(usage.status is UsageStatus.PARTIAL for usage in usages)
        unavailable = sum(usage.status is UsageStatus.UNAVAILABLE for usage in usages)
        estimated = any(
            usage.status is not UsageStatus.FINAL or usage.total_tokens is None for usage in usages
        )
        return SubagentUsageSnapshot(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=sum(usage.cache_read_tokens for usage in usages),
            cache_write_tokens=sum(usage.cache_write_tokens for usage in usages),
            reasoning_tokens=sum(usage.reasoning_tokens for usage in usages),
            reported_total=reported_total,
            accounted_tokens=accounted,
            limit=self._limit,
            final_requests=final,
            partial_requests=partial,
            unavailable_requests=unavailable,
            estimated=estimated,
            exhausted=accounted >= self._limit,
            exceeded=reported_total is not None and reported_total > self._limit,
        )

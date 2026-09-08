from hammer_code.domain.usage import TokenUsage, UsageLedger, UsageStatus


def test_ledger_replaces_snapshots_and_never_downgrades_final() -> None:
    ledger = UsageLedger()
    ledger.upsert("request", "turn", TokenUsage(2, 3, status=UsageStatus.PARTIAL))
    ledger.upsert("request", "turn", TokenUsage(4, 5, status=UsageStatus.FINAL))
    ledger.upsert("request", "turn", TokenUsage(8, 9, status=UsageStatus.PARTIAL))
    summary = ledger.for_turn("turn")
    assert summary.usage.total_tokens == 9
    assert summary.final_requests == 1
    assert TokenUsage(2, 3, reasoning_tokens=2).total_tokens == 5


def test_unavailable_is_not_reported_as_zero() -> None:
    ledger = UsageLedger()
    ledger.mark_unavailable("request", "turn")
    assert ledger.for_turn("turn").usage.total_tokens is None

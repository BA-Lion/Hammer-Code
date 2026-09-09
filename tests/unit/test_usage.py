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
    summary = ledger.for_turn("turn")
    assert summary.usage.total_tokens is None
    assert summary.unavailable_requests == 1


def test_empty_ledger_reports_unknown_usage_without_requests() -> None:
    ledger = UsageLedger()
    for summary in (ledger.for_turn("turn"), ledger.for_conversation()):
        assert summary.usage.total_tokens is None
        assert summary.usage.status is UsageStatus.UNAVAILABLE
        assert summary.final_requests == 0
        assert summary.partial_requests == 0
        assert summary.unavailable_requests == 0

from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.subagent.usage import SubagentUsageTracker


def test_usage_tracker_reserves_settles_and_does_not_double_count_details() -> None:
    tracker = SubagentUsageTracker(100)

    assert tracker.reserve("one", 20, 50) == 50
    reserved = tracker.snapshot()
    assert reserved.accounted_tokens == 70
    assert reserved.reported_total is None
    assert reserved.unavailable_requests == 1
    assert reserved.estimated

    tracker.observe(
        "one",
        TokenUsage(15, 5, cache_read_tokens=3, reasoning_tokens=2, status=UsageStatus.PARTIAL),
    )
    partial = tracker.snapshot()
    assert partial.reported_total == 20
    assert partial.accounted_tokens == 70
    assert partial.reasoning_tokens == 2

    tracker.observe(
        "one",
        TokenUsage(
            16,
            6,
            cache_read_tokens=4,
            cache_write_tokens=1,
            reasoning_tokens=3,
            status=UsageStatus.FINAL,
        ),
    )
    final = tracker.snapshot()
    assert final.input_tokens == 16
    assert final.output_tokens == 6
    assert final.reported_total == 22
    assert final.accounted_tokens == 22
    assert final.cache_read_tokens == 4
    assert final.cache_write_tokens == 1
    assert final.reasoning_tokens == 3
    assert final.final_requests == 1
    assert not final.estimated


def test_usage_tracker_never_downgrades_final_and_marks_exceeded() -> None:
    tracker = SubagentUsageTracker(10)
    assert tracker.reserve("one", 1, 9) == 9
    tracker.observe("one", TokenUsage(7, 5, status=UsageStatus.FINAL))
    tracker.observe("one", TokenUsage(1, 1, status=UsageStatus.PARTIAL))

    snapshot = tracker.snapshot()
    assert snapshot.reported_total == 12
    assert snapshot.accounted_tokens == 12
    assert snapshot.final_requests == 1
    assert snapshot.exhausted
    assert snapshot.exceeded


def test_usage_tracker_refuses_when_input_leaves_no_output_budget() -> None:
    tracker = SubagentUsageTracker(10)

    assert tracker.reserve("one", 10, 5) is None
    assert tracker.snapshot().accounted_tokens == 0

from io import StringIO

from rich.console import Console

from hammer_code.domain.usage import TokenUsage, UsageStatus, UsageSummary
from hammer_code.ui.console import ConsoleUI


def test_console_never_turns_unavailable_usage_into_zero() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False))
    ui.usage(UsageSummary(TokenUsage(None, None, status=UsageStatus.UNAVAILABLE), 0, 0, 1))
    assert "unavailable" in output.getvalue()


def test_console_prints_one_thinking_line_for_non_terminal_output() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False))
    ui.reasoning_status()
    ui.reasoning_status()
    assert output.getvalue().count("Thinking…") == 1


def test_console_allows_thinking_status_on_next_turn_after_usage() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False))
    summary = UsageSummary(TokenUsage(None, None, status=UsageStatus.UNAVAILABLE), 0, 0, 1)
    ui.reasoning_status()
    ui.usage(summary)
    ui.reasoning_status()
    assert output.getvalue().count("Thinking…") == 2

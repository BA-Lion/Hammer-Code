from io import StringIO

from rich.console import Console

from hammer_code.domain.usage import TokenUsage, UsageStatus, UsageSummary
from hammer_code.ui.console import ConsoleUI


def test_console_never_turns_unavailable_usage_into_zero() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False))
    ui.usage(UsageSummary(TokenUsage(None, None, status=UsageStatus.UNAVAILABLE), 0, 0, 1))
    assert "unavailable" in output.getvalue()

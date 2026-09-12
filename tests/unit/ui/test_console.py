from io import StringIO

from rich.console import Console

from hammer_code.domain.events import CompactEvent
from hammer_code.domain.messages import ReasoningVisibility, ToolCallBlock
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


def test_console_streaming_does_not_insert_line_breaks_between_cjk_deltas() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False, width=12))
    chunks = ("中文路径测试甲乙", "丙丁戊己庚辛壬癸", "以及Markdown标记")

    for chunk in chunks:
        ui.text_delta(chunk)

    assert output.getvalue() == "".join(chunks)


def test_console_reasoning_stream_uses_the_same_soft_wrap_path() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False, width=12))
    chunks = ("分析中文路径", "以及Markdown标记")

    for chunk in chunks:
        ui.reasoning_delta(chunk, ReasoningVisibility.VISIBLE)

    assert output.getvalue() == "".join(chunks)


def test_console_tool_notice_is_a_styled_block_separate_from_model_text() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=True, color_system="standard", width=80))
    call = ToolCallBlock("call-1", "read_file", {"path": "README.md"}, '{"path":"README.md"}')

    ui.text_delta("模型过程")
    ui.tool_call_notice(call)
    ui.text_delta("最终回复")
    ui.usage(UsageSummary(TokenUsage(None, None, status=UsageStatus.UNAVAILABLE), 0, 0, 1))

    rendered = output.getvalue()
    assert "模型过程\n" in rendered
    assert "Tool" in rendered
    assert "read_file" in rendered
    assert "\x1b[" in rendered
    assert "最终回复\n" in rendered


def test_console_coalesces_post_tool_token_deltas_until_the_next_block() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False, width=80))
    call = ToolCallBlock("call-1", "read_file", {"path": "README.md"}, '{"path":"README.md"}')
    chunks = ("没人", "收", "」", "）", "都", "是", "我", "新", "拟", "的", "。")

    ui.tool_call_notice(call)
    after_tool = output.getvalue()
    for chunk in chunks:
        ui.text_delta(chunk)

    assert output.getvalue() == after_tool

    ui.usage(UsageSummary(TokenUsage(None, None, status=UsageStatus.UNAVAILABLE), 0, 0, 1))
    assert "".join(chunks) + "\nUsage:" in output.getvalue()


def test_console_flushes_a_coalesced_stream_after_reaching_terminal_width() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False, width=12))
    chunks = ("中文", "路径", "测试")

    for chunk in chunks:
        ui.text_delta(chunk)

    assert output.getvalue() == "".join(chunks)


def test_console_displays_context_estimates_as_a_compaction_block() -> None:
    output = StringIO()
    ui = ConsoleUI(Console(file=output, force_terminal=False))
    ui.compact(CompactEvent(100, 20, 80))
    assert "Compacted context: ~100 → ~20 tokens; saved ~80" in output.getvalue()

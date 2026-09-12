from pathlib import Path

from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import ContextConfig
from hammer_code.domain.messages import TextBlock, ToolResultBlock
from hammer_code.tools.results import bound_tool_results
from hammer_code.tools.runtime import RuntimeStore


def test_oversized_result_is_stored_with_address_before_preview(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        tool_result_overflow_tokens=5,
        tool_result_preview_tokens=3,
        stale_tool_result_tokens=1,
        tool_batch_tokens=20,
    )
    try:
        result = ToolResultBlock("call", (TextBlock("中" * 40),), False)
        bounded = bound_tool_results((result,), runtime, config, TokenEstimator())
        text = bounded[0].content[0].text
        assert text.startswith(".hammer-code/tmp/")
        assert "truncated" in text.lower()
        assert runtime.result_path("call").exists()
    finally:
        runtime.cleanup()


def test_batch_bounding_preserves_every_call_id_and_order(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path)
    config = ContextConfig(
        tool_result_overflow_tokens=3,
        tool_result_preview_tokens=2,
        stale_tool_result_tokens=1,
        tool_batch_tokens=18,
    )
    try:
        results = tuple(
            ToolResultBlock(str(index), (TextBlock("output " * 20),), index == 1)
            for index in range(3)
        )
        bounded = bound_tool_results(results, runtime, config, TokenEstimator())
        assert [item.call_id for item in bounded] == ["0", "1", "2"]
        assert [item.is_error for item in bounded] == [False, True, False]
    finally:
        runtime.cleanup()

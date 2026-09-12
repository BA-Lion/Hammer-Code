from pathlib import Path

from hammer_code.tools.runtime import RuntimeStore


def test_runtime_store_is_session_scoped_and_clear_can_continue_writing(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path)
    try:
        first = runtime.write_result("call", "first")
        assert first.parent == runtime.results_dir
        assert runtime.write_result("call", "second").read_text(encoding="utf-8") == "second"
        assert runtime.delete_results(("missing", "call")) == ()
        assert not first.exists()
        assert runtime.clear_results()
        assert runtime.write_result("new", "value").exists()
    finally:
        runtime.cleanup()

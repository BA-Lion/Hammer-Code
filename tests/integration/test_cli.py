import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

from hammer_code import cli
from hammer_code.cli import main
from hammer_code.config import AppConfig, resolve_profile


def test_help_requires_no_config_or_key() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_cli_builds_a_no_tool_prompt_for_chat_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project" / ".hammer-code" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_chat_completions",
                    "model": "test-model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "TEST_KEY",
                    "max_output_tokens": 10,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    resolved = resolve_profile(config, None, {"TEST_KEY": "not-in-prompt"})
    captured: dict[str, str] = {}

    class FakeUI:
        def confirmation(self, _: str) -> bool:
            return True

        def banner(self, *_: object) -> None:
            pass

    class FakeClient:
        async def aclose(self) -> None:
            pass

    class FakeChatLoop:
        def __init__(self, *_: object) -> None:
            captured["system_prompt"] = _[3]  # type: ignore[assignment]

        async def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "ConsoleUI", FakeUI)
    monkeypatch.setattr(cli, "discover_config", lambda _: config_path)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "resolve_profile", lambda *_: resolved)
    monkeypatch.setattr(cli, "create_model_client", lambda _: FakeClient())
    monkeypatch.setattr(cli, "ChatLoop", FakeChatLoop)

    assert asyncio.run(cli._run(Namespace(config=None, profile=None))) == 0
    prompt = captured["system_prompt"]
    assert f"Project root: {config_path.parent.parent.resolve()}" in prompt
    assert "Enabled tools: create_file, edit_file, glob, grep, read_file, shell" in prompt
    assert "permissions are checked separately" in prompt

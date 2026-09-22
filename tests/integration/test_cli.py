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

        def skill_warning(self, message: str) -> None:
            del message

    class FakeClient:
        async def aclose(self) -> None:
            pass

    class FakeChatLoop:
        def __init__(self, *_: object) -> None:
            captured["system_prompt"] = _[0].system_prompt  # type: ignore[union-attr]

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
    assert "Enabled tools:" in prompt
    for name in (
        "create_file",
        "edit_file",
        "glob",
        "grep",
        "inspect_subagent_worktree",
        "read_file",
        "resolve_subagent_worktree",
    ):
        assert name in prompt
    assert "run_subagent" in prompt
    assert "subagent_task" not in prompt
    assert "permissions are checked separately" in prompt


def test_cli_closes_shared_resources_when_initial_agent_construction_fails(
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
    events: list[str] = []

    class FakeUI:
        def confirmation(self, _: str) -> bool:
            return True

        def error(self, message: str) -> None:
            events.append(message)

    class FakeClient:
        async def aclose(self) -> None:
            events.append("client closed")

    class FakeMcpManager:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def close(self) -> None:
            events.append("mcp closed")

    class FailingFactory:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def create_new(self) -> object:
            from hammer_code.errors import HammerCodeError

            raise HammerCodeError("initial session failed")

    monkeypatch.setattr(cli, "ConsoleUI", FakeUI)
    monkeypatch.setattr(cli, "discover_config", lambda _: config_path)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "resolve_profile", lambda *_: resolved)
    monkeypatch.setattr(cli, "create_model_client", lambda _: FakeClient())
    monkeypatch.setattr(cli, "McpManager", FakeMcpManager)
    monkeypatch.setattr(cli, "PrimaryAgentFactory", FailingFactory)

    assert asyncio.run(cli._run(Namespace(config=None, profile=None))) == 2
    assert events == ["mcp closed", "client closed", "initial session failed"]

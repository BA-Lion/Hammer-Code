from __future__ import annotations

from pathlib import Path

import pytest

from hammer_code import cli


def configuration(root: Path) -> Path:
    path = root / ".hammer-code/config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        'default_profile="x"\n[profiles.x]\nprotocol="openai_chat_completions"\n'
        'model="m"\nbase_url="https://api.openai.com/v1"\n'
        'api_key_env="MISSING_CHAT_KEY"\nmax_output_tokens=100\ntimeout_seconds=5\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("action", ["--prepare-skill-model", "--skill-vector-status"])
async def test_local_actions_need_no_chat_key_or_session(
    tmp_path: Path, monkeypatch, action: str
) -> None:
    path = configuration(tmp_path)
    monkeypatch.delenv("MISSING_CHAT_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("local action must not initialize model clients, sessions or migrations")

    monkeypatch.setattr(cli, "create_model_client", forbidden)
    monkeypatch.setattr(cli, "SessionManager", forbidden)
    monkeypatch.setattr(cli, "migrate_local_state", forbidden)
    roots = []

    async def prepare(root):
        roots.append(root)

    monkeypatch.setattr(cli, "prepare_model", prepare)
    assert await cli._run(cli.parser().parse_args(["--config", str(path), action])) == 0
    assert roots == ([tmp_path] if action == "--prepare-skill-model" else [])
    assert not (tmp_path / ".hammer-code/sessions").exists()


def test_local_actions_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--prepare-skill-model", "--resume", "latest"])


async def test_cleanup_closes_all_after_failure() -> None:
    order = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def aclose(self):
            order.append(self.name)
            if self.name == "vector":
                raise RuntimeError("injected")

        close = aclose

    await cli._close_resources(
        *(Resource(name) for name in ("vector", "mcp", "worktree", "client"))
    )
    assert order == ["vector", "mcp", "worktree", "client"]

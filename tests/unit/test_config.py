from pathlib import Path

import pytest

from hammer_code.config import EndpointTrustPolicy, discover_config, load_config, resolve_profile
from hammer_code.errors import ConfigurationError, UntrustedEndpointError


def _config(base_url: str = "https://api.openai.com/v1") -> str:
    return f'''default_profile = "main"
[profiles.main]
protocol = "openai_responses"
model = "model"
base_url = "{base_url}"
api_key_env = "TEST_KEY"
max_output_tokens = 10
timeout_seconds = 5
max_retries = 0
'''


def test_load_config_rejects_unknown_and_plaintext_secret(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_config() + "api_key = 'leak'\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_discovery_and_secret_resolution(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    config_path = root / ".hammer-code" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(_config(), encoding="utf-8")
    config = load_config(discover_config(None, root / "child"))
    resolved = resolve_profile(config, None, {"TEST_KEY": "value"})
    assert resolved.api_key.get_secret_value() == "value"
    assert "value" not in repr(resolved)


def test_endpoint_policy_rejects_remote_http_and_confirms_custom() -> None:
    from hammer_code.config import AppConfig

    remote = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_responses",
                    "model": "x",
                    "base_url": "http://example.test",
                    "api_key_env": "K",
                    "max_output_tokens": 1,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    ).profiles["x"]
    with pytest.raises(UntrustedEndpointError):
        EndpointTrustPolicy().validate_and_confirm(remote, lambda _: True)
    custom = remote.model_copy(update={"base_url": "https://example.test"})
    with pytest.raises(UntrustedEndpointError):
        EndpointTrustPolicy().validate_and_confirm(custom, lambda _: False)


def test_mcp_config_is_strict_and_last_duplicate_keeps_its_final_order() -> None:
    from hammer_code.config import AppConfig, McpHttpConfig, McpStdioConfig

    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_responses",
                    "model": "model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "TEST_KEY",
                    "max_output_tokens": 10,
                    "timeout_seconds": 5,
                    "max_retries": 0,
                }
            },
            "mcp": [
                {
                    "name": "first",
                    "description": "first server",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                    "env": {"TOKEN": "MCP_TOKEN"},
                },
                {
                    "name": "second",
                    "description": "second server",
                    "transport": "streamable_http",
                    "endpoint": "http://127.0.0.1:8000/mcp",
                },
                {
                    "name": "first",
                    "description": "replacement",
                    "transport": "stdio",
                    "command": "node",
                },
            ],
        }
    )
    assert [item.name for item in config.mcp] == ["second", "first"]
    assert isinstance(config.mcp[0], McpHttpConfig)
    assert config.mcp[0].bearer_token_env is None
    assert isinstance(config.mcp[1], McpStdioConfig)
    assert config.mcp[1].command == "node"


@pytest.mark.parametrize(
    "mcp",
    [
        {
            "name": "bad name",
            "description": "valid",
            "transport": "stdio",
            "command": "python",
        },
        {
            "name": "good",
            "description": "two\nlines",
            "transport": "stdio",
            "command": "python",
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "streamable_http",
            "endpoint": "http://example.test/mcp",
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "streamable_http",
            "endpoint": "https://example.test/mcp",
            "bearer_token_env": "invalid-name",
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "streamable_http",
            "endpoint": "https://example.test/mcp",
            "bearer_token_env": " ",
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "streamable_http",
            "endpoint": "https://example.test/mcp",
            "bearer_token": "plaintext-is-not-allowed",
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "streamable_http",
            "endpoint": "https://example.test/mcp",
            "env": {"TOKEN": "SOURCE_TOKEN"},
        },
        {
            "name": "good",
            "description": "valid",
            "transport": "stdio",
            "command": "python",
            "env": {"bad-key": "TOKEN"},
        },
    ],
)
def test_mcp_config_rejects_unsafe_values(mcp: dict[str, object]) -> None:
    from hammer_code.config import AppConfig

    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "default_profile": "main",
                "profiles": {
                    "main": {
                        "protocol": "openai_responses",
                        "model": "model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "TEST_KEY",
                        "max_output_tokens": 10,
                        "timeout_seconds": 5,
                        "max_retries": 0,
                    }
                },
                "mcp": [mcp],
            }
        )


def test_mcp_http_bearer_environment_reference_is_retained_without_resolving() -> None:
    from hammer_code.config import McpHttpConfig

    config = McpHttpConfig(
        name="authenticated",
        description="Authenticated MCP",
        transport="streamable_http",
        endpoint="https://example.test/mcp",
        bearer_token_env="EXAMPLE_MCP_TOKEN",
    )
    assert config.bearer_token_env == "EXAMPLE_MCP_TOKEN"


def test_skill_evolution_config_is_opt_in_and_validates_prune_and_merge_relationships() -> None:
    from hammer_code.config import SkillEvolutionConfig

    assert not SkillEvolutionConfig().enabled
    with pytest.raises(ValueError):
        SkillEvolutionConfig(forced_merge_score=0.5, forced_merge_margin=0.5)
    with pytest.raises(ValueError):
        SkillEvolutionConfig(prune_min_retrieve=5, prune_min_relevant=6)


def test_subagent_config_is_bounded_and_defaulted() -> None:
    from hammer_code.config import AppConfig, SubagentConfig

    assert SubagentConfig().default_max_iterations == 20
    assert SubagentConfig().max_background_tasks == 4
    assert SubagentConfig().max_task_tokens == 200_000
    with pytest.raises(ValueError):
        SubagentConfig(default_max_iterations=51)
    with pytest.raises(ValueError):
        SubagentConfig(max_task_tokens=999)
    with pytest.raises(ValueError):
        SubagentConfig(max_task_tokens=10_000_001)
    config = AppConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {
                "main": {
                    "protocol": "openai_responses",
                    "model": "model",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "TEST_KEY",
                    "max_output_tokens": 10,
                    "timeout_seconds": 5,
                    "max_retries": 0,
                }
            },
            "subagent": {
                "default_max_iterations": 5,
                "max_background_tasks": 2,
                "max_task_tokens": 123_456,
            },
        }
    )
    assert config.subagent.default_max_iterations == 5
    assert config.subagent.max_task_tokens == 123_456


def test_agent_team_config_is_disabled_and_bounded_by_default() -> None:
    from hammer_code.config import AgentTeamConfig

    assert not AgentTeamConfig().coordination_mode
    assert AgentTeamConfig().max_team_tokens == 1_000_000
    with pytest.raises(ValueError):
        AgentTeamConfig(max_team_tokens=9_999)
    with pytest.raises(ValueError):
        AgentTeamConfig(max_active_assignments=17)


def test_deprecated_subagent_task_disabled_name_remains_compatible() -> None:
    from hammer_code.config import ToolConfig

    assert ToolConfig(disabled=("subagent_task",)).disabled == ("subagent_task",)

from hammer_code.config import AppConfig, resolve_profile
from hammer_code.llm.factory import create_model_client


def test_factory_selects_exact_protocol() -> None:
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_chat_completions",
                    "model": "m",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "K",
                    "max_output_tokens": 2,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    assert (
        create_model_client(resolve_profile(config, None, {"K": "x"})).capabilities.protocol
        == "openai_chat_completions"
    )

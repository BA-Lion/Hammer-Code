"""Safe command-line startup sequence."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

from hammer_code.app.chat_loop import ChatLoop
from hammer_code.config import EndpointTrustPolicy, discover_config, load_config, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.errors import HammerCodeError
from hammer_code.llm.factory import create_model_client
from hammer_code.prompts import INITIAL_SYSTEM_PROMPT
from hammer_code.ui.console import ConsoleUI


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="hammer-code", description="Hammer Code protocol-neutral CLI"
    )
    value.add_argument("--config", type=Path, help="Path to a project config.toml")
    value.add_argument("--profile", help="Override the configured profile")
    return value


async def _run(args: argparse.Namespace) -> int:
    ui = ConsoleUI()
    try:
        path = discover_config(args.config)
        config = load_config(path)
        profile_name = args.profile or config.default_profile
        profile = config.profiles.get(profile_name)
        if profile is None:
            raise HammerCodeError(f"Unknown profile: {profile_name}")
        EndpointTrustPolicy().validate_and_confirm(profile, ui.confirmation)
        resolved = resolve_profile(config, args.profile)
        client = create_model_client(resolved)
    except HammerCodeError as exc:
        ui.error(str(exc))
        return 2
    manager = ConversationManager()
    manager.create(resolved)
    ui.banner(
        resolved.name,
        resolved.profile.protocol,
        resolved.profile.model,
        urlparse(resolved.profile.base_url).netloc,
    )
    try:
        await ChatLoop(
            manager,
            client,
            ui,
            INITIAL_SYSTEM_PROMPT,
            resolved.profile.max_output_tokens,
            config.ui.show_reasoning,
        ).run()
    finally:
        await client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return asyncio.run(_run(args))

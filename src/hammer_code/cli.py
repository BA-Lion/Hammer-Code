"""Safe command-line startup sequence."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

from hammer_code.app.chat_loop import ChatLoop
from hammer_code.app.context_window import ContextWindow
from hammer_code.config import EndpointTrustPolicy, discover_config, load_config, resolve_profile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.errors import HammerCodeError
from hammer_code.llm.factory import create_model_client
from hammer_code.mcp.manager import McpManager
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.rules import RuleStore
from hammer_code.permissions.service import PermissionService
from hammer_code.prompts import PromptRuntimeContext, build_system_prompt
from hammer_code.tools.base import ToolExecutionContext
from hammer_code.tools.builtin import (
    CreateFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    ShellTool,
    ToolSearchTool,
)
from hammer_code.tools.builtin.shell import sanitized_environment
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore
from hammer_code.ui.console import ConsoleUI


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="hammer-code", description="Hammer Code protocol-neutral CLI"
    )
    value.add_argument("--config", type=Path, help="Path to a project config.toml")
    value.add_argument("--profile", help="Override the configured profile")
    value.add_argument(
        "--permission-mode",
        choices=tuple(mode.value for mode in PermissionMode),
        default=PermissionMode.DEFAULT.value,
        help="Local tool permission mode",
    )
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
        mode = PermissionMode(getattr(args, "permission_mode", PermissionMode.DEFAULT.value))
        if mode is PermissionMode.UNATTENDED and not ui.confirmation(
            "Unattended mode permits approved Shell commands without an OS sandbox. Continue?"
        ):
            raise HammerCodeError("Unattended mode was not confirmed")
        resolved = resolve_profile(config, args.profile)
        workspace_root = path.parent.parent.resolve()
        runtime = RuntimeStore(workspace_root)
        runtime.cleanup_old()
        registry = ToolRegistry(config.tools.disabled)
        for tool in (
            ReadFileTool(),
            EditFileTool(),
            CreateFileTool(),
            GrepTool(),
            GlobTool(),
            ShellTool(),
        ):
            registry.register(tool)
        rules = RuleStore(workspace_root)
        permissions = PermissionService(PermissionChecker(mode, rules), ui, rules)
        context = ToolExecutionContext(
            workspace_root, Path.cwd().resolve(), runtime.session_dir, sanitized_environment()
        )
        executor = ToolExecutor(registry, permissions, context, runtime)
        mcp_manager = McpManager(config.mcp, registry, workspace_root, ui=ui)
        registry.register(ToolSearchTool(registry, mcp_manager))
        system_prompt = build_system_prompt(
            PromptRuntimeContext(
                cwd=Path.cwd().resolve(),
                project_root=workspace_root,
                platform=sys.platform,
                shell_backend="PowerShell",
                permission_mode=mode.value,
                enabled_tools=registry.exposed_names(),
            )
        )
        context_window = ContextWindow(system_prompt)
        client = create_model_client(resolved)
    except HammerCodeError as exc:
        ui.error(str(exc))
        return 2
    except OSError:
        ui.error("Unable to resolve the project runtime paths.")
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
        mcp_manager.start()
        await ChatLoop(
            manager,
            client,
            ui,
            system_prompt,
            resolved.profile.max_output_tokens,
            config.ui.show_reasoning,
            registry,
            executor,
            context_window,
            mcp_manager,
        ).run()
    finally:
        try:
            await mcp_manager.close()
        finally:
            await client.aclose()
            runtime.cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return asyncio.run(_run(args))

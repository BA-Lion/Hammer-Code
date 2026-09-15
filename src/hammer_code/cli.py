"""Safe command-line startup sequence."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

from hammer_code.app.chat_loop import ChatLoop
from hammer_code.app.commands import CommandRegistry, register_builtin_commands
from hammer_code.app.runtime import PrimaryAgentFactory
from hammer_code.config import EndpointTrustPolicy, discover_config, load_config, resolve_profile
from hammer_code.errors import HammerCodeError
from hammer_code.llm.factory import create_model_client
from hammer_code.mcp.manager import McpManager
from hammer_code.memory.store import MemoryStore
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import PermissionMode
from hammer_code.permissions.rules import RuleStore
from hammer_code.permissions.service import PermissionService
from hammer_code.project_instructions import ProjectInstructionLoader
from hammer_code.session.manager import SessionManager
from hammer_code.tools.builtin import (
    CreateFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    ShellTool,
    ToolSearchTool,
)
from hammer_code.tools.registry import ToolRegistry
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
    actions = value.add_mutually_exclusive_group()
    actions.add_argument("--resume", metavar="ID_OR_LATEST", help="Resume a compatible session")
    actions.add_argument(
        "--list-sessions", action="store_true", help="List saved sessions and exit"
    )
    return value


async def _run(args: argparse.Namespace) -> int:
    ui = ConsoleUI()
    client = None
    mcp_manager = None
    try:
        path = discover_config(args.config)
        config = load_config(path)
        workspace_root = path.parent.parent.resolve()
        sessions = SessionManager(workspace_root)
        await asyncio.to_thread(sessions.cleanup)
        if getattr(args, "list_sessions", False):
            ui.sessions(await asyncio.to_thread(sessions.list))
            return 0
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
        client = create_model_client(resolved)
        mcp_manager = McpManager(config.mcp, registry, workspace_root, ui=ui)
        registry.register(ToolSearchTool(registry, mcp_manager))
        factory = PrimaryAgentFactory(
            workspace_root=workspace_root,
            cwd=Path.cwd(),
            config=config,
            resolved=resolved,
            client=client,
            ui=ui,
            registry=registry,
            mcp_manager=mcp_manager,
            permissions=permissions,
            sessions=sessions,
            memory_store=MemoryStore(workspace_root),
            maintenance_lock=asyncio.Lock(),
            project_instructions=ProjectInstructionLoader(workspace_root).load(),
            show_reasoning=config.ui.show_reasoning,
        )
        resume_id = getattr(args, "resume", None)
        current = await (factory.resume(resume_id) if resume_id else factory.create_new())
    except HammerCodeError as exc:
        if mcp_manager is not None:
            try:
                await mcp_manager.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        ui.error(str(exc))
        return 2
    except OSError:
        if mcp_manager is not None:
            try:
                await mcp_manager.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        ui.error("Unable to resolve the project runtime paths.")
        return 2
    ui.banner(
        resolved.name,
        resolved.profile.protocol,
        resolved.profile.model,
        urlparse(resolved.profile.base_url).netloc,
    )
    commands = CommandRegistry()
    register_builtin_commands(commands)
    try:
        mcp_manager.start()
        await ChatLoop(current, factory, commands, sessions, factory.memory_store, ui).run()
    finally:
        try:
            await mcp_manager.close()
        finally:
            await client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(parser().parse_args(argv)))

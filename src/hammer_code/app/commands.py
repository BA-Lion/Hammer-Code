"""Narrow, local slash-command parsing and dispatch primitives."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from hammer_code.errors import CommandError, HammerCodeError
from hammer_code.memory.models import MemoryCategory
from hammer_code.permissions.models import PermissionMode

if TYPE_CHECKING:
    from hammer_code.app.agent import PrimaryAgent  # type: ignore[reportMissingImports]
    from hammer_code.ui.console import ConsolePort


_COMMAND_NAME = re.compile(r"[a-z][a-z0-9-]*")
_COMMAND_INPUT = re.compile(r"/([a-z][a-z0-9-]*)(?:[ \t]+(.*))?")


@dataclass(frozen=True)
class CommandInvocation:
    """A command name with its still-unparsed argument text."""

    name: str
    arguments: str


class CommandAction(StrEnum):
    CONTINUE = "continue"
    EXIT = "exit"
    NEW_SESSION = "new_session"
    RESUME_SESSION = "resume_session"


@dataclass(frozen=True)
class CommandOutcome:
    """A handler's requested lifecycle transition for the outer chat loop."""

    action: CommandAction = CommandAction.CONTINUE
    session_id: str | None = None


class CommandHost(Protocol):
    """The controlled application capabilities available to command handlers.

    The concrete outer loop supplies these operations.  In particular, this
    deliberately excludes ConversationManager, configuration, and writable
    MemoryStore access.
    """

    async def list_sessions(self) -> tuple[object, ...]: ...

    async def delete_session(self, session_id: str) -> None: ...

    def is_session_draining(self, session_id: str) -> bool: ...

    async def list_memory(self, category: str | None = None) -> tuple[object, ...]: ...

    async def read_memory(self, category: str, relative_path: str) -> str: ...


@dataclass(frozen=True)
class CommandContext:
    agent: PrimaryAgent
    host: CommandHost
    ui: ConsolePort


CommandHandler = Callable[[CommandContext, CommandInvocation], Awaitable[CommandOutcome]]


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    hidden: bool = False


def _validate_name(name: str) -> None:
    if _COMMAND_NAME.fullmatch(name) is None:
        raise CommandError(f"Invalid command name: {name!r}")


def parse_command(text: str) -> CommandInvocation | None:
    """Parse a local command without interpreting its arguments.

    Boundary whitespace is discarded.  Once the input starts with ``/``, an
    invalid command is always an error rather than a model-bound text message.
    """

    normalized = text.strip()
    if not normalized.startswith("/"):
        return None
    match = _COMMAND_INPUT.fullmatch(normalized)
    if match is None:
        raise CommandError("Invalid command. Use /help.")
    return CommandInvocation(name=match.group(1), arguments=match.group(2) or "")


class CommandRegistry:
    """A deterministic registry of canonical command names and aliases."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: Command) -> None:
        _validate_name(command.name)
        claimed_names = (command.name, *command.aliases)
        if len(set(claimed_names)) != len(claimed_names):
            raise CommandError(f"Command {command.name!r} repeats a name or alias")
        for name in claimed_names:
            _validate_name(name)
            if name in self._commands or name in self._aliases:
                raise CommandError(f"Command name or alias already registered: {name!r}")
        self._commands[command.name] = command
        self._aliases.update({alias: command.name for alias in command.aliases})

    def resolve(self, name: str) -> Command | None:
        canonical_name = self._aliases.get(name, name)
        return self._commands.get(canonical_name)

    def visible(self) -> tuple[Command, ...]:
        return tuple(command for _, command in sorted(self._commands.items()) if not command.hidden)

    def prefix_matches(self, prefix: str) -> tuple[Command, ...]:
        canonical_names = {
            canonical for name, canonical in self._aliases.items() if name.startswith(prefix)
        }
        canonical_names.update(name for name in self._commands if name.startswith(prefix))
        return tuple(
            command
            for name in sorted(canonical_names)
            if not (command := self._commands[name]).hidden
        )


async def dispatch(
    registry: CommandRegistry,
    context: CommandContext,
    invocation: CommandInvocation,
) -> CommandOutcome:
    """Run one known local command and turn expected failures into safe UI text."""

    command = registry.resolve(invocation.name)
    if command is None:
        matches = registry.prefix_matches(invocation.name)
        if matches:
            suggestions = ", ".join(f"/{item.name}" for item in matches)
            context.ui.error(f"Unknown command '/{invocation.name}'. Try: {suggestions}.")
        else:
            context.ui.error(f"Unknown command '/{invocation.name}'. Use /help.")
        return CommandOutcome()
    try:
        return await command.handler(context, invocation)
    except (CommandError, HammerCodeError) as exc:
        context.ui.error(str(exc))
        return CommandOutcome()


def register_builtin_commands(registry: CommandRegistry) -> None:
    """Register the small, explicitly supported local command surface."""

    async def help_handler(
        context: CommandContext, invocation: CommandInvocation
    ) -> CommandOutcome:
        return await _help(registry, context, invocation)

    registry.register(Command("help", "Show local commands", "/help [command]", help_handler))
    registry.register(Command("status", "Show current runtime status", "/status", _status))
    registry.register(Command("usage", "Show token usage", "/usage", _usage))
    registry.register(Command("clear", "Clear the current conversation", "/clear", _clear))
    registry.register(Command("compact", "Compact the current context", "/compact", _compact))
    registry.register(
        Command(
            "permission",
            "Change the runtime permission mode",
            "/permission [default|accept_edits|strict|unattended]",
            _permission,
        )
    )
    registry.register(
        Command("session", "Manage sessions", "/session list|new|resume|delete", _session)
    )
    registry.register(Command("memory", "Read project memory", "/memory list|read", _memory))
    registry.register(Command("exit", "Exit Hammer Code", "/exit", _exit, aliases=("quit",)))


async def _help(
    registry: CommandRegistry, context: CommandContext, invocation: CommandInvocation
) -> CommandOutcome:
    if not invocation.arguments:
        lines = [
            f"/{command.name}  {command.usage}  {command.description}"
            for command in registry.visible()
        ]
        context.ui.info("\n".join(lines))
        return CommandOutcome()
    command = registry.resolve(invocation.arguments.strip())
    if command is None or command.hidden:
        raise CommandError("Unknown command. Use /help.")
    aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
    context.ui.info(f"/{command.name}{aliases}\n{command.usage}\n{command.description}")
    return CommandOutcome()


async def _status(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    if invocation.arguments:
        raise CommandError("Usage: /status")
    status = context.agent.status()
    context.ui.info(
        f"profile={status.profile}; protocol={status.protocol}; model={status.model}; "
        f"session={status.session_id}; permission={status.permission_mode.value}; "
        f"persistence_degraded={status.persistence_degraded}"
    )
    context.ui.usage(status.usage)
    return CommandOutcome()


async def _usage(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    if invocation.arguments:
        raise CommandError("Usage: /usage")
    context.ui.usage(context.agent.status().usage)
    return CommandOutcome()


async def _clear(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    if invocation.arguments:
        raise CommandError("Usage: /clear")
    await context.agent.clear()
    return CommandOutcome()


async def _compact(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    if invocation.arguments:
        raise CommandError("Usage: /compact")
    await context.agent.compact()
    return CommandOutcome()


async def _permission(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    value = invocation.arguments.strip()
    if not value:
        current = context.agent.status().permission_mode
        selected = await context.ui.choose(
            f"Permission mode (current: {current.value})",
            tuple(mode.value for mode in PermissionMode),
        )
        if selected is None:
            return CommandOutcome()
        value = selected
    try:
        mode = PermissionMode(value)
    except ValueError as exc:
        raise CommandError("Usage: /permission [default|accept_edits|strict|unattended]") from exc
    if mode is PermissionMode.UNATTENDED and not await context.ui.confirm(
        "Unattended mode permits approved Shell commands without an OS sandbox. Continue?"
    ):
        return CommandOutcome()
    context.agent.set_permission_mode(mode)
    context.ui.info(f"Permission mode: {mode.value}")
    return CommandOutcome()


async def _session(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    parts = invocation.arguments.split(maxsplit=1)
    if not parts or parts[0] == "list":
        if len(parts) > 1:
            raise CommandError("Usage: /session list|new|resume|delete")
        items = await context.host.list_sessions()
        context.ui.sessions(items)
        return CommandOutcome()
    action = parts[0]
    if action == "new":
        if len(parts) != 1:
            raise CommandError("Usage: /session new")
        return CommandOutcome(CommandAction.NEW_SESSION)
    if action == "resume" and len(parts) == 2 and parts[1] and " " not in parts[1]:
        return CommandOutcome(CommandAction.RESUME_SESSION, parts[1])
    if action == "delete" and len(parts) == 2 and parts[1] and " " not in parts[1]:
        session_id = parts[1]
        if session_id == context.agent.session_id or context.host.is_session_draining(session_id):
            raise CommandError("Cannot delete the current or draining session")
        if not await context.ui.confirm(f"Delete session {session_id}? This cannot be undone."):
            return CommandOutcome()
        if session_id == context.agent.session_id or context.host.is_session_draining(session_id):
            raise CommandError("Cannot delete the current or draining session")
        await context.host.delete_session(session_id)
        context.ui.info(f"Deleted session {session_id}.")
        return CommandOutcome()
    raise CommandError("Usage: /session list|new|resume|delete")


async def _memory(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    parts = invocation.arguments.split(maxsplit=2)
    if not parts or parts[0] == "list":
        if len(parts) > 2:
            raise CommandError("Usage: /memory list [category]")
        category = parts[1] if len(parts) == 2 else None
        if category is not None:
            try:
                MemoryCategory(category)
            except ValueError as exc:
                raise CommandError("Unknown Memory category") from exc
        items = await context.host.list_memory(category)
        text = "\n".join(str(item) for item in items)
        context.ui.info(_bounded_memory_text(text))
        return CommandOutcome()
    if parts[0] == "read" and len(parts) == 3:
        try:
            category = MemoryCategory(parts[1]).value
        except ValueError as exc:
            raise CommandError("Unknown Memory category") from exc
        text = await context.host.read_memory(category, parts[2])
        context.ui.info(_bounded_memory_text(text))
        return CommandOutcome()
    raise CommandError("Usage: /memory list [category] | /memory read <category> <relative-path>")


async def _exit(context: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
    if invocation.arguments:
        raise CommandError("Usage: /exit")
    return CommandOutcome(CommandAction.EXIT)


def _bounded_memory_text(text: str) -> str:
    return text[:20_000] + ("\n[Memory output truncated]" if len(text) > 20_000 else "")

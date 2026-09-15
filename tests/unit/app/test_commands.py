"""Tests for local command parsing, discovery, and safe dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hammer_code.app.commands import (
    Command,
    CommandAction,
    CommandContext,
    CommandError,
    CommandInvocation,
    CommandOutcome,
    CommandRegistry,
    dispatch,
    parse_command,
)
from hammer_code.errors import HammerCodeError


@dataclass
class FakeUI:
    errors: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)


async def _continue(_: CommandContext, __: CommandInvocation) -> CommandOutcome:
    return CommandOutcome()


def command(name: str, *, aliases: tuple[str, ...] = (), hidden: bool = False) -> Command:
    return Command(name, f"{name} description", f"/{name}", _continue, aliases, hidden)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", None),
        ("   plain text   ", None),
        (" /help ", CommandInvocation("help", "")),
        ("/memory list project  notes", CommandInvocation("memory", "list project  notes")),
        ("/status\tverbose", CommandInvocation("status", "verbose")),
    ],
)
def test_parse_command_returns_plain_text_or_raw_arguments(
    text: str, expected: CommandInvocation | None
) -> None:
    assert parse_command(text) == expected


@pytest.mark.parametrize("text", ["/", "//help", "/Help", "/help_me", "/help/x", "/help\nnext"])
def test_parse_command_rejects_invalid_slash_input(text: str) -> None:
    with pytest.raises(CommandError, match="Invalid command"):
        parse_command(text)


def test_registry_resolves_aliases_and_sorts_visible_and_prefix_matches() -> None:
    registry = CommandRegistry()
    registry.register(command("status", aliases=("state",)))
    registry.register(command("secret", hidden=True))
    registry.register(command("session", aliases=("sessions",)))

    assert registry.resolve("state") is registry.resolve("status")
    assert [item.name for item in registry.visible()] == ["session", "status"]
    assert [item.name for item in registry.prefix_matches("s")] == ["session", "status"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (command("help"), command("help")),
        (command("help", aliases=("h",)), command("status", aliases=("h",))),
        (command("help", aliases=("h",)), command("h")),
        (command("help"), command("status", aliases=("help",))),
    ],
)
def test_registry_rejects_canonical_and_alias_conflicts(first: Command, second: Command) -> None:
    registry = CommandRegistry()
    registry.register(first)
    with pytest.raises(CommandError, match="already registered"):
        registry.register(second)


@pytest.mark.parametrize("bad", ["Help", "with_space", "", "-invalid"])
def test_registry_rejects_invalid_names(bad: str) -> None:
    registry = CommandRegistry()
    with pytest.raises(CommandError, match="Invalid command name"):
        registry.register(command(bad))


@pytest.mark.asyncio
async def test_dispatch_invokes_handler_and_returns_lifecycle_outcome() -> None:
    seen: list[CommandInvocation] = []

    async def handler(_: CommandContext, invocation: CommandInvocation) -> CommandOutcome:
        seen.append(invocation)
        return CommandOutcome(CommandAction.EXIT)

    registry = CommandRegistry()
    registry.register(Command("exit", "Leave", "/exit", handler))
    ui = FakeUI()
    context = CommandContext(agent=None, host=None, ui=ui)  # type: ignore[arg-type]

    outcome = await dispatch(registry, context, CommandInvocation("exit", ""))

    assert outcome.action is CommandAction.EXIT
    assert seen == [CommandInvocation("exit", "")]
    assert ui.errors == []


@pytest.mark.asyncio
async def test_dispatch_reports_expected_handler_error_and_continues() -> None:
    async def handler(_: CommandContext, __: CommandInvocation) -> CommandOutcome:
        raise HammerCodeError("Safe failure")

    registry = CommandRegistry()
    registry.register(Command("status", "Status", "/status", handler))
    ui = FakeUI()
    context = CommandContext(agent=None, host=None, ui=ui)  # type: ignore[arg-type]

    assert await dispatch(registry, context, CommandInvocation("status", "")) == CommandOutcome()
    assert ui.errors == ["Safe failure"]


@pytest.mark.asyncio
async def test_dispatch_suggests_prefix_or_help_for_unknown_command() -> None:
    registry = CommandRegistry()
    registry.register(command("session"))
    ui = FakeUI()
    context = CommandContext(agent=None, host=None, ui=ui)  # type: ignore[arg-type]

    assert await dispatch(registry, context, CommandInvocation("ses", "")) == CommandOutcome()
    assert ui.errors == ["Unknown command '/ses'. Try: /session."]

    assert await dispatch(registry, context, CommandInvocation("wat", "")) == CommandOutcome()
    assert ui.errors[-1] == "Unknown command '/wat'. Use /help."

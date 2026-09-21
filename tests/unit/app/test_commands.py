"""Tests for local command parsing, discovery, and safe dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

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
    register_builtin_commands,
)
from hammer_code.errors import HammerCodeError
from hammer_code.subagent.models import BackgroundTaskStatus
from hammer_code.subagent.service import SubagentTaskSnapshot
from hammer_code.subagent.usage import SubagentUsageSnapshot


@dataclass
class FakeUI:
    errors: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)


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


def _task_snapshot(status: BackgroundTaskStatus = BackgroundTaskStatus.RUNNING):
    usage = SubagentUsageSnapshot(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=2,
        cache_write_tokens=0,
        reasoning_tokens=1,
        reported_total=15,
        accounted_tokens=15,
        limit=200_000,
        final_requests=1,
        partial_requests=0,
        unavailable_requests=0,
        estimated=False,
        exhausted=False,
        exceeded=False,
    )
    return SubagentTaskSnapshot(
        id="12345678-1234-1234-1234-123456789abc",
        agent="review",
        task="review the change",
        status=status,
        started_at=datetime(2026, 9, 21, tzinfo=UTC),
        ended_at=None,
        result=None,
        error=None,
        usage=usage,
    )


@pytest.mark.asyncio
async def test_subagent_command_lists_gets_and_cancels_local_task_snapshots() -> None:
    class Agent:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def list_subagent_tasks(self):
            self.calls.append(("list", ""))
            return (_task_snapshot(),)

        async def get_subagent_task(self, task_id: str):
            self.calls.append(("get", task_id))
            return _task_snapshot()

        async def cancel_subagent_task(self, task_id: str):
            self.calls.append(("cancel", task_id))
            return _task_snapshot(BackgroundTaskStatus.CANCELLED)

    registry = CommandRegistry()
    register_builtin_commands(registry)
    agent = Agent()
    ui = FakeUI()
    context = CommandContext(agent=agent, host=None, ui=ui)  # type: ignore[arg-type]

    await dispatch(registry, context, CommandInvocation("subagent", ""))
    await dispatch(registry, context, CommandInvocation("subagent", "get 12345678"))
    await dispatch(registry, context, CommandInvocation("subagent", "cancel 12345678"))

    assert agent.calls == [("list", ""), ("get", "12345678"), ("cancel", "12345678")]
    assert "reported_total=15" in ui.infos[0]
    assert "accounted=15" in ui.infos[1]
    assert "status=cancelled" in ui.infos[2]


@pytest.mark.asyncio
async def test_subagent_command_rejects_extra_arguments() -> None:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    ui = FakeUI()
    context = CommandContext(agent=None, host=None, ui=ui)  # type: ignore[arg-type]

    await dispatch(registry, context, CommandInvocation("subagent", "list extra"))

    assert ui.errors == ["Usage: /subagent list | /subagent get <id> | /subagent cancel <id>"]

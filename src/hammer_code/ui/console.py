"""Small replaceable Rich console interface."""

from __future__ import annotations

import asyncio
from typing import Protocol

from rich.console import Console

from hammer_code.domain.messages import ReasoningVisibility, ToolCallBlock
from hammer_code.domain.usage import UsageSummary


class ConsolePort(Protocol):
    async def prompt(self) -> str: ...
    def text_delta(self, text: str) -> None: ...
    def reasoning_status(self) -> None: ...
    def reasoning_delta(self, text: str, visibility: ReasoningVisibility) -> None: ...
    def tool_call_notice(self, call: ToolCallBlock) -> None: ...
    def error(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def usage(self, summary: UsageSummary) -> None: ...
    def help(self) -> None: ...


class ConsoleUI:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def banner(self, profile: str, protocol: str, model: str, origin: str) -> None:
        try:
            self.console.print(
                f"[bold cyan]⚒ Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )
        except UnicodeEncodeError:
            self.console.print(
                f"[bold cyan]Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )

    async def prompt(self) -> str:
        return await asyncio.to_thread(input, "hammer> ")

    def text_delta(self, text: str) -> None:
        self.console.print(text, end="", markup=False, highlight=False)

    def reasoning_status(self) -> None:
        self.console.print("[dim]Thinking…[/]")

    def reasoning_delta(self, text: str, visibility: ReasoningVisibility) -> None:
        self.console.print(text, end="", markup=False, highlight=False)

    def tool_call_notice(self, call: ToolCallBlock) -> None:
        self.console.print(f"\n[dim]Tool call parsed: {call.name} (not executed)[/]")

    def error(self, message: str) -> None:
        self.console.print(f"[red]Error:[/] {message}")

    def info(self, message: str) -> None:
        self.console.print(message)

    def usage(self, summary: UsageSummary) -> None:
        total = summary.usage.total_tokens
        total_text = str(total) if total is not None else "unavailable"
        self.console.print(
            "\n[dim]Usage: "
            f"total={total_text}; final={summary.final_requests}, "
            f"partial={summary.partial_requests}, unavailable={summary.unavailable_requests}[/]"
        )

    def confirmation(self, prompt: str) -> bool:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}

    def help(self) -> None:
        self.console.print("Local commands: /help, /clear, /usage, /exit")

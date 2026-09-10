"""Small replaceable Rich console interface."""

from __future__ import annotations

import asyncio
from typing import Protocol

from rich.console import Console
from rich.status import Status

from hammer_code.domain.messages import ReasoningVisibility, ToolCallBlock
from hammer_code.domain.usage import UsageSummary
from hammer_code.permissions.models import ApprovalChoice, PermissionRequest


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
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice: ...


class ConsoleUI:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._thinking_active = False
        self._thinking_status: Status | None = None

    def _finish_thinking(self) -> None:
        if self._thinking_status is not None:
            self._thinking_status.stop()
            self._thinking_status = None
        self._thinking_active = False

    def banner(self, profile: str, protocol: str, model: str, origin: str) -> None:
        self._finish_thinking()
        try:
            self.console.print(
                f"[bold cyan]⚒ Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )
        except UnicodeEncodeError:
            self.console.print(
                f"[bold cyan]Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )

    async def prompt(self) -> str:
        self._finish_thinking()
        return await asyncio.to_thread(input, "hammer> ")

    def text_delta(self, text: str) -> None:
        self._finish_thinking()
        self.console.print(text, end="", markup=False, highlight=False)

    def reasoning_status(self) -> None:
        if self._thinking_active:
            return
        self._thinking_active = True
        if self.console.is_terminal:
            self._thinking_status = self.console.status(
                "[dim]Thinking…[/]", spinner="dots", spinner_style="dim"
            )
            self._thinking_status.start()
        else:
            self.console.print("[dim]Thinking…[/]")

    def reasoning_delta(self, text: str, visibility: ReasoningVisibility) -> None:
        self._finish_thinking()
        self.console.print(text, end="", markup=False, highlight=False)

    def tool_call_notice(self, call: ToolCallBlock) -> None:
        self._finish_thinking()
        self.console.print(f"\n[dim]Tool call: {call.name}[/]")

    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice:
        self._finish_thinking()
        details = request.normalized_command or str(request.normalized_arguments)
        prompt = (
            f"Approve {request.tool_name} ({reason}): {details}\n"
            "1) allow once  2) allow and persist  3) deny [3] "
        )
        while True:
            try:
                answer = (await asyncio.to_thread(input, prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                return ApprovalChoice.DENY
            match answer:
                case "1":
                    return ApprovalChoice.ALLOW_ONCE
                case "2":
                    return ApprovalChoice.ALLOW_AND_PERSIST
                case "3" | "":
                    return ApprovalChoice.DENY
            self.console.print("Enter 1, 2, or 3.")

    def error(self, message: str) -> None:
        self._finish_thinking()
        self.console.print(f"[red]Error:[/] {message}")

    def info(self, message: str) -> None:
        self._finish_thinking()
        self.console.print(message)

    def usage(self, summary: UsageSummary) -> None:
        self._finish_thinking()
        total = summary.usage.total_tokens
        total_text = str(total) if total is not None else "unavailable"
        self.console.print(
            "\n[dim]Usage: "
            f"total={total_text}; final={summary.final_requests}, "
            f"partial={summary.partial_requests}, unavailable={summary.unavailable_requests}[/]"
        )

    def confirmation(self, prompt: str) -> bool:
        self._finish_thinking()
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}

    def help(self) -> None:
        self._finish_thinking()
        self.console.print("Local commands: /help, /clear, /usage, /exit")

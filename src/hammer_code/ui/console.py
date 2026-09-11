"""Small replaceable Rich console interface."""

from __future__ import annotations

import asyncio
from typing import Protocol

from rich.cells import cell_len
from rich.console import Console
from rich.status import Status
from rich.text import Text

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
    def mcp_status(self, name: str, status: str, detail: str | None = None) -> None: ...
    def usage(self, summary: UsageSummary) -> None: ...
    def help(self) -> None: ...
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice: ...


class ConsoleUI:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._thinking_active = False
        self._thinking_status: Status | None = None
        self._stream_line_open = False
        self._stream_buffer = ""
        self._stream_style: str | None = None

    def _finish_thinking(self) -> None:
        if self._thinking_status is not None:
            self._thinking_status.stop()
            self._thinking_status = None
        self._thinking_active = False

    def _finish_stream_line(self) -> None:
        self._flush_stream_buffer()
        if self._stream_line_open:
            self.console.print()
            self._stream_line_open = False

    def _begin_block(self) -> None:
        self._finish_thinking()
        self._finish_stream_line()

    def _write_stream_chunk(self, text: str, style: str | None) -> None:
        self.console.out(text, end="", style=style, highlight=False)
        if text:
            self._stream_line_open = not text.endswith(("\n", "\r"))

    def _flush_stream_buffer(self) -> None:
        if not self._stream_buffer:
            return
        text = self._stream_buffer
        style = self._stream_style
        self._stream_buffer = ""
        self._stream_style = None
        self._write_stream_chunk(text, style)

    def _stream_text(self, text: str, *, style: str | None = None) -> None:
        self._finish_thinking()
        if not text:
            return
        if self._stream_buffer and style != self._stream_style:
            self._flush_stream_buffer()
        self._stream_style = style
        self._stream_buffer += text

        newline_at = max(self._stream_buffer.rfind("\n"), self._stream_buffer.rfind("\r"))
        if newline_at >= 0:
            completed_lines = self._stream_buffer[: newline_at + 1]
            self._stream_buffer = self._stream_buffer[newline_at + 1 :]
            self._write_stream_chunk(completed_lines, style)
            if not self._stream_buffer:
                self._stream_style = None

        target_width = max(1, self.console.width - 1)
        if self._stream_buffer and cell_len(self._stream_buffer) >= target_width:
            self._flush_stream_buffer()

    def banner(self, profile: str, protocol: str, model: str, origin: str) -> None:
        self._begin_block()
        try:
            self.console.print(
                f"[bold cyan]⚒ Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )
        except UnicodeEncodeError:
            self.console.print(
                f"[bold cyan]Hammer Code[/]  {profile} · {protocol} · {model}\n{origin}"
            )

    async def prompt(self) -> str:
        self._begin_block()
        return await asyncio.to_thread(input, "hammer> ")

    def text_delta(self, text: str) -> None:
        self._stream_text(text)

    def reasoning_status(self) -> None:
        if self._thinking_active:
            return
        self._finish_stream_line()
        self._thinking_active = True
        if self.console.is_terminal:
            self._thinking_status = self.console.status(
                "[dim]Thinking…[/]", spinner="dots", spinner_style="dim"
            )
            self._thinking_status.start()
        else:
            self.console.print("[dim]Thinking…[/]")

    def reasoning_delta(self, text: str, visibility: ReasoningVisibility) -> None:
        self._stream_text(text, style="dim")

    def tool_call_notice(self, call: ToolCallBlock) -> None:
        self._begin_block()
        self.console.print(Text.assemble(("Tool", "bold cyan"), "  ", (call.name, "dim cyan")))

    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice:
        self._begin_block()
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
        self._begin_block()
        self.console.print(f"[red]Error:[/] {message}")

    def info(self, message: str) -> None:
        self._begin_block()
        self.console.print(message)

    def mcp_status(self, name: str, status: str, detail: str | None = None) -> None:
        self._begin_block()
        message = f"MCP {name}: {status}"
        if detail:
            message += f" ({detail})"
        self.console.print(f"[dim]{message}[/]")

    def usage(self, summary: UsageSummary) -> None:
        self._begin_block()
        total = summary.usage.total_tokens
        total_text = str(total) if total is not None else "unavailable"
        self.console.print(
            "[dim]Usage: "
            f"total={total_text}; final={summary.final_requests}, "
            f"partial={summary.partial_requests}, unavailable={summary.unavailable_requests}[/]"
        )

    def confirmation(self, prompt: str) -> bool:
        self._begin_block()
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}

    def help(self) -> None:
        self._begin_block()
        self.console.print("Local commands: /help, /clear, /usage, /exit")

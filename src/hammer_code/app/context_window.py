"""Immutable assembly of one model request context, without history ownership."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from hammer_code.domain.events import ToolDefinition
from hammer_code.domain.messages import Message


@dataclass(frozen=True)
class ContextSnapshot:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]


class ContextWindow:
    def __init__(self, base_system_prompt: str) -> None:
        self._base_system_prompt = base_system_prompt

    def set_base_system_prompt(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Base system prompt must not be empty")
        self._base_system_prompt = text

    def snapshot(
        self,
        *,
        mcp_prompt: str,
        project_instructions: str = "",
        skill_prompt: str = "",
        memory_prompt: str = "",
        stale_prompt: str = "",
        recovery_prompt: str = "",
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ContextSnapshot:
        sections = tuple(
            section.strip()
            for section in (
                self._base_system_prompt,
                project_instructions,
                mcp_prompt,
                skill_prompt,
                memory_prompt,
                stale_prompt,
                recovery_prompt,
            )
            if section.strip()
        )
        prompt = "\n\n".join(sections) + ("\n" if sections else "")
        return ContextSnapshot(
            prompt,
            tuple(messages),
            tuple(
                ToolDefinition(
                    tool.name,
                    tool.description,
                    copy.deepcopy(tool.parameters),
                )
                for tool in tools
            ),
        )

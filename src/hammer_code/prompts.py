"""Protocol-neutral, deterministic system-prompt composition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class PromptStage(IntEnum):
    """Stable semantic ordering for system-prompt sections."""

    IDENTITY = 10
    SYSTEM_RULES = 20
    DOING_TASKS = 30
    EXECUTING_ACTIONS = 40
    USING_TOOLS = 50
    TONE_STYLE = 60
    TEXT_OUTPUT = 70
    RUNTIME_ENVIRONMENT = 80


@dataclass(frozen=True)
class PromptSection:
    """One independently ordered unit of a system prompt."""

    name: str
    stage: PromptStage
    priority: int
    content: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Prompt section name must not be empty")


@dataclass(frozen=True)
class PromptRuntimeContext:
    """Trusted runtime facts that may be disclosed to the model."""

    cwd: Path
    project_root: Path
    platform: str
    shell_backend: str | None
    permission_mode: str | None
    enabled_tools: tuple[str, ...]

    def __post_init__(self) -> None:
        tool_names = tuple(self.enabled_tools)
        if any(not name.strip() for name in tool_names):
            raise ValueError("Enabled tool names must not be empty")
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Enabled tool names must be unique")
        object.__setattr__(self, "enabled_tools", tuple(sorted(tool_names)))


class PromptBuilder:
    """Build a normalized prompt without depending on registration order."""

    def __init__(self, sections: Iterable[PromptSection]) -> None:
        self._sections = tuple(sections)
        names = tuple(section.name for section in self._sections)
        if len(set(names)) != len(names):
            raise ValueError("Prompt section names must be unique")

    def _ordered_sections(self) -> tuple[PromptSection, ...]:
        return tuple(
            sorted(
                (section for section in self._sections if section.content.strip()),
                key=lambda section: (int(section.stage), section.priority, section.name),
            )
        )

    def build(self) -> str:
        """Return non-empty normalized sections separated by one blank line."""
        sections = self._ordered_sections()
        if not sections:
            return ""
        return "\n\n".join(section.content.strip() for section in sections) + "\n"

    def section_names(self) -> tuple[str, ...]:
        """Return the names that participate in :meth:`build` order."""
        return tuple(section.name for section in self._ordered_sections())


def default_static_sections() -> tuple[PromptSection, ...]:
    """Return the seven implemented, capability-neutral static sections."""
    return (
        PromptSection(
            "identity",
            PromptStage.IDENTITY,
            0,
            "# Identity\n"
            "You are Hammer Code, a programming assistant for software engineering tasks.",
        ),
        PromptSection(
            "system_rules",
            PromptStage.SYSTEM_RULES,
            0,
            "# System rules\nBe accurate and honest about facts, inferences, and uncertainty. "
            "Follow the user's current authorization and never claim an operation is complete "
            "when it is not.",
        ),
        PromptSection(
            "doing_tasks",
            PromptStage.DOING_TASKS,
            0,
            "# Doing tasks\nUnderstand the goal and constraints first. Inspect relevant code, "
            "configuration, "
            "and tests; after changes, verify them and report checks that were not run.",
        ),
        PromptSection(
            "executing_actions",
            PromptStage.EXECUTING_ACTIONS,
            0,
            "# Executing actions\nKeep changes to the smallest useful scope and preserve user "
            "work. "
            "Stop when requirements conflict, scope is unauthorized, or work needs a dependency, "
            "migration, or irreversible action.",
        ),
        PromptSection(
            "using_tools",
            PromptStage.USING_TOOLS,
            0,
            "# Using tools\nUse only tools actually provided to you, and use results to correct "
            "failed calls. "
            "A prompt never bypasses authorization; do not repeat tool schemas here.",
        ),
        PromptSection(
            "tone_style",
            PromptStage.TONE_STYLE,
            0,
            "# Tone and style\n"
            "Be clear, direct, and appropriately detailed for the user's technical level.",
        ),
        PromptSection(
            "text_output",
            PromptStage.TEXT_OUTPUT,
            0,
            "# Text output\nLead with the result. When useful, name key files, validation, and "
            "remaining risks. "
            "Do not invent citations or command results.",
        ),
    )


def runtime_environment_section(context: PromptRuntimeContext) -> PromptSection:
    """Describe enabled runtime capability without exposing schemas or policy rules."""
    shell_backend = context.shell_backend or "none"
    permission_mode = context.permission_mode or "none"
    enabled_tools = ", ".join(context.enabled_tools) if context.enabled_tools else "none"
    capability_summary = (
        "No file or command tools are available. You cannot read files, execute commands, "
        "or modify files."
        if not context.enabled_tools
        else (
            "Use only the listed enabled tools through their declared interfaces; permissions are "
            "checked separately."
        )
    )
    return PromptSection(
        "runtime_environment",
        PromptStage.RUNTIME_ENVIRONMENT,
        0,
        "# Runtime environment\n"
        f"Platform: {context.platform}\n"
        f"Project root: {context.project_root.resolve()}\n"
        f"Current working directory: {context.cwd.resolve()}\n"
        f"Shell backend: {shell_backend}\n"
        f"Permission mode: {permission_mode}\n"
        f"Enabled tools: {enabled_tools}\n"
        f"{capability_summary}",
    )


def build_system_prompt(context: PromptRuntimeContext) -> str:
    """Build the complete system prompt from static and trusted runtime sections."""
    return PromptBuilder((*default_static_sections(), runtime_environment_section(context))).build()

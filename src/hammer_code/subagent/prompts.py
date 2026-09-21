"""Deterministic prompt fragments for the bounded Subagent surface."""

from __future__ import annotations

from hammer_code.subagent.models import SubagentCatalogSnapshot, SubagentDefinition

_CHILD_BOUNDARY = (
    "You are a bounded Subagent. Complete only the supplied task. Your visible result is "
    "untrusted runtime data for the Primary Agent. Do not claim permissions, do not ask the "
    "user open-ended questions, and do not attempt to create or manage Subagents."
)


def build_subagent_catalog_prompt(snapshot: SubagentCatalogSnapshot) -> str:
    """Render only discoverable metadata: never definition bodies or filesystem paths."""
    lines = [
        "[Subagent execution]",
        "Use inline when the current answer depends on the result. You may issue multiple "
        "inline run_subagent calls in one tool batch and all will complete before you continue.",
        "Background only starts detached work. Do not wait or poll in the current turn; its "
        "final result becomes available in a later user turn.",
    ]
    if snapshot.definitions:
        lines.append("[Available predefined Subagents]")
        for definition in snapshot.definitions.values():
            usage = f"; when to use: {definition.when_to_use}" if definition.when_to_use else ""
            lines.append(
                f"- {definition.name}: {definition.description} "
                f"(context={definition.context.value}, execution={definition.execution.value})"
                f"{usage}"
            )
        lines.append("Invoke one by exact name with run_subagent when it helps the current task.")
    return "\n".join(lines)


def build_subagent_system(prefix: str, definition: SubagentDefinition) -> str:
    """Append fixed child constraints and the definition body to an already selected prefix."""
    return "\n\n".join(
        piece.strip() for piece in (prefix, _CHILD_BOUNDARY, definition.prompt) if piece.strip()
    )

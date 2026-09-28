"""Pure durable projection for transient successful ``use_skill`` bodies."""

from __future__ import annotations

from collections.abc import Mapping

from hammer_code.domain.messages import Message, TextBlock, ToolResultBlock
from hammer_code.skill.models import SkillLoadOutcome


def project_durable_skill_messages(
    live_messages: tuple[Message, ...],
    outcomes: Mapping[str, SkillLoadOutcome],
    *,
    original_user_message: Message | None = None,
) -> tuple[Message, ...]:
    projected: list[Message] = []
    for position, message in enumerate(live_messages):
        if position == 0 and original_user_message is not None:
            projected.append(original_user_message)
            continue
        blocks = []
        for block in message.content:
            outcome = outcomes.get(block.call_id) if isinstance(block, ToolResultBlock) else None
            if (
                isinstance(block, ToolResultBlock)
                and outcome is not None
                and not block.is_error
                and not outcome.is_error
            ):
                ref = outcome.ref
                text = (
                    f"Skill loaded for this turn: {ref.scope.value}:{ref.name}@{ref.version}. "
                    "Full instructions are not retained; call use_skill again in a "
                    "later top-level turn."
                )
                blocks.append(ToolResultBlock(block.call_id, (TextBlock(text),), False))
            else:
                blocks.append(block)
        projected.append(Message(message.role, tuple(blocks)))
    return tuple(projected)

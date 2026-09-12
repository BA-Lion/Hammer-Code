"""Deterministic local token estimates without a model-specific tokenizer."""

from __future__ import annotations

import json

from hammer_code.domain.events import ToolDefinition
from hammer_code.domain.messages import (
    Message,
    ProviderStateBlock,
    ReasoningBlock,
    RefusalBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


class TokenEstimator:
    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    def estimate_messages(self, messages: tuple[Message, ...]) -> int:
        total = 0
        for message in messages:
            total += 4
            for block in message.content:
                total += 4
                if isinstance(block, (TextBlock, ReasoningBlock)):
                    total += self.estimate_text(block.text)
                elif isinstance(block, RefusalBlock):
                    total += self.estimate_text(block.reason)
                elif isinstance(block, ToolCallBlock):
                    total += self.estimate_text(block.call_id)
                    total += self.estimate_text(block.name)
                    total += self.estimate_text(block.raw_arguments)
                elif isinstance(block, ToolResultBlock):
                    total += self.estimate_text(block.call_id) + 1
                    total += sum(self.estimate_text(item.text) for item in block.content)
                elif isinstance(block, ProviderStateBlock):
                    total += self.estimate_text(block.protocol) + self.estimate_text(block.kind)
                    total += self.estimate_text(
                        json.dumps(
                            block.data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        )
                    )
        return total

    def estimate_tools(self, tools: tuple[ToolDefinition, ...]) -> int:
        return sum(
            8
            + self.estimate_text(tool.name)
            + self.estimate_text(tool.description)
            + self.estimate_text(
                json.dumps(
                    tool.parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            for tool in tools
        )

    def estimate_request(
        self, system_prompt: str, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> int:
        return (
            self.estimate_text(system_prompt)
            + self.estimate_messages(messages)
            + self.estimate_tools(tools)
        )

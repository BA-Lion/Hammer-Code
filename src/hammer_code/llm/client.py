"""The only model-client interface visible to the application layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from hammer_code.domain.events import ModelEvent, ModelRequest


@dataclass(frozen=True)
class ClientCapabilities:
    protocol: str
    streaming: bool
    tool_call_parsing: bool
    reasoning_events: bool
    usage_reporting: bool


class ModelClient(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ClientCapabilities: ...

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    @abstractmethod
    async def aclose(self) -> None: ...

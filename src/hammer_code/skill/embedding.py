"""Provider-independent, metadata-only embedding contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from hammer_code.skill.models import SkillDefinition, SkillScope

Identity = tuple[SkillScope, str]
Vector = tuple[float, ...]


class EmbeddingError(ValueError):
    """A bounded failure category, never a raw provider payload."""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    when_to_use: str | None = None

    @classmethod
    def from_definition(cls, definition: SkillDefinition) -> SkillMetadata:
        return cls(definition.name, definition.description, definition.when_to_use)

    def text(self) -> str:
        value = f"name: {self.name}\ndescription: {self.description}"
        return value + (f"\nwhen-to-use: {self.when_to_use}" if self.when_to_use else "")


@dataclass(frozen=True)
class EmbeddingQuery:
    text: str
    metadata: SkillMetadata | None = None

    @classmethod
    def from_metadata(cls, metadata: SkillMetadata) -> EmbeddingQuery:
        return cls(metadata.text(), metadata)


def validate_vector(vector: Vector, dimensions: int = 384) -> None:
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("invalid_vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0 or abs(norm - 1.0) > 1e-3:
        raise EmbeddingError("invalid_vector_norm")


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed_query(self, query: EmbeddingQuery) -> Vector: ...

    async def embed_documents(self, documents: tuple[SkillMetadata, ...]) -> tuple[Vector, ...]: ...

    async def aclose(self) -> None: ...

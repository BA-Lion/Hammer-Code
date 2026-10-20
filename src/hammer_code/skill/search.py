"""Shared two-route retrieval; business callers choose query, scope and top-k."""

from __future__ import annotations

import math
from typing import Literal, Protocol

from hammer_code.skill.embedding import EmbeddingQuery, Identity
from hammer_code.skill.models import CatalogSnapshot, RetrievedSkill
from hammer_code.skill.retrieval import Bm25Index


class VectorSearch(Protocol):
    async def rank(
        self, query: EmbeddingQuery, identities: frozenset[Identity], limit: int
    ) -> tuple[tuple[Identity, float], ...]: ...


class SkillSearchService:
    def __init__(self, vector: VectorSearch | None = None) -> None:
        self.vector = vector

    async def search(
        self,
        query: EmbeddingQuery,
        top_k: int,
        *,
        snapshot: CatalogSnapshot,
        purpose: Literal["recommend", "maintenance"] = "recommend",
    ) -> tuple[RetrievedSkill, ...]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if purpose not in {"recommend", "maintenance"}:
            raise ValueError("unknown search purpose")
        definitions = (
            tuple(snapshot.by_identity.values())
            if purpose == "maintenance"
            else tuple(item for item in snapshot.effective.values() if item.model_invocable)
        )
        allowed = {(item.scope, item.name): item for item in definitions}
        index = snapshot.index or Bm25Index(definitions)
        lexical = index.rank(query.text, definitions, 2 * top_k)
        vector = ()
        if self.vector is not None:
            try:
                vector = await self.vector.rank(query, frozenset(allowed), 2 * top_k)
            except Exception:
                vector = ()
        dense: dict[Identity, float] = {}
        for identity, score in vector:
            if identity in allowed and identity not in dense and math.isfinite(score):
                dense[identity] = score
        dense_keys = sorted(dense, key=lambda key: (-dense[key], key))[: 2 * top_k]
        if not dense_keys:
            return lexical[:top_k]
        lexical_ranks = {
            (item.ref.scope, item.ref.name): rank for rank, item in enumerate(lexical, 1)
        }
        dense_ranks = {identity: rank for rank, identity in enumerate(dense_keys, 1)}
        scores: dict[Identity, float] = {}
        for ranks in (lexical_ranks, dense_ranks):
            for identity, rank in ranks.items():
                scores[identity] = scores.get(identity, 0.0) + 1 / (60 + rank)
        keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
        coverage = {(item.ref.scope, item.ref.name): item.coverage for item in lexical}
        highest = max(scores.values())
        return tuple(
            RetrievedSkill(
                allowed[key].ref,
                scores[key],
                scores[key] / highest,
                coverage.get(key, 0.0),
                "rrf",
                lexical_ranks.get(key),
                dense_ranks.get(key),
            )
            for key in keys
        )

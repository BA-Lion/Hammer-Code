"""Deterministic metadata-only BM25F retrieval for Skills."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from hammer_code.skill.models import RetrievedSkill, SkillDefinition

_ASCII_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_-]+|[^\w]+")
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_WEIGHTS = {"name": 4.0, "when": 3.0, "description": 2.0}
_K1 = 1.2
_B = 0.75


def tokenize(value: str) -> tuple[str, ...]:
    """Split identifiers while preserving CJK uni- and bi-gram recall."""
    normalized = unicodedata.normalize("NFKC", value)
    tokens: list[str] = []
    for piece in _ASCII_SPLIT.split(normalized):
        if not piece:
            continue
        cjk_runs = _CJK.findall(piece)
        non_cjk = _CJK.sub(" ", piece).casefold()
        tokens.extend(item for item in non_cjk.split() if item)
        for run in cjk_runs:
            tokens.append(run)
        # Consecutive CJK characters must also produce bigrams. Re-scan the
        # normalized source so runs are not broken by ASCII identifiers.
    for run in re.findall(r"[\u3400-\u9fff\uf900-\ufaff]+", normalized):
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(tokens)


@dataclass(frozen=True)
class _Document:
    definition: SkillDefinition
    fields: dict[str, Counter[str]]
    lengths: dict[str, int]


class Bm25Index:
    """A compact, immutable index containing no Skill prompt body."""

    def __init__(self, definitions: Iterable[SkillDefinition]) -> None:
        self._documents = tuple(self._document(item) for item in definitions)
        self._averages = {
            field: (
                sum(document.lengths[field] for document in self._documents) / len(self._documents)
                if self._documents
                else 0.0
            )
            for field in _WEIGHTS
        }
        self._df: dict[str, int] = defaultdict(int)
        for document in self._documents:
            for token in set().union(*(set(values) for values in document.fields.values())):
                self._df[token] += 1

    @staticmethod
    def _document(definition: SkillDefinition) -> _Document:
        fields = {
            "name": Counter(tokenize(definition.name)),
            "when": Counter(tokenize(definition.when_to_use or "")),
            "description": Counter(tokenize(definition.description)),
        }
        return _Document(
            definition, fields, {field: sum(value.values()) for field, value in fields.items()}
        )

    def _idf(self, token: str) -> float:
        count = self._df.get(token, 0)
        size = len(self._documents)
        return math.log(1 + (size - count + 0.5) / (count + 0.5)) if count else 0.0

    def _field_tf(self, token: str, document: _Document) -> float:
        total = 0.0
        for field, weight in _WEIGHTS.items():
            average = self._averages[field]
            length = document.lengths[field]
            denominator = 1 - _B + _B * length / average if average else 1.0
            total += weight * document.fields[field].get(token, 0) / denominator
        return total

    def score(self, query: str, definition: SkillDefinition) -> tuple[float, float]:
        document = next(
            (item for item in self._documents if item.definition == definition),
            self._document(definition),
        )
        query_tokens = tuple(dict.fromkeys(tokenize(query)))
        total_idf = sum(self._idf(token) for token in query_tokens)
        if not total_idf:
            return 0.0, 0.0
        score = 0.0
        matched_idf = 0.0
        for token in query_tokens:
            field_tf = self._field_tf(token, document)
            if field_tf:
                idf = self._idf(token)
                matched_idf += idf
                score += idf * (_K1 + 1) * field_tf / (_K1 + field_tf)
        return score, matched_idf / total_idf

    def retrieve(
        self,
        query: str,
        definitions: Iterable[SkillDefinition],
        *,
        top_k: int,
        relative_floor: float,
        coverage_floor: float,
    ) -> tuple[RetrievedSkill, ...]:
        scored = [
            (definition, *self.score(query, definition))
            for definition in definitions
            if definition.model_invocable
        ]
        scored = [item for item in scored if item[1] > 0]
        if not scored:
            return ()
        scored.sort(key=lambda item: (-item[1], item[0].scope.value != "project", item[0].name))
        top = scored[0][1]
        exact = query.strip().casefold()
        selected = [
            item
            for item in scored
            if item[0].name == exact
            or (item[1] >= top * relative_floor and item[2] >= coverage_floor)
        ][:top_k]
        return tuple(
            RetrievedSkill(item[0].ref, item[1], min(1.0, item[1] / top), item[2])
            for item in selected
        )

    def merge_candidates(
        self, query: str, definitions: Iterable[SkillDefinition], top_k: int
    ) -> tuple[RetrievedSkill, ...]:
        scored = [(definition, *self.score(query, definition)) for definition in definitions]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0].scope.value != "project", item[0].name))
        top = scored[0][1] if scored else 0.0
        return tuple(
            RetrievedSkill(item[0].ref, item[1], min(1.0, item[1] / top), item[2])
            for item in scored[:top_k]
        )

    def normalized_merge_candidates(
        self,
        query: str,
        candidate: SkillDefinition,
        definitions: Iterable[SkillDefinition],
        top_k: int,
    ) -> tuple[RetrievedSkill, ...]:
        """Rank merge targets against the candidate's own BM25F score.

        Relative-to-top ranking is useful for presenting alternatives; forced
        merge needs the stricter candidate/self denominator so a weak query
        cannot make an unrelated first result look equivalent.
        """
        self_score, _ = self.score(query, candidate)
        if self_score <= 0:
            return ()
        scored = [(definition, *self.score(query, definition)) for definition in definitions]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0].scope.value != "project", item[0].name))
        return tuple(
            RetrievedSkill(item[0].ref, item[1], min(1.0, item[1] / self_score), item[2])
            for item in scored[:top_k]
        )

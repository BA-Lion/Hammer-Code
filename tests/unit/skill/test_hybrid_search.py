from __future__ import annotations

import math
from pathlib import Path

import pytest

from hammer_code.skill.catalog import SkillCatalog
from hammer_code.skill.embedding import EmbeddingQuery
from hammer_code.skill.models import SkillScope
from hammer_code.skill.search import SkillSearchService


class Dense:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def rank(self, query, identities, limit):
        self.calls.append((query, identities, limit))
        return tuple(self.scores[:limit])


@pytest.mark.parametrize("k", [3, 10])
async def test_window_rrf_and_output_limits(tmp_path: Path, skill_factory, k: int) -> None:
    definitions = [skill_factory(f"skill-{i:02}") for i in range(45)]
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    dense = Dense(
        [((item.scope, item.name), 1.0 - i / 100) for i, item in enumerate(reversed(definitions))]
    )
    result = await SkillSearchService(dense).search(
        EmbeddingQuery("workflow"), k, snapshot=snapshot
    )
    assert len(result) == k
    assert dense.calls[0][2] == 2 * k
    assert all(item.score_kind == "rrf" for item in result)
    assert all(item.bm25_rank is None or item.bm25_rank <= 2 * k for item in result)
    assert all(item.vector_rank is None or item.vector_rank <= 2 * k for item in result)
    assert len({item.ref for item in result}) == k


async def test_rrf_hand_calculation_and_stable_tie(tmp_path: Path, skill_factory) -> None:
    a, b = skill_factory("alpha"), skill_factory("beta")
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    dense = Dense([((b.scope, b.name), 0.9), ((a.scope, a.name), 0.8)])
    result = await SkillSearchService(dense).search(
        EmbeddingQuery("workflow"), 3, snapshot=snapshot
    )
    assert [item.ref.name for item in result] == ["alpha", "beta"]
    assert result[0].raw_score == pytest.approx(1 / 61 + 1 / 62)
    assert result[0].raw_score == result[1].raw_score


async def test_scope_filtering_precedes_windows(tmp_path: Path, skill_factory) -> None:
    skill_factory("same", scope=SkillScope.USER)
    project = skill_factory("same")
    skill_factory("hidden", invocable=False)
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    dense = Dense([((project.scope, project.name), 0.01)])
    service = SkillSearchService(dense)
    result = await service.search(EmbeddingQuery("unrelated"), 3, snapshot=snapshot)
    assert [item.ref for item in result] == [project.ref]
    assert dense.calls[-1][1] == frozenset({(SkillScope.PROJECT, "same")})
    await service.search(EmbeddingQuery("workflow"), 10, snapshot=snapshot, purpose="maintenance")
    assert len(dense.calls[-1][1]) == 3


async def test_vector_only_duplicates_invalid_and_small_catalog(
    tmp_path: Path, skill_factory
) -> None:
    a = skill_factory("alpha")
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    key = (a.scope, a.name)
    dense = Dense([(key, 0.01), (key, 0.01), ((a.scope, "gone"), 1.0), (key, math.nan)])
    result = await SkillSearchService(dense).search(
        EmbeddingQuery("unrelated"), 3, snapshot=snapshot
    )
    assert len(result) == 1 and result[0].ref == a.ref
    assert result[0].raw_score == pytest.approx(1 / 61)


async def test_bm25_fallback_has_no_legacy_coverage_gate(tmp_path: Path, skill_factory) -> None:
    skill_factory("alpha")
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    result = await SkillSearchService().search(
        EmbeddingQuery("workflow " + " ".join(f"unseen{i}" for i in range(40))),
        3,
        snapshot=snapshot,
    )
    assert len(result) == 1 and result[0].coverage < 0.2
    assert result[0].score_kind == "bm25"
    assert (
        await SkillSearchService().search(EmbeddingQuery("unrelated"), 3, snapshot=snapshot) == ()
    )


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
async def test_invalid_top_k(tmp_path: Path, k) -> None:
    with pytest.raises(ValueError):
        await SkillSearchService().search(
            EmbeddingQuery("x"), k, snapshot=SkillCatalog(tmp_path).build_snapshot()
        )


async def test_provider_failure_falls_back(tmp_path: Path, skill_factory) -> None:
    skill_factory("alpha")

    class Broken:
        async def rank(self, query, identities, limit):
            raise RuntimeError("sensitive provider text")

    result = await SkillSearchService(Broken()).search(
        EmbeddingQuery("workflow"), 3, snapshot=SkillCatalog(tmp_path).build_snapshot()
    )
    assert len(result) == 1 and result[0].score_kind == "bm25"

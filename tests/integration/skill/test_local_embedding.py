from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hammer_code.config import SkillVectorConfig
from hammer_code.skill.embedding import EmbeddingQuery, SkillMetadata
from hammer_code.skill.local_embedding import LocalEmbeddingProvider
from hammer_code.skill.model_assets import prepare_model, verify_assets
from hammer_code.skill.models import SkillDefinition, SkillScope, SkillSource
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.store import PreparedSkillWrite
from hammer_code.skill.vector_runtime import SkillVectorRuntime

pytestmark = pytest.mark.skipif(
    not os.environ.get("HAMMER_CODE_TEST_MODEL_ROOT"),
    reason="explicit temporary model assets required",
)


async def test_real_cpu_pooling_against_raw_onnx_and_offline_reopen(monkeypatch) -> None:
    import httpx2
    import numpy as np

    root = Path(os.environ["HAMMER_CODE_TEST_MODEL_ROOT"])
    directory = verify_assets(root)

    class Offline(httpx2.AsyncClient):
        def stream(self, *args, **kwargs):
            pytest.fail("prepared assets must not download again")

    monkeypatch.setattr(httpx2, "AsyncClient", Offline)
    assert await prepare_model(root) == directory
    query = "Check the Python test coverage"
    for _ in range(2):
        with ThreadPoolExecutor(1) as executor:
            provider = LocalEmbeddingProvider(root, executor)
            await provider.initialize()
            try:
                actual = await provider.embed_query(EmbeddingQuery(query))
                tokenizer = provider._tokenizer
                tokenizer.no_truncation()
                encoded = tokenizer.encode("query: " + query)
                source = {
                    "input_ids": encoded.ids,
                    "attention_mask": encoded.attention_mask,
                    "token_type_ids": encoded.type_ids,
                }
                inputs = {
                    item.name: np.asarray([source[item.name]], dtype=np.int64)
                    for item in provider._session.get_inputs()
                }
                hidden = provider._session.run(["last_hidden_state"], inputs)[0]
                mask = np.asarray(encoded.attention_mask, dtype=np.float32)
                expected = (hidden[0] * mask[:, None]).sum(axis=0) / mask.sum()
                expected /= np.linalg.norm(expected)
                np.testing.assert_allclose(actual, expected, atol=2e-6)
                assert len(actual) == 384 and abs(np.linalg.norm(actual) - 1) < 1e-5
                metadata = SkillMetadata("review", "检查测试覆盖率 " * 400, "When reviewing " * 400)
                batch = await provider.embed_documents(
                    (metadata, SkillMetadata("other", "Second document"))
                )
                assert len(batch) == 2 and all(len(vector) == 384 for vector in batch)
                assert len(provider._tokens("query: ", metadata.text(), metadata).ids) <= 512
            finally:
                await provider.aclose()


async def test_real_runtime_and_persistent_vectors(tmp_path: Path) -> None:
    repo = SkillRepository(tmp_path)
    await repo.initialize()
    now = datetime.now(UTC)
    for index in range(4):
        name = f"test-{index}"
        definition = SkillDefinition(
            name,
            f"Review Python module {index}",
            "0.1.0",
            now,
            now,
            SkillSource.MANUAL,
            "Complete current body",
            SkillScope.PROJECT,
            tmp_path / ".hammer-code/skill/project" / name,
        )
        await repo.commit(PreparedSkillWrite("add", definition, None, "fixture", f"seed-{index}"))
    for _ in range(2):
        runtime = SkillVectorRuntime(
            repo, SkillVectorConfig(query_timeout_seconds=10), lambda _: None
        )
        provider = LocalEmbeddingProvider(
            Path(os.environ["HAMMER_CODE_TEST_MODEL_ROOT"]), runtime.executor
        )
        await provider.initialize()
        runtime.provider = provider
        runtime.start()
        try:
            await asyncio.wait_for(runtime.wait_idle(), 30)
            snapshot = await repo.snapshot_for_turn()
            result = await repo.search.search(
                EmbeddingQuery("分析项目的覆盖率"), 3, snapshot=snapshot
            )
            assert len(result) == 3
            assert all(item.score_kind == "rrf" for item in result)
            assert all(entry.retrieve == 0 for entry in repo.store.load_usage().values())
        finally:
            await runtime.aclose()

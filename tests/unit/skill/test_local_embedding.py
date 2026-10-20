from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hammer_code.skill.embedding import EmbeddingError, EmbeddingQuery, SkillMetadata
from hammer_code.skill.local_embedding import LocalEmbeddingProvider


class Tokens:
    def __init__(self):
        self.texts = []

    def no_truncation(self):
        pass

    def enable_truncation(self, **kwargs):
        assert kwargs == {"max_length": 512, "direction": "right"}

    def encode(self, text, **kwargs):
        self.texts.append(text)
        return SimpleNamespace(ids=[1, 2], attention_mask=[1, 0], type_ids=[0, 0])


class Session:
    def get_inputs(self):
        return [
            SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")
        ]

    def run(self, outputs, inputs):
        import numpy as np

        assert outputs == ["last_hidden_state"]
        assert all(value.dtype == np.int64 for value in inputs.values())
        hidden = np.zeros((1, 2, 384), dtype=np.float32)
        hidden[0, 0, 0] = 2
        hidden[0, 1, 1] = 100
        return [hidden]


async def test_padding_mask_pooling_normalization_prefix_and_batch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    with ThreadPoolExecutor(1) as executor:
        provider = LocalEmbeddingProvider(tmp_path, executor)
        provider._numpy = np
        provider._session = Session()
        tokenizer = Tokens()
        provider._tokenizer = tokenizer
        try:
            vector = await provider.embed_query(EmbeddingQuery("review tests"))
            assert vector == (1.0,) + (0.0,) * 383
            assert tokenizer.texts == ["query: review tests"]
        finally:
            await provider.aclose()
        assert provider._session is None


def test_metadata_budget_includes_special_tokens_and_preserves_name(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "name": 3,
                "description": 4,
                "passage": 5,
                "query": 6,
                ":": 7,
                "review": 8,
                "word": 9,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    with ThreadPoolExecutor(1) as executor:
        provider = LocalEmbeddingProvider(tmp_path, executor)
        provider._tokenizer = tokenizer
        metadata = SkillMetadata("review", "word " * 900, "word " * 900)
        query = provider._tokens("query: ", metadata.text(), metadata)
        passage = provider._tokens("passage: ", metadata.text(), metadata)
        assert len(query.ids) == len(passage.ids) == 512
        assert query.ids[0] == 1 and query.ids[-1] == 2
        assert "review" in query.tokens
        assert "description" in query.tokens
        assert query.tokens.count("query") == 1


async def test_invalid_output_is_rejected(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    class Broken(Session):
        def run(self, outputs, inputs):
            return [np.zeros((1, 384), dtype=np.float32)]

    with ThreadPoolExecutor(1) as executor:
        provider = LocalEmbeddingProvider(tmp_path, executor)
        provider._numpy, provider._tokenizer, provider._session = np, Tokens(), Broken()
        try:
            with pytest.raises(EmbeddingError):
                await provider.embed_query(EmbeddingQuery("x"))
        finally:
            await provider.aclose()

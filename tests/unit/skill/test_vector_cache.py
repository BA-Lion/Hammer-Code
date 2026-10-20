from __future__ import annotations

import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hammer_code.skill.embedding import EmbeddingError
from hammer_code.skill.models import SkillScope
from hammer_code.skill.vector_cache import VectorCache


def basis(index: int = 0):
    return tuple(float(i == index) for i in range(384))


def test_roundtrip_scope_matrix_and_reopen(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    cache = VectorCache(tmp_path)
    cache.open()
    try:
        a, b = (SkillScope.PROJECT, "same"), (SkillScope.USER, "same")
        cache.upsert(a, basis())
        cache.upsert(b, basis(1))
        assert cache.connection is not None
        assert [
            row[1] for row in cache.connection.execute("PRAGMA table_info(skill_embeddings)")
        ] == ["scope", "name", "embedding", "updated_at"]
        assert (
            cache.connection.execute("SELECT length(embedding) FROM skill_embeddings").fetchone()[0]
            == 1536
        )
        assert cache.rank(basis(), frozenset({a, b}), 1) == ((a, 1.0),)
        assert cache.rank(basis(), frozenset({b}), 3) == ((b, 0.0),)
        cache.close()
        cache.open()
        assert cache.vectors[a] == basis()
    finally:
        cache.close()


def test_failed_transaction_preserves_memory_blob_and_timestamp(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path)
    cache.open()
    try:
        key = (SkillScope.PROJECT, "a")
        cache.upsert(key, basis())
        connection = cache.connection
        assert connection is not None
        previous = connection.execute(
            "SELECT embedding, updated_at FROM skill_embeddings"
        ).fetchone()
        connection.execute(
            "CREATE TRIGGER reject_update BEFORE UPDATE ON skill_embeddings "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
        with pytest.raises(EmbeddingError):
            cache.upsert(key, basis(1))
        assert (
            connection.execute("SELECT embedding, updated_at FROM skill_embeddings").fetchone()
            == previous
        )
        assert cache.vectors[key] == basis()
        assert not connection.in_transaction
    finally:
        cache.close()


@pytest.mark.parametrize("blob", [b"x" * 1535, "x" * 1536])
def test_incomplete_and_text_blobs_rejected(tmp_path: Path, blob) -> None:
    cache = VectorCache(tmp_path)
    cache.open()
    try:
        assert cache.connection is not None
        with pytest.raises(sqlite3.IntegrityError):
            cache.connection.execute(
                "INSERT INTO skill_embeddings(scope,name,embedding) VALUES('project','a',?)",
                (blob,),
            )
    finally:
        cache.close()


def test_invalid_values_skip_and_thread_affinity(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path)
    cache.open()
    try:
        connection = cache.connection
        assert connection is not None
        connection.execute(
            "INSERT INTO skill_embeddings(scope,name,embedding) VALUES('project','bad',?)",
            (struct.pack("<384f", *([float("nan")] * 384)),),
        )
        with ThreadPoolExecutor(1) as executor:
            with pytest.raises(sqlite3.ProgrammingError):
                executor.submit(connection.execute, "SELECT 1").result()
        cache.close()
        cache.open()
        assert not cache.vectors
    finally:
        cache.close()


def test_corrupt_cache_rebuild_isolated(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path)
    cache.open()
    cache.close()
    from hammer_code.skill.model_assets import safe_directory

    directory = safe_directory(tmp_path, ".hammer-code/skill/state")
    (directory / "embeddings.sqlite3").write_bytes(b"not sqlite")
    with pytest.raises(EmbeddingError):
        cache.open()
    cache.rebuild()
    try:
        assert list(directory.glob("*.corrupt-*"))
        cache.upsert((SkillScope.PROJECT, "ok"), basis())
    finally:
        cache.close()

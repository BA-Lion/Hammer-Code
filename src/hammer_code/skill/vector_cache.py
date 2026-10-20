"""Derived vectors: one complete atomic SQLite row per Skill identity."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from uuid import uuid4

from hammer_code.skill.embedding import EmbeddingError, Identity, Vector, validate_vector
from hammer_code.skill.model_assets import is_reparse, safe_directory
from hammer_code.skill.models import SkillScope

DDL = """CREATE TABLE IF NOT EXISTS skill_embeddings (
    scope TEXT NOT NULL CHECK (scope IN ('project', 'user')),
    name TEXT NOT NULL,
    embedding BLOB NOT NULL CHECK (
        typeof(embedding) = 'blob' AND length(embedding) = 1536
    ),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (scope, name)
)"""
UPSERT = """INSERT INTO skill_embeddings (scope, name, embedding) VALUES (?, ?, ?)
ON CONFLICT (scope, name) DO UPDATE SET
embedding = excluded.embedding, updated_at = excluded.updated_at"""


class VectorCache:
    """Construct/open/use/close exclusively on the runtime's dedicated thread."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.connection: sqlite3.Connection | None = None
        self.vectors: dict[Identity, Vector] = {}

    def open(self) -> None:
        directory = safe_directory(self.root, ".hammer-code/skill/state", create=True)
        path = directory / "embeddings.sqlite3"
        if is_reparse(path):
            raise EmbeddingError("unsafe_cache_path")
        try:
            self.connection = sqlite3.connect(path, timeout=1, isolation_level=None)
            if self.connection.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise EmbeddingError("unsupported_cache_schema")
            self.connection.execute(DDL)
            self.connection.execute("PRAGMA user_version = 1")
            self.vectors.clear()
            for scope, name, blob in self.connection.execute(
                "SELECT scope,name,embedding FROM skill_embeddings"
            ):
                try:
                    if not isinstance(blob, bytes):
                        continue
                    vector = struct.unpack("<384f", blob)
                    validate_vector(vector)
                    self.vectors[(SkillScope(scope), name)] = vector
                except (ValueError, struct.error):
                    continue
        except (sqlite3.Error, OSError, EmbeddingError) as exc:
            self.close()
            raise EmbeddingError("cache_unavailable") from exc

    def upsert(self, identity: Identity, vector: Vector) -> None:
        validate_vector(vector)
        blob = struct.pack("<384f", *vector)
        connection = self.connection
        if connection is None:
            raise EmbeddingError("cache_not_open")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(UPSERT, (identity[0].value, identity[1], blob))
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise EmbeddingError("cache_write_failed") from exc
        self.vectors[identity] = struct.unpack("<384f", blob)

    def rank(
        self, query: Vector, identities: frozenset[Identity], limit: int
    ) -> tuple[tuple[Identity, float], ...]:
        import numpy as np

        validate_vector(query)
        keys = sorted(identities.intersection(self.vectors))
        if not keys:
            return ()
        matrix = np.asarray([self.vectors[key] for key in keys], dtype=np.float32)
        scores = matrix @ np.asarray(query, dtype=np.float32)
        result = [
            (key, float(score))
            for key, score in zip(keys, scores, strict=True)
            if np.isfinite(score)
        ]
        result.sort(key=lambda item: (-item[1], item[0]))
        return tuple(result[:limit])

    def rebuild(self) -> None:
        self.close()
        directory = safe_directory(self.root, ".hammer-code/skill/state", create=True)
        path = directory / "embeddings.sqlite3"
        if is_reparse(path):
            raise EmbeddingError("unsafe_cache_path")
        if path.exists():
            path.rename(directory / f"embeddings.sqlite3.corrupt-{uuid4().hex}")
        self.open()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.vectors.clear()

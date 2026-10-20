"""Local E5 implementation; optional imports occur only during initialization."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

from hammer_code.skill.embedding import (
    EmbeddingError,
    EmbeddingQuery,
    SkillMetadata,
    Vector,
    validate_vector,
)
from hammer_code.skill.model_assets import manifest, verify_assets


class LocalEmbeddingProvider:
    dimensions = 384

    def __init__(self, root: Path, executor: Executor) -> None:
        self.root = root
        self.executor = executor
        self._session: Any = None
        self._tokenizer: Any = None
        self._numpy: Any = None

    async def initialize(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.executor, self._initialize)

    def _initialize(self) -> None:
        try:
            import numpy
            import onnxruntime
            from tokenizers import Tokenizer

            directory = verify_assets(self.root)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            session = onnxruntime.InferenceSession(
                str(directory / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            names = {item.name for item in session.get_inputs()}
            if not {"input_ids", "attention_mask"} <= names or names - {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            }:
                raise EmbeddingError("incompatible_model_inputs")
            if manifest()["output"] not in {item.name for item in session.get_outputs()}:
                raise EmbeddingError("incompatible_model_output")
            self._numpy = numpy
            self._session = session
            self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            self._tokenizer.no_padding()
        except EmbeddingError:
            raise
        except ImportError as exc:
            raise EmbeddingError("missing_dependencies") from exc
        except Exception as exc:
            raise EmbeddingError("local_provider_unavailable") from exc

    def _tokens(self, prefix: str, text: str, metadata: SkillMetadata | None) -> Any:
        tokenizer = self._tokenizer
        tokenizer.no_truncation()
        if metadata is not None:
            # Budget individual fields; never create multiple query chunks.
            text = f"name: {metadata.name}"
            for label, value in (
                ("description", metadata.description),
                ("when-to-use", metadata.when_to_use),
            ):
                if not value:
                    continue
                ids = tokenizer.encode(value, add_special_tokens=False).ids
                low, high = 0, len(ids)
                while low < high:
                    mid = (low + high + 1) // 2
                    proposed = (
                        text
                        + f"\n{label}: "
                        + tokenizer.decode(ids[:mid], skip_special_tokens=False)
                    )
                    if len(tokenizer.encode(prefix + proposed).ids) <= 512:
                        low = mid
                    else:
                        high = mid - 1
                if low:
                    text += f"\n{label}: " + tokenizer.decode(ids[:low], skip_special_tokens=False)
        tokenizer.enable_truncation(max_length=512, direction="right")
        return tokenizer.encode(prefix + text, add_special_tokens=True)

    def _encode(self, prefix: str, text: str, metadata: SkillMetadata | None) -> Vector:
        if self._session is None:
            raise EmbeddingError("provider_not_ready")
        np = self._numpy
        encoding = self._tokens(prefix, text, metadata)
        values = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }
        inputs = {
            item.name: np.asarray([values[item.name]], dtype=np.int64)
            for item in self._session.get_inputs()
        }
        hidden = self._session.run([manifest()["output"]], inputs)[0]
        if hidden.ndim != 3 or hidden.shape != (1, len(encoding.ids), 384):
            raise EmbeddingError("invalid_model_output")
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        vector = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
        vector = vector[0].astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise EmbeddingError("invalid_vector_norm")
        result = tuple(float(value) for value in vector / norm)
        validate_vector(result)
        return result

    async def embed_query(self, query: EmbeddingQuery) -> Vector:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self._encode, "query: ", query.text, query.metadata
        )

    async def embed_documents(self, documents: tuple[SkillMetadata, ...]) -> tuple[Vector, ...]:
        result = []
        for document in documents:
            result.append(
                await asyncio.get_running_loop().run_in_executor(
                    self.executor, self._encode, "passage: ", document.text(), document
                )
            )
        return tuple(result)

    async def aclose(self) -> None:
        def close() -> None:
            self._session = None
            self._tokenizer = None
            self._numpy = None

        await asyncio.get_running_loop().run_in_executor(self.executor, close)

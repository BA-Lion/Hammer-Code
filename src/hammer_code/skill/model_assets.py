"""Explicit, pinned public model preparation; ordinary startup never downloads."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4

from hammer_code.skill.embedding import EmbeddingError


def manifest() -> dict[str, Any]:
    value = json.loads(Path(__file__).with_name("embedding-model.json").read_text("utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise EmbeddingError("invalid_model_manifest")
    return value


def is_reparse(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    info = path.lstat()
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def safe_directory(root: Path, relative: str, *, create: bool = False) -> Path:
    base = root.absolute()
    if os.name == "nt" and not str(base).startswith("\\\\?\\"):
        value = str(base)
        base = Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)
    destination = base / relative
    if ".." in Path(relative).parts or Path(relative).anchor:
        raise EmbeddingError("unsafe_asset_path")
    for part in (*reversed(destination.parents), destination):
        if is_reparse(part) or (part.exists() and not part.is_dir()):
            raise EmbeddingError("unsafe_asset_path")
        if create and not part.exists():
            part.mkdir()
    return destination


def model_directory(root: Path, *, create: bool = False) -> Path:
    return safe_directory(
        root,
        f".hammer-code/skill/state/models/intfloat-multilingual-e5-small/{manifest()['revision']}",
        create=create,
    )


def _file(directory: Path, name: str) -> Path:
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise EmbeddingError("unsafe_asset_path")
    path = directory / name
    if is_reparse(path) or (path.exists() and not path.is_file()):
        raise EmbeddingError("unsafe_asset_path")
    return path


def _valid(path: Path, asset: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == asset["sha256"]


def verify_assets(root: Path) -> Path:
    directory = model_directory(root)
    expected = manifest()
    ready = _file(directory, "manifest.json")
    if not ready.is_file():
        raise EmbeddingError("missing_assets")
    try:
        if json.loads(ready.read_text("utf-8")) != expected:
            raise EmbeddingError("invalid_assets")
        for asset in expected["assets"]:
            if not _valid(_file(directory, asset["name"]), asset):
                raise EmbeddingError("invalid_assets")
    except (OSError, ValueError, KeyError) as exc:
        raise EmbeddingError("invalid_assets") from exc
    return directory


async def _file_io(function, *args):
    """A cancelled preparation must finish its current write before closing the file."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def prepare_model(root: Path) -> Path:
    """Download only explicitly requested assets, publishing the ready marker last."""
    import httpx2

    directory = model_directory(root, create=True)
    expected = manifest()
    try:
        async with httpx2.AsyncClient(follow_redirects=True, timeout=120) as client:
            for asset in expected["assets"]:
                path = _file(directory, asset["name"])
                if await asyncio.to_thread(_valid, path, asset):
                    continue
                partial = _file(directory, f"{asset['name']}.{uuid4().hex}.part")
                digest = hashlib.sha256()
                size = 0
                try:
                    url = (
                        f"https://huggingface.co/{expected['model']}/resolve/"
                        f"{expected['revision']}/{asset['source']}"
                    )
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        if response.url.scheme != "https":
                            raise EmbeddingError("unsafe_download_origin")
                        with partial.open("xb") as stream:
                            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                                size += len(chunk)
                                if size > asset["size"]:
                                    raise EmbeddingError("invalid_download")
                                digest.update(chunk)
                                await _file_io(stream.write, chunk)
                            await _file_io(stream.flush)
                            await _file_io(os.fsync, stream.fileno())
                    if size != asset["size"] or digest.hexdigest() != asset["sha256"]:
                        raise EmbeddingError("invalid_download")
                    os.replace(partial, path)
                finally:
                    partial.unlink(missing_ok=True)
        ready = _file(directory, "manifest.json")
        temporary = _file(directory, f"manifest.{uuid4().hex}.part")
        try:
            temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
            os.replace(temporary, ready)
        finally:
            temporary.unlink(missing_ok=True)
        return directory
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise EmbeddingError("model_preparation_failed") from exc


def vector_status(root: Path) -> dict[str, object]:
    """Presence diagnostics only: no imports of optional runtimes, writes, or network."""
    dependencies = all(
        importlib.util.find_spec(name) for name in ("numpy", "tokenizers", "onnxruntime")
    )
    directory = model_directory(root)
    ready = _file(directory, "manifest.json").is_file() and all(
        _file(directory, asset["name"]).is_file() for asset in manifest()["assets"]
    )
    cache = safe_directory(root, ".hammer-code/skill/state") / "embeddings.sqlite3"
    return {
        "dependencies_present": dependencies,
        "assets_present": ready,
        "cache_present": cache.is_file(),
    }

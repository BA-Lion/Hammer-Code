from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hammer_code.skill import model_assets
from hammer_code.skill.embedding import EmbeddingError


@pytest.fixture
def fake_assets(monkeypatch):
    import httpx2

    content = b"complete fake model"
    specification = {
        "model": "example/model",
        "revision": "a" * 40,
        "assets": [
            {
                "name": "model.onnx",
                "source": "onnx/model.onnx",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(model_assets, "manifest", lambda: specification)
    calls = []

    class Response:
        url = SimpleNamespace(scheme="https")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def aiter_bytes(self, **kwargs):
            yield content[:5]
            yield content[5:]

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url):
            calls.append(url)
            return Response()

    monkeypatch.setattr(httpx2, "AsyncClient", Client)
    return specification, calls


async def test_prepare_atomic_verified_reuse_and_long_path(tmp_path: Path, fake_assets) -> None:
    specification, calls = fake_assets
    root = tmp_path / ("long-project-name" * 6)
    root.mkdir()
    directory = await model_assets.prepare_model(root)
    assert model_assets.verify_assets(root) == directory
    assert not list(directory.glob("*.part"))
    assert specification["revision"] in calls[0] and "/main/" not in calls[0]
    await model_assets.prepare_model(root)
    assert len(calls) == 1
    (directory / "model.onnx").write_bytes(b"broken")
    with pytest.raises(EmbeddingError):
        model_assets.verify_assets(root)


async def test_bad_download_never_publishes_manifest(tmp_path: Path, fake_assets) -> None:
    specification, _ = fake_assets
    specification["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(EmbeddingError):
        await model_assets.prepare_model(tmp_path)
    directory = model_assets.model_directory(tmp_path)
    assert not (directory / "manifest.json").exists()
    assert not list(directory.glob("*.part"))


def test_status_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.rglob("*"))
    status = model_assets.vector_status(tmp_path)
    assert not status["assets_present"] and not status["cache_present"]
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("name", ["../model", "x/model", "x\\model", ".."])
def test_asset_paths_cannot_escape(tmp_path: Path, name: str) -> None:
    with pytest.raises(EmbeddingError):
        model_assets._file(tmp_path, name)


def test_reparse_parent_rejected(tmp_path: Path, monkeypatch) -> None:
    real = model_assets.is_reparse
    monkeypatch.setattr(
        model_assets, "is_reparse", lambda path: path.name == ".hammer-code" or real(path)
    )
    with pytest.raises(EmbeddingError):
        model_assets.model_directory(tmp_path, create=True)

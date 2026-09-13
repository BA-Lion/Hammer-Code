from __future__ import annotations

from hammer_code.memory.models import MemoryBatch
from hammer_code.memory.store import MemoryStore


def test_memory_store_add_and_orphan_merge_identity(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    first = MemoryBatch.model_validate(
        {
            "operations": [
                {
                    "action": "add",
                    "category": "user-preferences",
                    "memory_id": "concise-output",
                    "path": "style.md",
                    "content": "Prefer concise output.",
                    "index_description": "Preferred response style",
                }
            ]
        }
    )
    store.commit_batch(store.prepare_batch(first))
    category = tmp_path / ".hammer-code" / "memory" / "user-preferences"
    (category / "orphan.md").write_text("Existing fact.\n", encoding="utf-8")
    repair = MemoryBatch.model_validate(
        {
            "operations": [
                {
                    "action": "merge",
                    "category": "user-preferences",
                    "memory_id": "orphan-fact",
                    "path": "orphan.md",
                    "content": ["- Durable follow-up"],
                    "index_description": "Recovered topic",
                }
            ]
        }
    )
    store.commit_batch(store.prepare_batch(repair))

    catalog = store.load_catalog()
    entries = catalog.categories[next(iter(catalog.categories))].index.entries
    assert {entry.memory_id for entry in entries} == {"concise-output", "orphan-fact"}
    assert (category / "orphan.md").read_text(
        encoding="utf-8"
    ) == "Existing fact.\n- Durable follow-up\n"

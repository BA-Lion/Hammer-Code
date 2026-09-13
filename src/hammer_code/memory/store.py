"""Validated, forward-only topic and Index writes for local Memory."""

from __future__ import annotations

import hashlib
import os
import stat
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from hammer_code.memory.index import MemoryIndexError, parse_index, serialize_index
from hammer_code.memory.models import (
    AddOperation,
    IndexEntry,
    MemoryBatch,
    MemoryCategory,
    MemoryIndex,
    MergeOperation,
    OverrideOperation,
    normalize_document,
    validate_description,
    validate_merge_lines,
    validate_path,
)
from hammer_code.session.session import atomic_write_text

_CATEGORIES = tuple(MemoryCategory)


class MemoryStoreError(ValueError):
    pass


@dataclass(frozen=True)
class CategoryCatalog:
    raw_index: str
    index_hash: str
    index: MemoryIndex
    topics: dict[str, str]
    orphans: tuple[str, ...]


@dataclass(frozen=True)
class MemoryCatalog:
    categories: dict[MemoryCategory, CategoryCatalog]


@dataclass(frozen=True)
class PreparedBatch:
    topic_writes: tuple[tuple[MemoryCategory, str, str], ...]
    indexes: dict[MemoryCategory, tuple[str, str, MemoryIndex]]


class MemoryStore:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve() / ".hammer-code" / "memory"

    def load_catalog(self) -> MemoryCatalog:
        categories: dict[MemoryCategory, CategoryCatalog] = {}
        for category in _CATEGORIES:
            directory = self.root / category.value
            index_path = directory / "index.md"
            if not directory.exists() and not index_path.exists():
                raw = ""
                index = MemoryIndex()
                topics: dict[str, str] = {}
            else:
                if not directory.is_dir() or _is_reparse(directory):
                    raise MemoryStoreError("Memory category has an unsafe identity")
                try:
                    raw = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
                    index = parse_index(raw)
                    topics = {
                        path.relative_to(directory).as_posix(): path.read_text(encoding="utf-8")
                        for path in directory.rglob("*.md")
                        if path.name != "index.md" and path.is_file() and not _is_reparse(path)
                    }
                except (OSError, UnicodeDecodeError, MemoryIndexError) as exc:
                    raise MemoryStoreError("Unable to load a Memory category") from exc
            referenced = {entry.path for entry in index.entries}
            categories[category] = CategoryCatalog(
                raw,
                hashlib.sha256(raw.encode()).hexdigest(),
                index,
                topics,
                tuple(sorted(set(topics) - referenced)),
            )
        return MemoryCatalog(categories)

    def prepare_batch(self, batch: MemoryBatch) -> PreparedBatch:
        catalog = self.load_catalog()
        targets: set[tuple[MemoryCategory, str]] = set()
        ids: dict[MemoryCategory, set[str]] = {
            category: {entry.memory_id for entry in value.index.entries}
            for category, value in catalog.categories.items()
        }
        writes: dict[tuple[MemoryCategory, str], str] = {}
        indexes: dict[MemoryCategory, OrderedDict[str, IndexEntry]] = {
            category: OrderedDict((entry.path, entry) for entry in value.index.entries)
            for category, value in catalog.categories.items()
        }
        affected: set[MemoryCategory] = set()
        for operation in batch.operations:
            if operation.action == "skip":
                continue
            path = validate_path(operation.path)
            key = (operation.category, path)
            if key in targets:
                raise MemoryStoreError("Memory batch has duplicate topic targets")
            targets.add(key)
            existing = catalog.categories[operation.category].topics.get(path)
            if isinstance(operation, AddOperation):
                if existing is not None or operation.memory_id in ids[operation.category]:
                    raise MemoryStoreError("Memory add conflicts with an existing topic or id")
                content = normalize_document(operation.content)
                description = validate_description(operation.index_description)
                ids[operation.category].add(operation.memory_id)
                indexes[operation.category][path] = IndexEntry(
                    memory_id=operation.memory_id, path=path, description=description
                )
            elif isinstance(operation, MergeOperation):
                if existing is None:
                    raise MemoryStoreError("Memory merge requires an existing topic")
                new_lines = validate_merge_lines(operation.content)
                existing_entry = indexes[operation.category].get(path)
                if (
                    existing_entry is not None
                    and operation.memory_id is not None
                    and operation.memory_id != existing_entry.memory_id
                ):
                    raise MemoryStoreError("Memory id does not match the existing Index entry")
                old_lines = existing.splitlines()
                additions = [line for line in new_lines if line not in set(old_lines)]
                content = (
                    normalize_document(
                        existing
                        + ("\n" if existing and not existing.endswith("\n") else "")
                        + "\n".join(additions)
                    )
                    if additions
                    else existing
                )
                if operation.index_description is not None:
                    description = validate_description(operation.index_description)
                    entry = indexes[operation.category].get(path)
                    if entry is None:
                        if operation.memory_id is None:
                            raise MemoryStoreError("Orphan merge requires a memory_id")
                        if operation.memory_id in ids[operation.category]:
                            raise MemoryStoreError("Memory id already exists in this category")
                        ids[operation.category].add(operation.memory_id)
                        indexes[operation.category][path] = IndexEntry(
                            memory_id=operation.memory_id, path=path, description=description
                        )
                    else:
                        if (
                            operation.memory_id is not None
                            and operation.memory_id != entry.memory_id
                        ):
                            raise MemoryStoreError(
                                "Memory id does not match the existing Index entry"
                            )
                        indexes[operation.category][path] = entry.model_copy(
                            update={"description": description}
                        )
                elif indexes[operation.category].get(path) is None:
                    raise MemoryStoreError("Orphan merge requires an Index description")
            elif isinstance(operation, OverrideOperation):
                if existing is None:
                    raise MemoryStoreError("Memory override requires an existing topic")
                content = normalize_document(operation.content)
                description = validate_description(operation.index_description)
                entry = indexes[operation.category].get(path)
                if entry is None:
                    if operation.memory_id is None:
                        raise MemoryStoreError("Orphan override requires a memory_id")
                    if operation.memory_id in ids[operation.category]:
                        raise MemoryStoreError("Memory id already exists in this category")
                    ids[operation.category].add(operation.memory_id)
                    indexes[operation.category][path] = IndexEntry(
                        memory_id=operation.memory_id, path=path, description=description
                    )
                else:
                    if operation.memory_id is not None and operation.memory_id != entry.memory_id:
                        raise MemoryStoreError("Memory id does not match the existing Index entry")
                    indexes[operation.category][path] = entry.model_copy(
                        update={"description": description}
                    )
            else:
                raise MemoryStoreError("Unknown Memory operation")
            if content != existing:
                writes[key] = content
            if (
                content != existing
                or operation.index_description is not None
                or isinstance(operation, AddOperation)
            ):
                affected.add(operation.category)
        index_updates: dict[MemoryCategory, tuple[str, str, MemoryIndex]] = {}
        for category in affected:
            before = catalog.categories[category]
            index = MemoryIndex(entries=tuple(indexes[category].values()))
            index_updates[category] = (before.index_hash, serialize_index(index), index)
        return PreparedBatch(
            tuple(
                (category, path, content)
                for (category, path), content in sorted(
                    writes.items(), key=lambda item: (item[0][0].value, item[0][1])
                )
            ),
            index_updates,
        )

    def commit_batch(self, prepared: PreparedBatch) -> None:
        writes_by_category: dict[MemoryCategory, list[tuple[str, str]]] = {}
        for category, path, content in prepared.topic_writes:
            writes_by_category.setdefault(category, []).append((path, content))
        for category in _CATEGORIES:
            directory = self.root / category.value
            for path, content in writes_by_category.get(category, []):
                target = self._topic_path(directory, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(target, content)
            update = prepared.indexes.get(category)
            if update is None:
                continue
            previous_hash, text, expected = update
            index_path = directory / "index.md"
            current = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
            if hashlib.sha256(current.encode()).hexdigest() != previous_hash:
                raise MemoryStoreError("Memory Index changed concurrently")
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_text(index_path, text)
            if parse_index(index_path.read_text(encoding="utf-8")) != expected:
                raise MemoryStoreError("Memory Index verification failed")
        self._verify(prepared)

    def _topic_path(self, directory: Path, relative: str) -> Path:
        target = directory / validate_path(relative)
        if os.path.commonpath((str(directory.resolve()), str(target.resolve().parent))) != str(
            directory.resolve()
        ):
            raise MemoryStoreError("Memory topic path escapes its category")
        return target

    def _verify(self, prepared: PreparedBatch) -> None:
        for category, path, content in prepared.topic_writes:
            target = self._topic_path(self.root / category.value, path)
            if target.read_text(encoding="utf-8") != content:
                raise MemoryStoreError("Memory topic verification failed")


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)

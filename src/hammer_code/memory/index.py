"""Deterministic parser and serializer for a category's fixed Markdown index."""

from __future__ import annotations

import re

from hammer_code.memory.models import IndexEntry, MemoryIndex, validate_description, validate_path

_ENTRY = re.compile(r"^- \[([a-z0-9][a-z0-9-]{0,63})\]\(([^)]+)\): (.+)$")


class MemoryIndexError(ValueError):
    pass


def parse_index(text: str) -> MemoryIndex:
    if not text:
        return MemoryIndex()
    entries: list[IndexEntry] = []
    ids: set[str] = set()
    paths: set[str] = set()
    for line in text.splitlines():
        match = _ENTRY.fullmatch(line)
        if match is None:
            raise MemoryIndexError("Memory Index contains an invalid entry")
        memory_id, path, description = match.groups()
        try:
            path = validate_path(path)
            description = validate_description(description)
        except ValueError as exc:
            raise MemoryIndexError("Memory Index contains an invalid entry") from exc
        if memory_id in ids or path in paths:
            raise MemoryIndexError("Memory Index contains duplicate ids or paths")
        ids.add(memory_id)
        paths.add(path)
        entries.append(IndexEntry(memory_id=memory_id, path=path, description=description))
    return MemoryIndex(entries=tuple(entries))


def serialize_index(index: MemoryIndex) -> str:
    return "".join(
        f"- [{entry.memory_id}]({entry.path}): {entry.description}\n" for entry in index.entries
    )

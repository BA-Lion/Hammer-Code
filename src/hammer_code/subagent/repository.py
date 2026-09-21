"""Safe direct-child Subagent discovery with last-known-good reload semantics."""

from __future__ import annotations

import asyncio
import hashlib
import stat
from collections.abc import Callable
from pathlib import Path

from hammer_code.config import SubagentConfig
from hammer_code.errors import ConfigurationError
from hammer_code.subagent.models import SubagentCatalogSnapshot, empty_catalog_snapshot
from hammer_code.subagent.parser import SubagentParseError, parse_subagent


class SubagentRepositoryError(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class SubagentRepository:
    """Owns the immutable catalog used by one process's PrimaryAgent factories."""

    def __init__(
        self,
        workspace_root: Path,
        config: SubagentConfig,
        warning: Callable[[str], None] | None = None,
    ) -> None:
        self.root = (workspace_root.resolve() / ".hammer-code" / "subagents").resolve()
        self.config = config
        self._warning = warning
        self._current = empty_catalog_snapshot()
        self._lock = asyncio.Lock()
        self._last_warned_fingerprint: str | None = None

    def _paths_and_fingerprint(self) -> tuple[tuple[Path, ...], str]:
        if not self.root.exists():
            return (), hashlib.sha256().hexdigest()
        if not self.root.is_dir() or _is_reparse(self.root):
            raise SubagentRepositoryError("Subagent catalog root is unsafe")
        try:
            entries = tuple(sorted(self.root.iterdir(), key=lambda entry: entry.name))
        except OSError as exc:
            raise SubagentRepositoryError("Subagent catalog root could not be scanned") from exc
        paths: list[Path] = []
        digest = hashlib.sha256()
        for entry in entries:
            if entry.suffix.lower() != ".md":
                continue
            if not entry.is_file() or _is_reparse(entry):
                raise SubagentRepositoryError("Subagent catalog contains an unsafe definition")
            try:
                resolved = entry.resolve()
                if resolved.parent != self.root:
                    raise SubagentRepositoryError("Subagent definition escapes its catalog")
                payload = entry.read_bytes()
            except OSError as exc:
                raise SubagentRepositoryError("Subagent definition could not be read") from exc
            digest.update(entry.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
            paths.append(entry)
        return tuple(paths), digest.hexdigest()

    def _build_snapshot(self, previous: SubagentCatalogSnapshot | None) -> SubagentCatalogSnapshot:
        paths, fingerprint = self._paths_and_fingerprint()
        if previous is not None and previous.fingerprint == fingerprint:
            return previous
        definitions = {}
        for path in paths:
            try:
                definition = parse_subagent(path, self.config)
            except SubagentParseError as exc:
                raise SubagentRepositoryError(
                    f"Invalid Subagent definition '{path.name}': {exc}"
                ) from exc
            if definition.name in definitions:
                raise SubagentRepositoryError("Duplicate Subagent name")
            definitions[definition.name] = definition
        return SubagentCatalogSnapshot(
            generation=(previous.generation + 1) if previous is not None else 1,
            fingerprint=fingerprint,
            definitions=definitions,
        )

    async def initialize(self) -> SubagentCatalogSnapshot:
        """Strict initial load: a malformed existing catalog prevents startup."""
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self._build_snapshot, None)
            except SubagentRepositoryError as exc:
                raise ConfigurationError("Subagent catalog configuration is invalid") from exc
            self._last_warned_fingerprint = None
            return self._current

    async def snapshot_for_request(self) -> SubagentCatalogSnapshot:
        """Atomically publish a complete new snapshot or retain the prior complete one."""
        async with self._lock:
            try:
                self._current = await asyncio.to_thread(self._build_snapshot, self._current)
                self._last_warned_fingerprint = None
            except SubagentRepositoryError as exc:
                # A fingerprint is intentionally best-effort here: failed reads may prevent
                # obtaining one, but warnings still remain bounded until a successful reload.
                marker = str(exc)
                if self._warning is not None and marker != self._last_warned_fingerprint:
                    self._warning(f"Subagent catalog reload ignored: {exc}")
                    self._last_warned_fingerprint = marker
            return self._current

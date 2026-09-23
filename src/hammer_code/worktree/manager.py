"""Process-local lease lifecycle and safe, non-index-mutating result integration."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hammer_code.subagent.models import SubagentWorkspace
from hammer_code.worktree.git import GitError, GitRunner, RepositoryOwnershipLock
from hammer_code.worktree.models import (
    WorktreeLease,
    WorktreeLeaseState,
    WorktreeResolution,
    WorktreeResultMetadata,
)

_MAX_CHANGED_FILES = 200


class WorktreeUnavailableError(RuntimeError):
    pass


@dataclass
class _Record:
    lease: WorktreeLease
    integration_root: Path
    parent_task_id: str | None = None
    result_commit: str | None = None
    changed_files: tuple[str, ...] = ()
    truncated: bool = False
    inspection_id: str | None = None
    conflicts: tuple[str, ...] = ()


class WorktreeManager:
    """Own linked worktrees without mutating the caller's branch, HEAD, or index."""

    def __init__(self, workspace_root: Path, *, git: GitRunner | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.git = git or GitRunner()
        self._lock = asyncio.Lock()
        self._ownership: RepositoryOwnershipLock | None = None
        self._managed_root: Path | None = None
        self._records: dict[str, _Record] = {}
        self._available = False
        self._closed = False

    @property
    def available(self) -> bool:
        return self._available and not self._closed

    async def initialize(self) -> bool:
        """Validate repository identity, acquire ownership, then remove verified orphans."""
        async with self._lock:
            if self._closed:
                return False
            if self._available:
                return True
            try:
                top = (
                    (await self.git.run("rev-parse", "--show-toplevel", cwd=self.workspace_root))
                    .stdout.decode()
                    .strip()
                )
                common = (
                    (await self.git.run("rev-parse", "--git-common-dir", cwd=self.workspace_root))
                    .stdout.decode()
                    .strip()
                )
                bare = (
                    (
                        await self.git.run(
                            "rev-parse", "--is-bare-repository", cwd=self.workspace_root
                        )
                    )
                    .stdout.decode()
                    .strip()
                )
                if Path(top).resolve() != self.workspace_root or bare != "false":
                    return False
                common_dir = (
                    (self.workspace_root / common).resolve()
                    if not Path(common).is_absolute()
                    else Path(common).resolve()
                )
                if not common_dir.is_dir():
                    return False
                ownership = RepositoryOwnershipLock(common_dir / "hammer-code-worktree.lock")
                if not ownership.acquire():
                    return False
                self._ownership = ownership
                self._managed_root = (self.workspace_root / ".hammer-code" / "worktrees").resolve()
                await self._cleanup_orphans_locked()
                self._available = True
                return True
            except (GitError, OSError):
                if self._ownership is not None:
                    self._ownership.release()
                    self._ownership = None
                return False

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            for record in tuple(self._records.values()):
                if record.lease.state is not WorktreeLeaseState.CLOSED:
                    try:
                        await self._remove_locked(record)
                    except (GitError, OSError):
                        pass
            self._records.clear()
            if self._ownership is not None:
                self._ownership.release()
                self._ownership = None
            self._available = False
            self._closed = True

    async def create(
        self, task_id: str, relative_cwd: Path, initialization_files: tuple[str, ...] = ()
    ) -> WorktreeLease:
        async with self._lock:
            self._require_available()
            if task_id in self._records or not _is_uuid(task_id):
                raise WorktreeUnavailableError("worktree task id is invalid or already active")
            relative = Path(".") if relative_cwd == Path(".") else _safe_relative(relative_cwd)
            baseline = await self._snapshot_commit_locked()
            assert self._managed_root is not None
            path = (self._managed_root / task_id).resolve()
            self._assert_managed_path(path)
            try:
                await self.git.run(
                    "worktree", "add", "--detach", str(path), baseline, cwd=self.workspace_root
                )
                child_cwd = (path / relative).resolve()
                if not child_cwd.is_dir() or not _contained(child_cwd, path):
                    raise WorktreeUnavailableError(
                        "the requested child working directory is unavailable"
                    )
                await self._copy_initialization_files_locked(path, initialization_files)
            except Exception:
                try:
                    await self._remove_path_locked(path)
                except Exception:
                    pass
                raise
            lease = WorktreeLease(
                task_id, path, baseline, relative, datetime.now(UTC), WorktreeLeaseState.ACTIVE
            )
            self._records[task_id] = _Record(lease, self.workspace_root)
            return lease

    async def create_child(
        self,
        task_id: str,
        parent_task_id: str,
        relative_cwd: Path,
        initialization_files: tuple[str, ...] = (),
    ) -> WorktreeLease:
        """Create a linked child whose only permitted integration target is its parent.

        The caller supplies opaque IDs only.  Source, target and commit are recovered
        under the manager lock, so a model cannot redirect a child into another tree.
        """
        async with self._lock:
            self._require_available()
            if task_id in self._records or not _is_uuid(task_id):
                raise WorktreeUnavailableError("worktree task id is invalid or already active")
            parent = self._record(parent_task_id)
            if (
                parent.parent_task_id is not None
                or parent.lease.state is not WorktreeLeaseState.ACTIVE
            ):
                raise WorktreeUnavailableError("worktree parent is not an active Team root")
            if not _contained(parent.lease.path, self._managed_root or self.workspace_root):
                raise WorktreeUnavailableError("worktree parent is unsafe")
            relative = Path(".") if relative_cwd == Path(".") else _safe_relative(relative_cwd)
            baseline = await self._snapshot_commit_locked(parent.lease.path)
            assert self._managed_root is not None
            path = (self._managed_root / task_id).resolve()
            self._assert_managed_path(path)
            try:
                await self.git.run(
                    "worktree", "add", "--detach", str(path), baseline, cwd=self.workspace_root
                )
                child_cwd = (path / relative).resolve()
                if not child_cwd.is_dir() or not _contained(child_cwd, path):
                    raise WorktreeUnavailableError(
                        "the requested child working directory is unavailable"
                    )
                await self._copy_initialization_files_locked(
                    path, initialization_files, source_root=parent.lease.path
                )
            except Exception:
                try:
                    await self._remove_path_locked(path)
                except Exception:
                    pass
                raise
            lease = WorktreeLease(
                task_id,
                path,
                baseline,
                relative,
                datetime.now(UTC),
                WorktreeLeaseState.ACTIVE,
                parent_task_id,
            )
            self._records[task_id] = _Record(lease, parent.lease.path, parent_task_id)
            return lease

    async def finalize(self, task_id: str) -> WorktreeResultMetadata:
        async with self._lock:
            record = self._record(task_id)
            if record.parent_task_id is None and self._has_active_children(record.lease.task_id):
                raise WorktreeUnavailableError("children_active")
            if record.lease.state is not WorktreeLeaseState.ACTIVE:
                return self._metadata(record)
            result = await self._result_commit_locked(record)
            record.result_commit = result
            changed = await self._changed_files_locked(record.lease.baseline_commit, result)
            record.changed_files, record.truncated = changed
            if not record.changed_files and not record.truncated:
                await self._remove_locked(record)
                return WorktreeResultMetadata(
                    task_id,
                    SubagentWorkspace.WORKTREE,
                    "none",
                    record.lease.baseline_commit,
                    result,
                )
            record.lease = replace(record.lease, state=WorktreeLeaseState.PENDING_INTEGRATION)
            return self._metadata(record)

    async def discard(self, task_id: str) -> WorktreeResolution:
        async with self._lock:
            record = self._record(task_id)
            if record.parent_task_id is None and self._has_active_children(record.lease.task_id):
                raise WorktreeUnavailableError("children_active")
            await self._remove_locked(record)
            return WorktreeResolution(task_id, "discarded", False, (), "closed")

    async def discard_all(self) -> None:
        async with self._lock:
            for record in tuple(self._records.values()):
                if record.lease.state is not WorktreeLeaseState.CLOSED:
                    await self._remove_locked(record)

    async def inspect(
        self, task_id: str, detail: str = "summary", paths: tuple[str, ...] = ()
    ) -> dict[str, object]:
        async with self._lock:
            record = self._record(task_id)
            if record.lease.state is WorktreeLeaseState.CLOSED:
                raise WorktreeUnavailableError("worktree draft is unavailable")
            if detail == "summary":
                return self._summary(record)
            if detail == "diff":
                self._validate_paths(paths, record.changed_files)
                assert record.result_commit is not None
                args = [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    record.lease.baseline_commit,
                    record.result_commit,
                    "--",
                ]
                args.extend(paths)
                output = await self.git.run(*args, cwd=record.integration_root)
                return {
                    "task_id": task_id,
                    "detail": "diff",
                    "diff": output.stdout.decode("utf-8", errors="replace")[: 4 * 1024 * 1024],
                }
            if detail == "merge_inputs":
                self._validate_paths(paths, record.conflicts or record.changed_files)
                values = {path: await self._merge_input_locked(record, path) for path in paths}
                record.inspection_id = str(uuid4())
                return {
                    "task_id": task_id,
                    "detail": "merge_inputs",
                    "inspection_id": record.inspection_id,
                    "files": values,
                }
            raise ValueError("inspect detail is invalid")

    async def integrate(self, task_id: str) -> WorktreeResolution:
        async with self._lock:
            record = self._record(task_id)
            return await self._integrate_locked(record)

    async def agent_merge(
        self, task_id: str, inspection_id: str, edits: tuple[dict[str, object], ...]
    ) -> WorktreeResolution:
        async with self._lock:
            record = self._record(task_id)
            if (
                record.lease.state is not WorktreeLeaseState.CONFLICT
                or inspection_id != record.inspection_id
            ):
                return WorktreeResolution(
                    task_id,
                    "invalid_resolution",
                    False,
                    (),
                    "inspect the current conflict first",
                    "invalid_resolution",
                )
            supplied = {str(item.get("path", "")) for item in edits}
            if supplied != set(record.conflicts):
                return WorktreeResolution(
                    task_id,
                    "invalid_resolution",
                    False,
                    record.conflicts,
                    "provide every conflict path",
                    "invalid_resolution",
                )
            before = await self._target_fingerprints_locked(record, record.conflicts)
            targets = tuple(
                (record.integration_root / _safe_relative(Path(path))).resolve()
                for path in record.changed_files
            )
            backups = {target: _backup(target) for target in targets}
            non_conflicts = tuple(
                path for path in record.changed_files if path not in record.conflicts
            )
            index_before = (await self.git.run("write-tree", cwd=record.integration_root)).stdout
            try:
                if non_conflicts:
                    assert record.result_commit is not None
                    patch = (
                        await self.git.run(
                            "diff",
                            "--binary",
                            "--full-index",
                            "--no-ext-diff",
                            record.lease.baseline_commit,
                            record.result_commit,
                            "--",
                            *non_conflicts,
                            cwd=record.integration_root,
                        )
                    ).stdout
                    await self.git.run(
                        "apply",
                        "--check",
                        "--whitespace=nowarn",
                        cwd=record.integration_root,
                        input_data=patch,
                    )
                    await self.git.run(
                        "apply",
                        "--whitespace=nowarn",
                        cwd=record.integration_root,
                        input_data=patch,
                    )
                for item in edits:
                    path = _safe_relative(Path(str(item["path"])))
                    target = (record.integration_root / path).resolve()
                    if not _contained(target, record.integration_root):
                        raise ValueError("invalid merge path")
                    expected = str(item.get("expected_current_hash", ""))
                    if before[str(path)] != expected:
                        return WorktreeResolution(
                            task_id, "stale", False, (str(path),), "inspect again", "stale"
                        )
                    existed = target.is_file()
                    if item.get("action") == "delete":
                        if existed:
                            target.unlink()
                    elif item.get("action") == "write":
                        content = item.get("content")
                        if (
                            not isinstance(content, str)
                            or len(content.encode("utf-8")) > 4 * 1024 * 1024
                        ):
                            raise ValueError("merge content is invalid or too large")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(content, encoding="utf-8", newline="")
                    else:
                        raise ValueError("merge action is invalid")
                index_after = (await self.git.run("write-tree", cwd=record.integration_root)).stdout
                if index_after != index_before:
                    raise GitError("agent merge unexpectedly changed the index")
            except Exception:
                for target, (existed, data, mode) in backups.items():
                    if existed:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        if mode is not None:
                            os.chmod(target, mode)
                    elif target.exists():
                        target.unlink()
                return WorktreeResolution(
                    task_id,
                    "failed",
                    bool(backups),
                    tuple(str(path.relative_to(record.integration_root)) for path in backups),
                    "manual recovery may be required",
                    "failed",
                )
            await self._remove_locked(record)
            return WorktreeResolution(
                task_id,
                "integrated_by_leader"
                if record.parent_task_id is not None
                else "integrated_by_primary",
                True,
                record.changed_files,
                "run relevant verification",
            )

    def metadata(self, task_id: str) -> WorktreeResultMetadata:
        return self._metadata(self._record(task_id))

    def _require_available(self) -> None:
        if not self.available:
            raise WorktreeUnavailableError("worktree isolation is unavailable")

    def _record(self, task_id: str) -> _Record:
        record = self._records.get(task_id)
        if record is None:
            raise WorktreeUnavailableError("worktree draft is unavailable")
        return record

    async def _snapshot_commit_locked(self, source_root: Path | None = None) -> str:
        source = source_root or self.workspace_root
        head = (await self.git.run("rev-parse", "HEAD", cwd=source)).stdout.decode().strip()
        index = self._temporary_index("snapshot")
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        try:
            await self.git.run("read-tree", "HEAD", cwd=source, env=env)
            await self.git.run("add", "-A", cwd=source, env=env)
            tree = (await self.git.run("write-tree", cwd=source, env=env)).stdout.decode().strip()
            head_tree = (
                (await self.git.run("rev-parse", "HEAD^{tree}", cwd=source)).stdout.decode().strip()
            )
            if tree == head_tree:
                return head
            return (
                (
                    await self.git.run(
                        "commit-tree",
                        tree,
                        "-p",
                        head,
                        "-m",
                        "hammer-code worktree snapshot",
                        cwd=source,
                        env=env,
                    )
                )
                .stdout.decode()
                .strip()
            )
        finally:
            index.unlink(missing_ok=True)

    async def _result_commit_locked(self, record: _Record) -> str:
        index = self._temporary_index("result")
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        try:
            await self.git.run("read-tree", "HEAD", cwd=record.lease.path, env=env)
            await self.git.run("add", "-A", cwd=record.lease.path, env=env)
            tree = (
                (await self.git.run("write-tree", cwd=record.lease.path, env=env))
                .stdout.decode()
                .strip()
            )
            base_tree = (
                (
                    await self.git.run(
                        "rev-parse",
                        f"{record.lease.baseline_commit}^{{tree}}",
                        cwd=record.lease.path,
                    )
                )
                .stdout.decode()
                .strip()
            )
            if tree == base_tree:
                return record.lease.baseline_commit
            return (
                (
                    await self.git.run(
                        "commit-tree",
                        tree,
                        "-p",
                        record.lease.baseline_commit,
                        "-m",
                        "hammer-code worktree result",
                        cwd=record.lease.path,
                        env=env,
                    )
                )
                .stdout.decode()
                .strip()
            )
        finally:
            index.unlink(missing_ok=True)

    async def _changed_files_locked(
        self, baseline: str, result: str
    ) -> tuple[tuple[str, ...], bool]:
        output = await self.git.run(
            "diff", "--name-only", "-z", baseline, result, cwd=self.workspace_root
        )
        all_paths = tuple(
            path
            for path in output.stdout.decode("utf-8", errors="surrogateescape").split("\0")
            if path
        )
        return all_paths[:_MAX_CHANGED_FILES], len(all_paths) > _MAX_CHANGED_FILES

    async def _integrate_locked(self, record: _Record) -> WorktreeResolution:
        if (
            record.lease.state
            not in {WorktreeLeaseState.PENDING_INTEGRATION, WorktreeLeaseState.CONFLICT}
            or record.result_commit is None
        ):
            return WorktreeResolution(
                record.lease.task_id,
                "invalid_resolution",
                False,
                (),
                "inspect draft first",
                "invalid_resolution",
            )
        conflicts = await self._conflicts_locked(record)
        if conflicts:
            record.conflicts = conflicts
            record.lease = replace(record.lease, state=WorktreeLeaseState.CONFLICT)
            return WorktreeResolution(
                record.lease.task_id,
                "conflict",
                False,
                conflicts,
                "inspect merge_inputs",
                "conflict",
            )
        patch = (
            await self.git.run(
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                record.lease.baseline_commit,
                record.result_commit,
                cwd=record.integration_root,
            )
        ).stdout
        before_index = (await self.git.run("write-tree", cwd=record.integration_root)).stdout
        try:
            await self.git.run(
                "apply",
                "--check",
                "--whitespace=nowarn",
                cwd=record.integration_root,
                input_data=patch,
            )
            await self.git.run(
                "apply", "--whitespace=nowarn", cwd=record.integration_root, input_data=patch
            )
            after_index = (await self.git.run("write-tree", cwd=record.integration_root)).stdout
            if before_index != after_index:
                raise GitError("integration unexpectedly changed the index")
        except GitError:
            return WorktreeResolution(
                record.lease.task_id,
                "stale",
                False,
                record.changed_files,
                "inspect and resolve",
                "stale",
            )
        await self._remove_locked(record)
        return WorktreeResolution(
            record.lease.task_id, "integrated", True, record.changed_files, "closed"
        )

    async def _conflicts_locked(self, record: _Record) -> tuple[str, ...]:
        conflicts: list[str] = []
        for path in record.changed_files:
            changed = await self.git.run(
                "diff",
                "--quiet",
                record.lease.baseline_commit,
                "--",
                path,
                cwd=record.integration_root,
                check=False,
            )
            if changed.returncode != 0:
                conflicts.append(path)
        return tuple(conflicts)

    async def _merge_input_locked(self, record: _Record, path: str) -> dict[str, object]:
        relative = _safe_relative(Path(path))
        return {
            "base": await self._object_text(record.lease.baseline_commit, str(relative)),
            "current": _file_text(record.integration_root / relative),
            "result": await self._object_text(
                record.result_commit or record.lease.baseline_commit, str(relative)
            ),
            "current_hash": _hash_path(record.integration_root / relative),
        }

    async def _object_text(self, commit: str, path: str) -> dict[str, object]:
        result = await self.git.run(
            "show", f"{commit}:{path}", cwd=self.workspace_root, check=False
        )
        if result.returncode:
            return {"state": "unavailable"}
        try:
            return {
                "state": "text",
                "content": result.stdout.decode("utf-8"),
                "sha256": hashlib.sha256(result.stdout).hexdigest(),
            }
        except UnicodeDecodeError:
            return {"state": "binary", "sha256": hashlib.sha256(result.stdout).hexdigest()}

    async def _target_fingerprints_locked(
        self, record: _Record, paths: tuple[str, ...]
    ) -> dict[str, str]:
        return {
            path: _hash_path(record.integration_root / _safe_relative(Path(path))) for path in paths
        }

    async def _copy_initialization_files_locked(
        self, child: Path, paths: tuple[str, ...], *, source_root: Path | None = None
    ) -> None:
        source_root = source_root or self.workspace_root
        for item in paths:
            relative = _safe_relative(Path(item))
            source = (source_root / relative).resolve()
            target = (child / relative).resolve()
            if (
                not _contained(source, source_root)
                or not _contained(target, child)
                or not source.is_file()
                or source.is_symlink()
            ):
                raise WorktreeUnavailableError("initialization file is unsafe")
            ignored = await self.git.run(
                "check-ignore", "--quiet", "--", str(relative), cwd=source_root, check=False
            )
            if ignored.returncode != 0:
                raise WorktreeUnavailableError("initialization file must be ignored")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    async def _remove_locked(self, record: _Record) -> None:
        await self._remove_path_locked(record.lease.path)
        record.lease = replace(record.lease, state=WorktreeLeaseState.CLOSED)
        self._records.pop(record.lease.task_id, None)

    async def _remove_path_locked(self, path: Path) -> None:
        self._assert_managed_path(path)
        listed = await self.git.run("worktree", "list", "--porcelain", cwd=self.workspace_root)
        if f"worktree {path}" not in listed.stdout.decode("utf-8", errors="replace"):
            return
        await self.git.run("worktree", "remove", "--force", str(path), cwd=self.workspace_root)
        await self.git.run("worktree", "prune", cwd=self.workspace_root)

    async def _cleanup_orphans_locked(self) -> None:
        assert self._managed_root is not None
        listed = await self.git.run("worktree", "list", "--porcelain", cwd=self.workspace_root)
        for line in listed.stdout.decode("utf-8", errors="replace").splitlines():
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ")).resolve()
                if path != self.workspace_root and _contained(path, self._managed_root):
                    await self._remove_path_locked(path)
        await self.git.run("worktree", "prune", cwd=self.workspace_root)

    def _temporary_index(self, prefix: str) -> Path:
        runtime = self.workspace_root / ".hammer-code" / "runtime" / "worktree-transactions"
        runtime.mkdir(parents=True, exist_ok=True)
        return runtime / f"{prefix}-{uuid4().hex}.index"

    def _assert_managed_path(self, path: Path) -> None:
        if (
            self._managed_root is None
            or not _contained(path, self._managed_root)
            or path == self.workspace_root
        ):
            raise WorktreeUnavailableError("worktree path is outside the managed root")

    def _metadata(self, record: _Record) -> WorktreeResultMetadata:
        return WorktreeResultMetadata(
            record.lease.task_id,
            SubagentWorkspace.WORKTREE,
            "pending_integration"
            if record.lease.state
            in {WorktreeLeaseState.PENDING_INTEGRATION, WorktreeLeaseState.CONFLICT}
            else "none",
            record.lease.baseline_commit,
            record.result_commit,
            record.changed_files,
            record.truncated,
        )

    def _has_active_children(self, parent_task_id: str) -> bool:
        return any(item.parent_task_id == parent_task_id for item in self._records.values())

    def _summary(self, record: _Record) -> dict[str, object]:
        return {
            "task_id": record.lease.task_id,
            "state": record.lease.state.value,
            "baseline_commit": record.lease.baseline_commit,
            "result_commit": record.result_commit,
            "changed_files": record.changed_files,
            "truncated": record.truncated,
            "conflicts": record.conflicts,
        }

    @staticmethod
    def _validate_paths(paths: tuple[str, ...], allowed: tuple[str, ...]) -> None:
        if len(paths) > 20 or any(str(_safe_relative(Path(path))) not in allowed for path in paths):
            raise ValueError("inspect paths are invalid")


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path must be a normalized repository-relative path")
    return path


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid4().__class__(value)) == value
    except ValueError:
        return False


def _hash_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _backup(path: Path) -> tuple[bool, bytes, int | None]:
    existed = path.is_file()
    return (
        existed,
        path.read_bytes() if existed else b"",
        stat.S_IMODE(path.stat().st_mode) if existed else None,
    )


def _file_text(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"state": "missing"}
    value = path.read_bytes()
    try:
        return {
            "state": "text",
            "content": value.decode("utf-8"),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    except UnicodeDecodeError:
        return {"state": "binary", "sha256": hashlib.sha256(value).hexdigest()}

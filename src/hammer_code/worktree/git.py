"""Small argv-only Git adapter and process-held repository ownership lock."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitOutput:
    stdout: bytes
    stderr: bytes
    returncode: int


class GitRunner:
    """Run Git without a shell; diagnostics remain bounded and secret-free."""

    def __init__(
        self, *, timeout_seconds: float = 30.0, output_limit: int = 4 * 1024 * 1024
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit

    async def run(
        self,
        *args: str,
        cwd: Path,
        env: dict[str, str] | None = None,
        input_data: bytes | None = None,
        check: bool = True,
    ) -> GitOutput:
        if not args or any("\x00" in arg for arg in args):
            raise GitError("invalid Git arguments")
        child_env = dict(os.environ if env is None else env)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=cwd,
                env=child_env,
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise GitError("Git is unavailable") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data), self.timeout_seconds
            )
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise GitError("Git command timed out") from exc
        if len(stdout) > self.output_limit or len(stderr) > self.output_limit:
            raise GitError("Git command output exceeded the safety limit")
        result = GitOutput(stdout, stderr, process.returncode or 0)
        if check and result.returncode:
            raise GitError(self._message(result.stderr))
        return result

    @staticmethod
    def _message(stderr: bytes) -> str:
        text = (
            stderr.decode("utf-8", errors="replace").strip().replace("\r", " ").replace("\n", " ")
        )
        return f"Git command failed: {text[:500]}" if text else "Git command failed"


class RepositoryOwnershipLock:
    """An advisory OS lock retained for one manager lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: object | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)  # type: ignore[union-attr]
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        finally:
            handle.close()  # type: ignore[union-attr]

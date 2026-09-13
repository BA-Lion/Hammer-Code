"""Bounded loading of optional project-local ``hammer-code.md`` instructions."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from hammer_code.errors import HammerCodeError

_INCLUDE = re.compile(r"^@include\(([^()]+)\)$")


class ProjectInstructionError(HammerCodeError):
    pass


class ProjectInstructionLoader:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve(strict=True)

    def load(self) -> str:
        root_file = self.project_root / "hammer-code.md"
        if not root_file.exists():
            return ""
        return self._read(root_file, depth=0, chain=())

    def _read(self, candidate: Path, *, depth: int, chain: tuple[Path, ...]) -> str:
        if (
            "\x00" in str(candidate)
            or candidate.is_absolute()
            and candidate == Path(candidate.anchor)
        ):
            raise ProjectInstructionError("Project instruction include has an invalid path")
        if depth > 5:
            raise ProjectInstructionError("Project instruction include depth exceeds five")
        try:
            if _is_reparse(candidate) or not candidate.is_file():
                raise ProjectInstructionError("Project instruction include must be a regular file")
            resolved = candidate.resolve(strict=True)
            if os.path.commonpath((str(self.project_root), str(resolved))) != str(
                self.project_root
            ):
                raise ProjectInstructionError(
                    "Project instruction include escapes the project root"
                )
            if resolved in chain:
                raise ProjectInstructionError("Project instruction include cycle detected")
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ProjectInstructionError("Project instruction files must be UTF-8") from exc
        except OSError as exc:
            raise ProjectInstructionError("Unable to read project instruction include") from exc
        output: list[str] = []
        for line in text.splitlines():
            match = _INCLUDE.fullmatch(line)
            if match is None:
                output.append(line)
                continue
            target = match.group(1)
            raw = Path(target)
            if raw.is_absolute() or "\x00" in target:
                raise ProjectInstructionError("Project instruction include must be relative")
            output.append(
                self._read(resolved.parent / raw, depth=depth + 1, chain=(*chain, resolved))
            )
        return "\n".join(output).replace("\r\n", "\n").replace("\r", "\n")


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)

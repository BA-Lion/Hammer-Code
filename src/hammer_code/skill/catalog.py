"""Safe direct-child Skill discovery and immutable catalog snapshots."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

from hammer_code.skill.models import CatalogSnapshot, SkillDefinition, SkillScope
from hammer_code.skill.parser import SkillParseError, parse_skill
from hammer_code.skill.retrieval import Bm25Index


class SkillCatalogError(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class SkillCatalog:
    def __init__(self, project_root: Path) -> None:
        self.root = (project_root.resolve() / ".hammer-code" / "skill").resolve()

    def _scope_root(self, scope: SkillScope) -> Path:
        return self.root / scope.value

    def _children(self, scope: SkillScope) -> tuple[Path, ...]:
        root = self._scope_root(scope)
        if not root.exists():
            return ()
        if not root.is_dir() or _is_reparse(root):
            raise SkillCatalogError(f"Skill {scope.value} root is unsafe")
        try:
            result = tuple(sorted(root.iterdir(), key=lambda item: item.name))
        except OSError as exc:
            raise SkillCatalogError("Skill root could not be scanned") from exc
        for child in result:
            if not child.is_dir() or _is_reparse(child):
                raise SkillCatalogError("Skill root contains an unsafe entry")
            try:
                if child.resolve().parent != root.resolve():
                    raise SkillCatalogError("Skill directory escapes its scope")
            except OSError as exc:
                raise SkillCatalogError("Skill directory could not be resolved") from exc
        return result

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for scope in SkillScope:
            root = self._scope_root(scope)
            for child in self._children(scope):
                path = child / "SKILL.md"
                if not path.is_file() or _is_reparse(path):
                    raise SkillCatalogError("Skill directory must contain a regular SKILL.md")
                try:
                    payload = path.read_bytes()
                except OSError as exc:
                    raise SkillCatalogError("SKILL.md could not be fingerprinted") from exc
                digest.update(scope.value.encode())
                digest.update(b"\0")
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(b"\0")
                digest.update(str(len(payload)).encode())
                digest.update(hashlib.sha256(payload).digest())
        return digest.hexdigest()

    def build_snapshot(self, previous: CatalogSnapshot | None = None) -> CatalogSnapshot:
        fingerprint = self.fingerprint()
        if previous is not None and previous.fingerprint == fingerprint:
            return previous
        definitions: dict[tuple[SkillScope, str], SkillDefinition] = {}
        for scope in SkillScope:
            for child in self._children(scope):
                try:
                    definition = parse_skill(child / "SKILL.md", scope)
                except SkillParseError as exc:
                    raise SkillCatalogError(f"Invalid Skill in {scope.value}: {exc}") from exc
                identity = (scope, definition.name)
                if identity in definitions:
                    raise SkillCatalogError("Duplicate Skill name within one scope")
                definitions[identity] = definition
        effective: dict[str, SkillDefinition] = {}
        for scope in (SkillScope.PROJECT, SkillScope.USER):
            for (_, name), definition in sorted(definitions.items(), key=lambda item: item[0][1]):
                if definition.scope is scope:
                    effective.setdefault(name, definition)
        return CatalogSnapshot(
            (previous.generation + 1) if previous else 1,
            fingerprint,
            effective,
            definitions,
            Bm25Index(definitions.values()),
        )

    @staticmethod
    def resolve(
        snapshot: CatalogSnapshot, name_or_qualified: str, *, user: bool, model: bool
    ) -> SkillDefinition:
        normalized = name_or_qualified.strip()
        if ":" in normalized:
            raw_scope, name = normalized.split(":", 1)
            try:
                definition = snapshot.by_identity[(SkillScope(raw_scope), name)]
            except (ValueError, KeyError) as exc:
                raise SkillCatalogError("Unknown Skill") from exc
        else:
            try:
                definition = snapshot.effective[normalized]
            except KeyError as exc:
                raise SkillCatalogError("Unknown Skill") from exc
        if user and not definition.user_invocable:
            raise SkillCatalogError("Skill cannot be invoked locally")
        if model and (":" in normalized or not definition.model_invocable):
            raise SkillCatalogError("Skill cannot be invoked by the model")
        return definition

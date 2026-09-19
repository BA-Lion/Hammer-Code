from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hammer_code.skill.catalog import SkillCatalog, SkillCatalogError
from hammer_code.skill.models import SkillScope
from hammer_code.skill.parser import (
    SkillInvocationError,
    SkillParseError,
    parse_skill,
    render_skill,
)
from hammer_code.skill.retrieval import Bm25Index, tokenize


def _write_skill(
    root: Path,
    scope: str,
    folder: str,
    *,
    name: str,
    description: str,
    body: str = "Do {{arguments}} in {{skill_dir}}.",
    extra: str = "",
) -> Path:
    path = root / ".hammer-code" / "skill" / scope / folder / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f'---\nname: {name}\ndescription: "{description}"\n{extra}---\n{body}\n',
        encoding="utf-8",
    )
    return path


def test_parser_uses_safe_runtime_path_and_does_not_expand_twice(tmp_path: Path) -> None:
    path = _write_skill(
        tmp_path,
        "project",
        "cjk",
        name="cjk-help",
        description="处理中文文档",
        extra="arguments-required: true\n",
    )

    definition = parse_skill(path, SkillScope.PROJECT)

    assert definition.source.value == "imported"
    assert definition.skill_dir == path.parent.resolve()
    assert "{{skill_dir}}" in render_skill(definition, "{{skill_dir}}")
    with pytest.raises(SkillInvocationError):
        render_skill(definition, "")


def test_parser_accepts_hash_inside_a_quoted_string(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "project", "hash", name="hash-help", description="value # tag")

    assert parse_skill(path, SkillScope.PROJECT).description == "value # tag"


@pytest.mark.parametrize(
    "line",
    ["unknown: value", "description: one # comment", "allowed-tools: [a, a]", "name: Bad"],
)
def test_parser_rejects_unsupported_or_ambiguous_front_matter(tmp_path: Path, line: str) -> None:
    path = _write_skill(tmp_path, "project", "bad", name="valid", description="valid")
    source = path.read_text(encoding="utf-8").replace('description: "valid"', line)
    path.write_text(source, encoding="utf-8")

    with pytest.raises(SkillParseError):
        parse_skill(path, SkillScope.PROJECT)


def test_catalog_shadows_user_skill_but_retains_qualified_identity(tmp_path: Path) -> None:
    _write_skill(tmp_path, "project", "project-copy", name="review", description="project review")
    _write_skill(tmp_path, "user", "user-copy", name="review", description="user review")
    catalog = SkillCatalog(tmp_path)

    snapshot = catalog.build_snapshot()

    assert snapshot.effective["review"].scope is SkillScope.PROJECT
    assert snapshot.by_identity[(SkillScope.USER, "review")].description == "user review"
    assert catalog.resolve(snapshot, "user:review", user=True, model=False).scope is SkillScope.USER
    with pytest.raises(SkillCatalogError):
        catalog.resolve(snapshot, "user:review", user=False, model=True)


def test_catalog_fails_closed_for_invalid_child(tmp_path: Path) -> None:
    root = tmp_path / ".hammer-code" / "skill" / "project"
    root.mkdir(parents=True)
    (root / "not-a-directory.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SkillCatalogError):
        SkillCatalog(tmp_path).build_snapshot()


def test_metadata_retrieval_handles_identifier_and_cjk_tokens(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "project",
        "commit",
        name="commit-review",
        description="Review Git commits and changed files",
        extra="when-to-use: 提交代码前审查\n",
    )
    _write_skill(
        tmp_path,
        "project",
        "translate",
        name="translate-doc",
        description="Translate documents",
        extra="when-to-use: 翻译中文文档\n",
    )
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    assert snapshot.index is not None

    english = snapshot.index.retrieve(
        "please CommitReview this change",
        snapshot.effective.values(),
        top_k=3,
        relative_floor=0.1,
        coverage_floor=0.0,
    )
    chinese = snapshot.index.retrieve(
        "翻译文档",
        snapshot.effective.values(),
        top_k=3,
        relative_floor=0.1,
        coverage_floor=0.0,
    )

    assert english[0].ref.name == "commit-review"
    assert chinese[0].ref.name == "translate-doc"
    assert {"翻", "译", "翻译"}.issubset(tokenize("翻译"))
    assert isinstance(Bm25Index(snapshot.effective.values()), Bm25Index)


def test_normalized_merge_score_uses_candidate_self_score(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "project",
        "review",
        name="review",
        description="Review Python changes before release",
    )
    snapshot = SkillCatalog(tmp_path).build_snapshot()
    assert snapshot.index is not None
    existing = snapshot.effective["review"]
    candidate = replace(
        existing,
        name="review-guide",
        description="Review Python changes before release",
    )

    matches = snapshot.index.normalized_merge_candidates(
        "review Python changes before release", candidate, (existing,), 1
    )

    assert matches[0].ref == existing.ref
    assert 0 < matches[0].relative_score <= 1

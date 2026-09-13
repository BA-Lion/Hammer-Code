from __future__ import annotations

import pytest

from hammer_code.project_instructions import ProjectInstructionError, ProjectInstructionLoader


def test_project_instruction_loader_expands_bounded_relative_include(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "hammer-code.md").write_text("root\n@include(nested/rules.md)\n", encoding="utf-8")
    (tmp_path / "nested" / "rules.md").write_text("rules", encoding="utf-8")

    assert ProjectInstructionLoader(tmp_path).load() == "root\nrules"


def test_project_instruction_loader_rejects_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "hammer-code.md").write_text("@include(../outside.md)", encoding="utf-8")

    with pytest.raises(ProjectInstructionError):
        ProjectInstructionLoader(tmp_path).load()

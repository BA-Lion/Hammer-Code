from pathlib import Path

import pytest

from hammer_code.prompts import (
    PromptBuilder,
    PromptRuntimeContext,
    PromptSection,
    PromptStage,
    build_system_prompt,
    default_static_sections,
    runtime_environment_section,
)


def _section(name: str, stage: PromptStage, priority: int, content: str) -> PromptSection:
    return PromptSection(name, stage, priority, content)


def _context(tmp_path: Path, tools: tuple[str, ...] = ()) -> PromptRuntimeContext:
    return PromptRuntimeContext(
        cwd=tmp_path / "cwd",
        project_root=tmp_path / "project",
        platform="win32",
        shell_backend=None,
        permission_mode=None,
        enabled_tools=tools,
    )


def test_builder_sorts_deterministically_and_normalizes_newlines() -> None:
    sections = (
        _section("later", PromptStage.SYSTEM_RULES, 1, "\n later content \n"),
        _section("zeta", PromptStage.IDENTITY, 0, " zeta content "),
        _section("alpha", PromptStage.IDENTITY, 0, "alpha content"),
    )
    builder = PromptBuilder(reversed(sections))
    assert builder.section_names() == ("alpha", "zeta", "later")
    assert builder.build() == "alpha content\n\nzeta content\n\nlater content\n"
    assert PromptBuilder(sections).build() == builder.build()


def test_builder_rejects_duplicate_or_empty_names_and_ignores_blank_content() -> None:
    with pytest.raises(ValueError, match="empty"):
        _section("  ", PromptStage.IDENTITY, 0, "content")
    repeated = _section("same", PromptStage.IDENTITY, 0, "content")
    with pytest.raises(ValueError, match="unique"):
        PromptBuilder((repeated, repeated))
    assert PromptBuilder((_section("empty", PromptStage.IDENTITY, 0, " \n "),)).build() == ""
    assert (
        PromptBuilder((_section("empty", PromptStage.IDENTITY, 0, " \n "),)).section_names() == ()
    )


def test_runtime_context_is_immutable_and_canonicalizes_tools(tmp_path: Path) -> None:
    context = _context(tmp_path, ("write", "read"))
    section = _section("identity", PromptStage.IDENTITY, 0, "content")
    assert context.enabled_tools == ("read", "write")
    with pytest.raises(ValueError, match="unique"):
        _context(tmp_path, ("read", "read"))
    with pytest.raises(ValueError, match="empty"):
        _context(tmp_path, (" ",))
    with pytest.raises(AttributeError):
        context.platform = "linux"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        section.name = "changed"  # type: ignore[misc]


def test_default_static_sections_have_required_semantics_and_word_budget() -> None:
    sections = {section.name: section.content for section in default_static_sections()}
    assert set(sections) == {
        "identity",
        "system_rules",
        "doing_tasks",
        "executing_actions",
        "using_tools",
        "tone_style",
        "text_output",
    }
    assert "Hammer Code" in sections["identity"]
    assert "honest" in sections["system_rules"]
    assert "Inspect" in sections["doing_tasks"]
    assert "preserve user work" in sections["executing_actions"]
    assert "actually provided" in sections["using_tools"]
    assert "clear" in sections["tone_style"]
    assert "Lead with the result" in sections["text_output"]
    assert all(len(content.split()) <= 120 for content in sections.values())
    assert sum(len(content.split()) for content in sections.values()) <= 600


def test_runtime_section_matches_no_tool_capability_and_omits_environment_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPT_TEST_API_KEY", "secret-value-must-not-appear")
    prompt = build_system_prompt(_context(tmp_path))
    assert "Shell backend: none" in prompt
    assert "Permission mode: none" in prompt
    assert "Enabled tools: none" in prompt
    assert "cannot read files, execute commands, or modify files" in prompt
    assert "secret-value-must-not-appear" not in prompt
    assert "PROMPT_TEST_API_KEY" not in prompt


def test_runtime_section_sorts_enabled_tools_and_is_always_last(tmp_path: Path) -> None:
    context = PromptRuntimeContext(
        cwd=tmp_path / "cwd",
        project_root=tmp_path / "project",
        platform="win32",
        shell_backend="powershell",
        permission_mode="ask",
        enabled_tools=("write_file", "read_file"),
    )
    section = runtime_environment_section(context)
    prompt = build_system_prompt(context)
    assert "Enabled tools: read_file, write_file" in section.content
    assert "Shell backend: powershell" in section.content
    assert "Permission mode: ask" in section.content
    assert "schema" not in section.content.lower()
    assert prompt.endswith(section.content + "\n")
    assert (
        PromptBuilder((*default_static_sections(), section)).section_names()[-1]
        == "runtime_environment"
    )

from __future__ import annotations

from pathlib import Path

import pytest

from hammer_code.config import SubagentConfig
from hammer_code.errors import ConfigurationError
from hammer_code.subagent.parser import SubagentParseError, parse_subagent
from hammer_code.subagent.repository import SubagentRepository


def _write_definition(
    root: Path,
    name: str = "reviewer",
    *,
    description: str = "Review a bounded change.",
    body: str = "Return concise findings.",
    extra: str = "",
) -> Path:
    path = root / ".hammer-code" / "subagents" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_parser_applies_defaults_and_keeps_definition_immutable(tmp_path: Path) -> None:
    path = _write_definition(
        tmp_path,
        extra="when-to-use: Review before final validation.\nallowed-tools: [read_file, grep]\n"
        "disallowed-tools: [shell]\nmax-iterations: 7\n",
    )

    definition = parse_subagent(path, SubagentConfig())

    assert definition.name == "reviewer"
    assert definition.context.value == "isolated"
    assert definition.execution.value == "inline"
    assert definition.allowed_tools == ("read_file", "grep")
    assert definition.disallowed_tools == ("shell",)
    assert definition.max_iterations == 7
    assert definition.source_path == path.resolve()


@pytest.mark.parametrize(
    ("name", "extra", "body"),
    [
        ("bad", "unknown: value\n", "Body."),
        ("bad", "allowed-tools: [read_file, read_file]\n", "Body."),
        ("bad", "allowed-tools: [read_file]\ndisallowed-tools: [read_file]\n", "Body."),
        ("bad", "max-iterations: 51\n", "Body."),
        ("bad", "", ""),
    ],
)
def test_parser_rejects_ambiguous_or_out_of_bounds_definitions(
    tmp_path: Path, name: str, extra: str, body: str
) -> None:
    path = _write_definition(tmp_path, name, extra=extra, body=body)

    with pytest.raises(SubagentParseError):
        parse_subagent(path, SubagentConfig())


def test_parser_requires_front_matter_name_to_match_filename(tmp_path: Path) -> None:
    path = _write_definition(tmp_path, "filename")
    path.write_text(path.read_text(encoding="utf-8").replace("name: filename", "name: content"))

    with pytest.raises(SubagentParseError):
        parse_subagent(path, SubagentConfig())


@pytest.mark.asyncio
async def test_repository_uses_empty_catalog_when_directory_is_absent(tmp_path: Path) -> None:
    repository = SubagentRepository(tmp_path, SubagentConfig())

    snapshot = await repository.initialize()

    assert snapshot.generation == 1
    assert not snapshot.definitions
    assert await repository.snapshot_for_request() is snapshot


@pytest.mark.asyncio
async def test_repository_strict_initial_load_and_last_known_good_reload(tmp_path: Path) -> None:
    warnings: list[str] = []
    repository = SubagentRepository(tmp_path, SubagentConfig(), warnings.append)
    path = _write_definition(tmp_path)
    initial = await repository.initialize()
    assert initial.definitions["reviewer"].prompt == "Return concise findings."

    path.write_text("---\nname: reviewer\n", encoding="utf-8")
    preserved = await repository.snapshot_for_request()
    again = await repository.snapshot_for_request()

    assert preserved is initial
    assert again is initial
    assert len(warnings) == 1
    assert "Return concise findings." not in warnings[0]

    _write_definition(tmp_path, body="Updated.")
    updated = await repository.snapshot_for_request()
    assert updated.generation == 2
    assert updated.definitions["reviewer"].prompt == "Updated."


@pytest.mark.asyncio
async def test_repository_reloads_delete_and_rejects_invalid_initial_catalog(
    tmp_path: Path,
) -> None:
    _write_definition(tmp_path)
    repository = SubagentRepository(tmp_path, SubagentConfig())
    initial = await repository.initialize()
    (tmp_path / ".hammer-code" / "subagents" / "reviewer.md").unlink()
    emptied = await repository.snapshot_for_request()
    assert emptied.generation == initial.generation + 1
    assert not emptied.definitions

    _write_definition(tmp_path, extra="unknown: value\n")
    with pytest.raises(ConfigurationError):
        await SubagentRepository(tmp_path, SubagentConfig()).initialize()

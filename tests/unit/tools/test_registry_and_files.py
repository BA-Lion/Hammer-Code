from pathlib import Path

import pytest

from hammer_code.tools.base import ToolExecutionContext
from hammer_code.tools.builtin.files import (
    CreateFileInput,
    CreateFileTool,
    EditFileInput,
    EditFileTool,
    ReadFileInput,
    ReadFileTool,
)
from hammer_code.tools.builtin.search import GlobTool, GrepTool
from hammer_code.tools.builtin.shell import ShellTool
from hammer_code.tools.registry import ToolRegistry


def _context(root: Path) -> ToolExecutionContext:
    return ToolExecutionContext(root, root, root / ".hammer-code" / "runtime", {})


def test_registry_exposes_stable_object_schemas_and_honors_disabled() -> None:
    registry = ToolRegistry(("shell",))
    for tool in (
        ReadFileTool(),
        EditFileTool(),
        CreateFileTool(),
        GrepTool(),
        GlobTool(),
        ShellTool(),
    ):
        registry.register(tool)
    definitions = {definition.name: definition for definition in registry.definitions()}
    assert set(definitions) == {"read_file", "edit_file", "create_file", "grep", "glob"}
    assert definitions["read_file"].parameters["type"] == "object"
    assert definitions["read_file"].parameters["additionalProperties"] is False


@pytest.mark.asyncio
async def test_file_tools_enforce_workspace_and_unique_replacement(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")
    context = _context(tmp_path)
    read = await ReadFileTool().execute(context, ReadFileInput(path="example.txt"))
    assert "1: one" in read.content
    edited = await EditFileTool().execute(
        context, EditFileInput(path="example.txt", old_text="two", new_text="three")
    )
    assert not edited.is_error
    assert target.read_text(encoding="utf-8") == "one\nthree\n"
    duplicate = await EditFileTool().execute(
        context, EditFileInput(path="example.txt", old_text="missing", new_text="no")
    )
    assert duplicate.is_error
    created = await CreateFileTool().execute(
        context, CreateFileInput(path="new.txt", content="new")
    )
    assert not created.is_error
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new"
    outside = await CreateFileTool().execute(
        context, CreateFileInput(path="../outside.txt", content="no")
    )
    assert outside.is_error

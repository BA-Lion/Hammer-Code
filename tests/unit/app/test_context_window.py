from typing import cast

from hammer_code.app.context_window import ContextWindow
from hammer_code.domain.events import ToolDefinition


def test_snapshot_joins_prompts_and_copies_tool_parameters() -> None:
    window = ContextWindow(" base \n")
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    snapshot = window.snapshot(
        mcp_prompt="\n# MCP\nready\n",
        messages=(),
        tools=(ToolDefinition("tool", "description", parameters),),
    )
    assert snapshot.system_prompt == "base\n\n# MCP\nready\n"
    parameters["properties"]["x"]["type"] = "number"
    copied = cast(dict[str, object], snapshot.tools[0].parameters["properties"])
    copied_x = cast(dict[str, object], copied["x"])
    assert copied_x["type"] == "string"

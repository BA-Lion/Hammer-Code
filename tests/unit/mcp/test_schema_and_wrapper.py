import pytest
from pydantic import ValidationError

from hammer_code.mcp.schema import input_model_from_schema
from hammer_code.mcp.types import McpCallResult
from hammer_code.mcp.wrapper import McpToolWrapper, public_tool_name
from hammer_code.tools.base import ToolExecutionContext


def test_schema_converter_validates_nested_values_refs_and_preserves_original_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 2},
            "count": {"type": "integer", "minimum": 1},
            "nested": {"$ref": "#/$defs/Nested"},
            "optional": {"type": ["string", "null"]},
        },
        "required": ["name", "count", "nested"],
        "additionalProperties": False,
        "$defs": {
            "Nested": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            }
        },
    }
    model, original = input_model_from_schema(schema, "Example")
    value = model.model_validate(
        {"name": "ok", "count": 1, "nested": {"enabled": True}, "optional": None}
    )
    assert value.model_dump(exclude_unset=True)["nested"] == {"enabled": True}
    original["properties"]["name"]["minLength"] = 999  # type: ignore[index]
    assert schema["properties"]["name"]["minLength"] == 2  # type: ignore[index]
    with pytest.raises(ValidationError):
        model.model_validate({"name": "x", "count": 0, "nested": {"enabled": "yes"}})


def test_schema_converter_rejects_unsupported_and_recursive_references() -> None:
    with pytest.raises(Exception, match="unsupported MCP schema"):
        input_model_from_schema({"type": "object", "allOf": []}, "Bad")
    with pytest.raises(Exception, match="recursive"):
        input_model_from_schema(
            {
                "type": "object",
                "properties": {"node": {"$ref": "#/$defs/Node"}},
                "$defs": {"Node": {"$ref": "#/$defs/Node"}},
            },
            "Recursive",
        )


class _Client:
    async def execute(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        assert name == "original"
        assert arguments == {"value": "ok"}
        return McpCallResult("done", False)


@pytest.mark.asyncio
async def test_wrapper_is_deferred_command_with_stable_name_and_original_definition() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    wrapper = McpToolWrapper(_Client(), "Example.Server", "original", "does work", schema)  # type: ignore[arg-type]
    assert wrapper.should_defer
    assert wrapper.name == public_tool_name("Example.Server", "original")
    definition = wrapper.definition()
    definition.parameters["mutated"] = True
    assert "mutated" not in wrapper.definition().parameters
    result = await wrapper.execute(
        ToolExecutionContext.__new__(ToolExecutionContext),
        wrapper.input_model.model_validate({"value": "ok"}),
    )
    assert result.content == "done"

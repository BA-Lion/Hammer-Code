"""A deliberately bounded JSON Schema-to-Pydantic converter for MCP tools."""

from __future__ import annotations

import copy
import re
from functools import reduce
from operator import or_
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    create_model,
)

from hammer_code.errors import McpSchemaError

_UNSUPPORTED = {
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "dependentSchemas",
}


def input_model_from_schema(schema: object, name: str) -> tuple[type[BaseModel], dict[str, object]]:
    """Return strict local validation and an untouched copy for model projection."""
    if not isinstance(schema, dict):
        raise McpSchemaError("MCP tool input schema must be an object")
    raw = copy.deepcopy(schema)
    if not schema:
        return _model(name, {}, "allow"), raw
    if schema.get("type") != "object":
        raise McpSchemaError("MCP tool input schema must have type object")
    converter = _Converter(schema.get("$defs", {}))
    return converter.object_model(schema, name), raw


class _Converter:
    def __init__(self, definitions: object) -> None:
        if not isinstance(definitions, dict):
            raise McpSchemaError("$defs must be an object")
        self.definitions: dict[str, object] = definitions
        self._resolving: set[str] = set()

    def object_model(self, schema: dict[str, object], name: str) -> type[BaseModel]:
        self._reject_unsupported(schema)
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise McpSchemaError("object properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise McpSchemaError("required must be an array of property names")
        additional = schema.get("additionalProperties", True)
        if isinstance(additional, dict):
            raise McpSchemaError("schema additionalProperties is not supported")
        if not isinstance(additional, bool):
            raise McpSchemaError("additionalProperties must be boolean")
        fields: dict[str, tuple[Any, Any]] = {}
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str) or not isinstance(property_schema, dict):
                raise McpSchemaError("property schemas must be objects")
            annotation = self.annotation(property_schema, f"{name}_{property_name}")
            if property_name in required:
                default: Any = Field(..., **cast(Any, self.constraints(property_schema)))
            elif "default" in property_schema:
                default = Field(
                    property_schema["default"], **cast(Any, self.constraints(property_schema))
                )
            else:
                default = Field(None, **cast(Any, self.constraints(property_schema)))
            fields[property_name] = (annotation, default)
        return _model(name, fields, "forbid" if additional is False else "allow")

    def annotation(self, schema: dict[str, object], name: str) -> object:
        self._reject_unsupported(schema)
        if "$ref" in schema:
            return self._reference(schema["$ref"], name)
        if "const" in schema:
            return Literal[schema["const"]]  # type: ignore[valid-type]
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise McpSchemaError("enum must be a non-empty array")
            return Literal[tuple(values)]  # type: ignore[valid-type]
        variants: list[object] = []
        for key in ("anyOf", "oneOf"):
            if key in schema:
                choices = schema[key]
                if not isinstance(choices, list) or not choices:
                    raise McpSchemaError(f"{key} must be a non-empty array")
                variants.extend(
                    self.annotation(choice, f"{name}_{index}")
                    for index, choice in enumerate(choices)
                    if isinstance(choice, dict)
                )
                if len(variants) != len(choices):
                    raise McpSchemaError(f"{key} entries must be objects")
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            variants.extend(self._type_annotation(item, schema, name) for item in schema_type)
        elif isinstance(schema_type, str):
            variants.append(self._type_annotation(schema_type, schema, name))
        elif schema_type is None and variants:
            pass
        else:
            raise McpSchemaError("schema type is required")
        if not variants:
            raise McpSchemaError("schema has no supported type")
        return variants[0] if len(variants) == 1 else reduce(or_, variants)

    def _type_annotation(self, schema_type: object, schema: dict[str, object], name: str) -> object:
        match schema_type:
            case "string":
                return StrictStr
            case "integer":
                return StrictInt
            case "number":
                return float | int
            case "boolean":
                return StrictBool
            case "null":
                return type(None)
            case "array":
                items = schema.get("items", {})
                if not isinstance(items, dict):
                    raise McpSchemaError("array items must be an object")
                return list[self.annotation(items, f"{name}_item")]
            case "object":
                return self.object_model(schema, name)
            case _:
                raise McpSchemaError(f"unsupported schema type: {schema_type}")

    def _reference(self, reference: object, name: str) -> object:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise McpSchemaError("only local #/$defs references are supported")
        definition_name = reference.removeprefix("#/$defs/")
        if "/" in definition_name or definition_name not in self.definitions:
            raise McpSchemaError("MCP schema reference does not exist")
        if definition_name in self._resolving:
            raise McpSchemaError("recursive MCP schema references are not supported")
        target = self.definitions[definition_name]
        if not isinstance(target, dict):
            raise McpSchemaError("MCP schema reference is not an object")
        self._resolving.add(definition_name)
        try:
            return self.annotation(target, _safe_name(definition_name, name))
        finally:
            self._resolving.remove(definition_name)

    @staticmethod
    def _reject_unsupported(schema: dict[str, object]) -> None:
        unsupported = _UNSUPPORTED.intersection(schema)
        if unsupported:
            keywords = ", ".join(sorted(unsupported))
            raise McpSchemaError(f"unsupported MCP schema keywords: {keywords}")

    @staticmethod
    def constraints(schema: dict[str, object]) -> dict[str, Any]:
        mapping = {
            "minLength": "min_length",
            "maxLength": "max_length",
            "pattern": "pattern",
            "minimum": "ge",
            "maximum": "le",
            "exclusiveMinimum": "gt",
            "exclusiveMaximum": "lt",
            "multipleOf": "multiple_of",
            "minItems": "min_length",
            "maxItems": "max_length",
        }
        return {target: schema[source] for source, target in mapping.items() if source in schema}


def _model(
    name: str,
    fields: dict[str, tuple[Any, Any]],
    extra: Literal["allow", "forbid"],
) -> type[BaseModel]:
    return create_model(
        _safe_name(name, "McpInput"),
        __config__=ConfigDict(extra=extra, frozen=True, strict=True),
        **cast(Any, fields),
    )


def _safe_name(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return normalized or fallback

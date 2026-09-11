import pytest
from pydantic import BaseModel, ConfigDict

from hammer_code.errors import ToolError
from hammer_code.tools.base import (
    ConcurrencyPolicy,
    Tool,
    ToolCategory,
    ToolExecutionContext,
    ToolExecutionResult,
)
from hammer_code.tools.registry import ToolRegistry


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str


class _Tool(Tool):
    input_model = _Input
    category = ToolCategory.READ
    concurrency_policy = ConcurrencyPolicy.SERIAL

    def __init__(self, name: str, *, deferred: bool = False) -> None:
        self.name = name
        self.description = f"Description for {name}"
        self.should_defer = deferred

    async def execute(
        self, context: ToolExecutionContext, arguments: BaseModel
    ) -> ToolExecutionResult:
        return ToolExecutionResult("ok")


def test_deferred_tools_require_discovery_and_clear_resets_exposure() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("always"))
    registry.register(_Tool("deferred", deferred=True), owner="mcp-a")

    assert registry.exposed_names() == ("always",)
    assert tuple(tool.name for tool in registry.deferred_tools()) == ("deferred",)
    assert not registry.exposed("deferred")
    registry.discover("deferred")
    assert registry.exposed_names() == ("always", "deferred")
    registry.clear_discovered()
    assert not registry.exposed("deferred")


def test_owner_replacement_is_atomic_and_removes_discovery() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("local"))
    registry.replace_owner("mcp-a", (_Tool("old", deferred=True),))
    registry.discover("old")

    with pytest.raises(ValueError):
        registry.replace_owner("mcp-a", (_Tool("local", deferred=True),))
    assert registry.exposed("old")

    registry.replace_owner("mcp-a", (_Tool("new", deferred=True),))
    assert registry.get("old") is None
    assert not registry.exposed("new")
    registry.remove_owner("mcp-a")
    assert registry.get("new") is None


def test_discover_rejects_non_deferred_or_disabled_tools() -> None:
    registry = ToolRegistry(("disabled",))
    registry.register(_Tool("ordinary"))
    registry.register(_Tool("disabled", deferred=True))
    with pytest.raises(ToolError):
        registry.discover("ordinary")
    with pytest.raises(ToolError):
        registry.discover("disabled")

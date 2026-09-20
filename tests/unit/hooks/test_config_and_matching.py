from __future__ import annotations

from pathlib import Path

import pytest

from hammer_code.errors import ConfigurationError
from hammer_code.hooks.config import instantiate_hooks, load_hook_definitions
from hammer_code.hooks.models import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionMode,
    ConditionOperator,
    Hook,
    HookContext,
    LifecycleEvent,
    expand_template,
    matches_condition,
)


def test_missing_hook_file_is_disabled(tmp_path: Path) -> None:
    assert load_hook_definitions(tmp_path / "hooks.toml") == ()


def test_strict_config_loads_and_runtime_instances_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "hooks.toml"
    path.write_text(
        """[[hooks]]
id = "turn-note"
event = "turn_start"
once = true
[hooks.action]
type = "prompt"
prompt = "Handle $MESSAGE"
""",
        encoding="utf-8",
    )
    definitions = load_hook_definitions(path)
    first, second = instantiate_hooks(definitions), instantiate_hooks(definitions)
    first[0].executed = True
    assert first[0].id == "turn-note"
    assert not second[0].executed


@pytest.mark.parametrize(
    "text",
    (
        "\n".join(
            (
                "[[hooks]]",
                "id = 'bad'",
                "event = 'turn_start'",
                "executed = true",
                "[hooks.action]",
                "type = 'prompt'",
                "prompt = 'x'",
            )
        ),
        "\n".join(
            (
                "[[hooks]]",
                "id = 'one'",
                "event = 'turn_start'",
                "[hooks.action]",
                "type = 'agent'",
                "prompt = 'x'",
            )
        ),
        "\n".join(
            (
                "[[hooks]]",
                "id = 'one'",
                "event = 'turn_start'",
                "[hooks.action]",
                "type = 'http'",
                "url = 'https://x.test'",
                "[hooks.action.headers]",
                "Authorization = 'secret'",
            )
        ),
    ),
)
def test_invalid_config_fails_closed(tmp_path: Path, text: str) -> None:
    path = tmp_path / "hooks.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid Hook configuration"):
        load_hook_definitions(path)


@pytest.mark.parametrize(
    ("operator", "expected"),
    (
        (ConditionOperator.EQ, True),
        (ConditionOperator.NE, False),
        (ConditionOperator.REGEX, True),
        (ConditionOperator.GLOB, True),
    ),
)
def test_condition_operators(operator: ConditionOperator, expected: bool) -> None:
    value = "shell" if operator is not ConditionOperator.REGEX else "ell"
    if operator is ConditionOperator.GLOB:
        value = "sh*"
    group = ConditionGroup(ConditionMode.ALL, (Condition("tool_name", operator, value),))
    assert matches_condition(group, HookContext(tool_name="shell")) is expected


def test_missing_condition_value_does_not_match_not_equal() -> None:
    group = ConditionGroup(
        ConditionMode.ALL,
        (Condition("tool_args.path", ConditionOperator.NE, "blocked"),),
    )
    assert not matches_condition(group, HookContext())


def test_template_expansion_is_exact_and_preserves_powershell_variables() -> None:
    context = HookContext(
        event_name="turn_start", message="hello", tool_args={"count": 2, "enabled": True}
    )
    assert expand_template(
        "$EVENT $MESSAGE $TOOL_ARGS.count $TOOL_ARGS.enabled $env:PATH", context
    ) == ("turn_start hello 2 true $env:PATH")
    assert expand_template("$EVENTUAL", context) == "$EVENTUAL"
    with pytest.raises(ValueError):
        expand_template("$TOOL_ARGS.missing", context)


def test_hook_model_copies_mutable_mappings() -> None:
    headers = {"X-Mode": "test"}
    arguments = {"path": "a.txt"}
    action = Action(ActionType.HTTP, url="https://example.test", headers=headers)
    context = HookContext(tool_args=arguments)
    headers["X-Mode"] = "changed"
    arguments["path"] = "changed"
    assert action.headers["X-Mode"] == "test"
    assert context.tool_args["path"] == "a.txt"


def test_hook_declaration_can_reject_without_action_output() -> None:
    hook = Hook(
        id="deny-shell",
        event=LifecycleEvent.PRE_TOOL_USE,
        action=Action(ActionType.PROMPT, prompt="Denied"),
        reject=True,
    )
    assert hook.reject

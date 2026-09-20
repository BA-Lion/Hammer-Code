"""Stable public Hook configuration and runtime API."""

from hammer_code.hooks.config import (
    HookDefinitionConfig,
    instantiate_hooks,
    load_hook_definitions,
)
from hammer_code.hooks.models import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionMode,
    ConditionOperator,
    Hook,
    HookContext,
    HookDispatchResult,
    LifecycleEvent,
    PromptBatch,
    expand_template,
    matches_condition,
)

__all__ = [
    "Action",
    "ActionType",
    "Condition",
    "ConditionGroup",
    "ConditionMode",
    "ConditionOperator",
    "Hook",
    "HookContext",
    "HookDefinitionConfig",
    "HookDispatchResult",
    "LifecycleEvent",
    "PromptBatch",
    "expand_template",
    "instantiate_hooks",
    "load_hook_definitions",
    "matches_condition",
]

"""Bounded, in-process Subagent definitions and execution support."""

from typing import TYPE_CHECKING

from hammer_code.subagent.models import (
    BackgroundTaskStatus,
    SubagentCatalogSnapshot,
    SubagentContext,
    SubagentDefinition,
    SubagentExecution,
    SubagentSource,
)
from hammer_code.subagent.repository import SubagentRepository, SubagentRepositoryError

if TYPE_CHECKING:
    from hammer_code.subagent.runner import SubagentRunner
    from hammer_code.subagent.service import BackgroundTaskManager, SubagentService
    from hammer_code.subagent.tool import RunSubagentTool, SubagentTaskTool

__all__ = [
    "BackgroundTaskStatus",
    "BackgroundTaskManager",
    "RunSubagentTool",
    "SubagentCatalogSnapshot",
    "SubagentContext",
    "SubagentDefinition",
    "SubagentExecution",
    "SubagentRepository",
    "SubagentRepositoryError",
    "SubagentRunner",
    "SubagentService",
    "SubagentSource",
    "SubagentTaskTool",
]


def __getattr__(name: str) -> object:
    """Delay runtime-heavy exports so tool-package initialization stays acyclic."""
    if name == "SubagentRunner":
        from hammer_code.subagent.runner import SubagentRunner

        return SubagentRunner
    if name in {"BackgroundTaskManager", "SubagentService"}:
        from hammer_code.subagent.service import BackgroundTaskManager, SubagentService

        return {"BackgroundTaskManager": BackgroundTaskManager, "SubagentService": SubagentService}[
            name
        ]
    if name in {"RunSubagentTool", "SubagentTaskTool"}:
        from hammer_code.subagent.tool import RunSubagentTool, SubagentTaskTool

        return {"RunSubagentTool": RunSubagentTool, "SubagentTaskTool": SubagentTaskTool}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

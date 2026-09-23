from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hammer_code.agent_team.mailbox import receive_messages
from hammer_code.agent_team.models import AgentTeamDefinition, TeamAgentTemplate
from hammer_code.agent_team.runtime import AgentTeamRun, AgentTeamRuntimeStore
from hammer_code.agent_team.usage import TeamUsageTracker
from hammer_code.domain.usage import TokenUsage, UsageStatus
from hammer_code.subagent.models import SubagentContext, SubagentExecution, SubagentWorkspace


def _definition() -> AgentTeamDefinition:
    leader = TeamAgentTemplate(
        "leader",
        "Coordinate work.",
        None,
        "Coordinate work.",
        SubagentContext.ISOLATED,
        SubagentWorkspace.SHARED,
        None,
        (),
    )
    return AgentTeamDefinition("test-team", "Test work.", leader, SubagentExecution.INLINE, ())


@pytest.mark.asyncio
async def test_shared_usage_reserves_once_and_final_is_not_downgraded() -> None:
    usage = TeamUsageTracker(100)
    first, second = await asyncio.gather(
        usage.reserve("first", 20, 80), usage.reserve("second", 20, 80)
    )
    assert sorted(value for value in (first, second) if value is not None) == [80]
    chosen = "first" if first is not None else "second"
    await usage.observe(chosen, TokenUsage(10, 20, status=UsageStatus.FINAL))
    await usage.observe(chosen, TokenUsage(1, 1, status=UsageStatus.PARTIAL))
    snapshot = await usage.snapshot()
    assert snapshot.reported_total == 30
    assert snapshot.final_requests == 1


@pytest.mark.asyncio
async def test_assignment_mailboxes_are_addressed_and_deleted_on_finish(tmp_path: Path) -> None:
    run = AgentTeamRun(
        _definition(),
        "Do the work.",
        AgentTeamRuntimeStore(tmp_path),
        2,
        10_000,
        SubagentExecution.INLINE,
    )
    await run.initialize()
    task = await run.create_task("Child", "Do a bounded part.")
    template = TeamAgentTemplate(
        "worker",
        "Do work.",
        None,
        "Do work.",
        SubagentContext.ISOLATED,
        SubagentWorkspace.SHARED,
        None,
        (),
    )
    assignment = await run.create_assignment(template, task.id)
    async with run.lock:
        run._send_locked(run.leader_assignment_id, assignment.id, "Please inspect this.")
    messages = receive_messages(run.path, assignment.id)
    assert messages[0].sender_assignment_id == run.leader_assignment_id
    await run.finish_assignment(assignment.id, "completed", "Done.")
    async with run.lock:
        with pytest.raises(ValueError):
            run._send_locked(run.leader_assignment_id, assignment.id, "late")
    await run.cleanup()

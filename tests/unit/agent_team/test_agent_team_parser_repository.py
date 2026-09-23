from __future__ import annotations

from pathlib import Path

import pytest

from hammer_code.agent_team.models import TeamAgentTemplate
from hammer_code.agent_team.parser import AgentTeamParseError, parse_team
from hammer_code.agent_team.repository import AgentTeamRepository
from hammer_code.subagent.models import SubagentContext, SubagentWorkspace


def _write_team(root: Path, name: str = "review-team", *, extra: str = "") -> Path:
    path = root / ".hammer-code" / "agent-teams" / name / "config.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: Review changes.\n"
        "leader-name: reviewer\n"
        "leader-description: Coordinate a review.\n"
        "leader-context: isolated\n"
        "leader-execution: inline\n"
        "leader-workspace: worktree\n"
        "members: []\n"
        f"{extra}"
        "---\n"
        "Coordinate bounded review work.\n",
        encoding="utf-8",
    )
    return path


def _template(name: str = "reviewer") -> TeamAgentTemplate:
    return TeamAgentTemplate(
        name,
        "Review changes.",
        None,
        "Review bounded changes.",
        SubagentContext.ISOLATED,
        SubagentWorkspace.WORKTREE,
        ("read_file",),
        (),
    )


def test_team_parser_requires_strict_filename_and_fields(tmp_path: Path) -> None:
    path = _write_team(tmp_path)
    value = parse_team(path)
    assert value.name == "review-team"
    assert value.leader.workspace is SubagentWorkspace.WORKTREE

    _write_team(tmp_path, extra="unknown: nope\n")
    with pytest.raises(AgentTeamParseError):
        parse_team(path)


@pytest.mark.asyncio
async def test_repository_create_suffix_member_and_last_known_good(tmp_path: Path) -> None:
    repository = AgentTeamRepository(tmp_path)
    assert not (await repository.initialize()).definitions
    first = await repository.create_team("Review changes.", _template())
    second = await repository.create_team("Review changes.", _template())
    assert first.name == "reviewer"
    assert second.name == "reviewer-2"
    await repository.create_member(first.name, _template("tester"))
    snapshot = await repository.snapshot_for_request()
    assert snapshot.definitions[first.name].members == ("tester",)

    config = tmp_path / ".hammer-code" / "agent-teams" / first.name / "config.md"
    config.write_text("---\nname: broken\n", encoding="utf-8")
    preserved = await repository.snapshot_for_request()
    assert preserved is snapshot

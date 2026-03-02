from __future__ import annotations

import asyncio

from ana.core.session import RuntimeSessionState
from ana.tools.fs import WriteFileTool
from ana.tools.policy import ToolPolicy


def test_write_to_skills_local_allowed_by_policy_and_tool(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"fs.read", "web.read", "skills.install_quarantine"})
    decision = policy.precheck(
        "write_file",
        {"path": "skills_local/demo/SKILL.md", "content": "# skill"},
        session_state=state,
    )
    assert decision.allowed

    tool = WriteFileTool(workspace=workspace, skills_local=skills_local)
    result = asyncio.run(tool.run({"path": "skills_local/demo/SKILL.md", "content": "# skill"}))
    assert result.ok
    assert (skills_local / "demo/SKILL.md").exists()


def test_write_outside_workspace_is_blocked_by_policy_and_tool(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)
    outside_file = tmp_path / "outside.txt"

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"fs.read", "web.read", "fs.write_workspace"})
    decision = policy.precheck(
        "write_file",
        {"path": str(outside_file), "content": "x"},
        session_state=state,
    )
    assert not decision.allowed

    tool = WriteFileTool(workspace=workspace, skills_local=skills_local)
    result = asyncio.run(tool.run({"path": str(outside_file), "content": "x"}))
    assert not result.ok
    assert result.data == "write outside workspace/skills_local is blocked"

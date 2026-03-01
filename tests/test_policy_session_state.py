from __future__ import annotations

import asyncio

from aha.core.session import RuntimeSessionState
from aha.tools.fs import WriteFileTool
from aha.tools.policy import ToolPolicy
from aha.tools.registry import ToolRegistry
from aha.tools.runner import ToolRunner


def test_precheck_requires_capability_grant(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState()
    decision = policy.precheck(
        "write_file",
        {"path": "notes.txt", "content": "hello"},
        session_state=state,
    )
    assert decision.allowed
    assert decision.requires_confirmation
    assert decision.temporary_capability == "fs.write_workspace"


def test_temporary_capability_is_consumed_after_single_run(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    registry = ToolRegistry()
    registry.register(WriteFileTool(workspace=workspace, skills_local=skills_local))
    runner = ToolRunner(registry=registry, policy=policy)
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    result = asyncio.run(
        runner.run(
            "write_file",
            {"path": "once.txt", "content": "x"},
            session_state=state,
            confirm=approve,
        )
    )
    assert result.ok
    assert "fs.write_workspace" not in state.capabilities

from __future__ import annotations

import json

from ana.core.session import RuntimeSessionState
from ana.tools.policy import ToolPolicy


def test_skill_constraint_blocks_tool_not_in_allowed_list(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(
        capabilities={"fs.read", "shell.safe"},
        active_skills={
            "alpha": {
                "name": "alpha",
                "allowed_tool_names": ["read_file"],
                "required_capabilities": ["fs.read"],
            }
        },
    )
    decision = policy.precheck("shell", {"cmd": "ls", "_source_skill": "alpha"}, session_state=state)
    assert not decision.allowed
    assert "skill constraint blocked tool" in decision.reason


def test_skill_constraint_blocks_capability_not_declared(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)
    (workspace / "a.txt").write_text("x", encoding="utf-8")

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(
        capabilities={"fs.read"},
        active_skills={
            "alpha": {
                "name": "alpha",
                "allowed_tool_names": ["read_file"],
                "required_capabilities": ["shell.safe"],
            }
        },
    )
    decision = policy.precheck("read_file", {"path": "a.txt", "_source_skill": "alpha"}, session_state=state)
    assert not decision.allowed
    assert "skill constraint blocked capability" in decision.reason


def test_skill_constraint_only_applies_with_resolved_active_skill(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)
    (workspace / "a.txt").write_text("x", encoding="utf-8")

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"fs.read"}, active_skills={})
    decision = policy.precheck("read_file", {"path": "a.txt", "_source_skill": "missing"}, session_state=state)
    assert decision.allowed


def test_skill_constraint_requires_source_skill_for_side_effecting_actions(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(
        capabilities={"shell.safe"},
        active_skills={
            "alpha": {
                "name": "alpha",
                "allowed_tool_names": ["shell"],
                "required_capabilities": ["shell.safe"],
            }
        },
    )
    decision = policy.precheck("shell", {"cmd": "ls"}, session_state=state)
    assert not decision.allowed
    assert "missing _source_skill attribution" in decision.reason


def test_skill_manager_management_actions_do_not_require_source_skill(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(
        capabilities={"skills.read", "skills.install_quarantine"},
        active_skills={
            "alpha": {
                "name": "alpha",
                "allowed_tool_names": ["read_file"],
                "required_capabilities": ["fs.read"],
            }
        },
    )

    listed = policy.precheck("skill_manager", {"action": "list"}, session_state=state)
    assert listed.allowed
    assert not listed.requires_confirmation

    disable = policy.precheck("skill_manager", {"action": "disable", "name": "alpha"}, session_state=state)
    assert disable.allowed
    assert disable.requires_confirmation


def test_skill_manager_enable_escalation_is_blocked(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    quarantine = skills_local / "quarantine" / "s1"
    workspace.mkdir()
    quarantine.mkdir(parents=True)

    (quarantine / "SKILL.md").write_text(
        """---
name: target
version: 1.0.0
allowed_tool_names:
  - read_file
  - shell
required_capabilities:
  - fs.read
  - shell.safe
---
""",
        encoding="utf-8",
    )
    (quarantine / "CHECK.json").write_text(
        json.dumps(
            {
                "skill_id": "s1",
                "status": "PASS",
                "hash": "dummy",
                "checked_at": "2026-03-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(
        capabilities={"skills.install_quarantine"},
        active_skills={
            "alpha": {
                "name": "alpha",
                "allowed_tool_names": ["skill_manager", "read_file"],
                "required_capabilities": ["skills.install_quarantine", "fs.read"],
            }
        },
    )
    decision = policy.precheck(
        "skill_manager",
        {"action": "enable", "skill_id": "s1", "_source_skill": "alpha"},
        session_state=state,
    )
    assert not decision.allowed
    assert "enable escalation blocked" in decision.reason

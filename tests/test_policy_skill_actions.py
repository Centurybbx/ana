from __future__ import annotations

from ana.core.session import RuntimeSessionState
from ana.tools.policy import ToolPolicy


def test_skill_manager_list_and_check_use_read_capability(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.read"})

    list_decision = policy.precheck("skill_manager", {"action": "list"}, session_state=state)
    assert list_decision.allowed
    assert not list_decision.requires_confirmation

    check_decision = policy.precheck(
        "skill_manager",
        {"action": "check", "skill_id": "s1"},
        session_state=state,
    )
    assert check_decision.allowed
    assert not check_decision.requires_confirmation


def test_skill_manager_generate_requires_confirmation(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.install_quarantine"})
    decision = policy.precheck(
        "skill_manager",
        {"action": "generate", "name": "demo", "content": "x"},
        session_state=state,
    )
    assert decision.allowed
    assert decision.requires_confirmation
    assert decision.temporary_capability is None


def test_skill_manager_enable_plan_contains_manifest_and_diff_summary(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    quarantine = skills_local / "quarantine" / "s1"
    enabled = skills_local / "enabled" / "demo"
    workspace.mkdir()
    quarantine.mkdir(parents=True)
    enabled.mkdir(parents=True)

    (quarantine / "SKILL.md").write_text(
        """---
name: demo
version: 0.2.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

new content
""",
        encoding="utf-8",
    )
    (enabled / "SKILL.md").write_text(
        """---
name: demo
version: 0.1.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

old content
""",
        encoding="utf-8",
    )

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.install_quarantine"})
    decision = policy.precheck("skill_manager", {"action": "enable", "skill_id": "s1"}, session_state=state)

    assert decision.allowed
    assert decision.requires_confirmation
    assert "Required capabilities" in decision.plan
    assert "Allowed tools" in decision.plan
    assert "Diff summary" in decision.plan
    assert "Risk delta" in decision.plan
    assert "Pin" in decision.plan


def test_skill_manager_diff_is_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.read"})
    decision = policy.precheck("skill_manager", {"action": "diff", "skill_id": "s1"}, session_state=state)
    assert decision.allowed
    assert not decision.requires_confirmation


def test_skill_manager_enable_plan_mentions_built_in_override(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    quarantine = skills_local / "quarantine" / "s1"
    built_in = workspace / "skills" / "demo"
    workspace.mkdir()
    quarantine.mkdir(parents=True)
    built_in.mkdir(parents=True)
    (built_in / "SKILL.md").write_text("# built-in", encoding="utf-8")
    (quarantine / "SKILL.md").write_text(
        """---
name: demo
version: 0.2.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
""",
        encoding="utf-8",
    )

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.install_quarantine"})
    decision = policy.precheck("skill_manager", {"action": "enable", "skill_id": "s1"}, session_state=state)
    assert decision.allowed
    assert "Built-in override: yes" in decision.plan


def test_skill_manager_invalid_and_missing_args_blocked(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    state = RuntimeSessionState(capabilities={"skills.install_quarantine", "skills.read"})

    invalid = policy.precheck("skill_manager", {"action": "unknown"}, session_state=state)
    assert not invalid.allowed

    missing_check_id = policy.precheck("skill_manager", {"action": "check"}, session_state=state)
    assert not missing_check_id.allowed

    missing_name = policy.precheck("skill_manager", {"action": "rollback"}, session_state=state)
    assert not missing_name.allowed

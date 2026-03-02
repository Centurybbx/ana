from __future__ import annotations

import asyncio
import json

from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse
from ana.tools.memory import MemoryStore
from ana.tools.skill_manager import SkillManagerTool


def _read_json(result_data: str) -> dict:
    return json.loads(result_data)


def test_generate_writes_quarantine_and_manifest(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)

    result = asyncio.run(tool.run({"action": "generate", "name": "demo", "content": "hello"}))
    assert result.ok

    payload = _read_json(result.data)
    skill_id = payload["skill_id"]
    skill_file = skills_local / "quarantine" / skill_id / "SKILL.md"
    assert skill_file.exists()
    assert skill_file.read_text(encoding="utf-8").startswith("---")
    assert not (skills_local / "enabled" / "demo" / "SKILL.md").exists()

    audit_lines = (skills_local / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["action"] == "generate"


def test_check_fails_when_manifest_missing(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    skill_id = "s1"
    skill_file = skills_local / "quarantine" / skill_id / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text("no frontmatter", encoding="utf-8")

    result = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert not result.ok
    payload = _read_json(result.data)
    assert payload["check_status"] == "FAIL"
    assert payload["manifest_ok"] is False


def test_enable_updates_enabled_and_lock(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(tool.run({"action": "generate", "name": "demo", "content": "hello"}))
    skill_id = _read_json(generated.data)["skill_id"]
    checked = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert checked.ok

    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert enabled.ok
    payload = _read_json(enabled.data)
    assert payload["state"] == "enabled"
    assert (skills_local / "enabled" / "demo" / "SKILL.md").exists()

    lock = json.loads((skills_local / "lock.json").read_text(encoding="utf-8"))
    assert "demo" in lock

    audit_entries = [json.loads(line) for line in (skills_local / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(entry["action"] == "enable" for entry in audit_entries)


def test_rollback_restores_previous_snapshot(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    v1 = """---
name: demo
version: 0.1.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
external_links_policy: deny
---

v1
"""
    v2 = """---
name: demo
version: 0.2.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
external_links_policy: deny
---

v2
    """
    skill_1 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v1})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_1})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_1})).ok
    skill_2 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v2})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_2})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_2})).ok

    rolled = asyncio.run(tool.run({"action": "rollback", "name": "demo"}))
    assert rolled.ok
    enabled_file = skills_local / "enabled" / "demo" / "SKILL.md"
    assert "v1" in enabled_file.read_text(encoding="utf-8")


def test_rollback_with_target_uses_requested_snapshot(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    v1 = """---
name: demo
version: 0.1.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
external_links_policy: deny
---

v1
"""
    v2 = """---
name: demo
version: 0.2.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
external_links_policy: deny
---

v2
"""
    v3 = """---
name: demo
version: 0.3.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
external_links_policy: deny
---

v3
"""
    skill_1 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v1})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_1})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_1})).ok
    skill_2 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v2})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_2})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_2})).ok
    skill_3 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v3})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_3})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_3})).ok

    snapshots = sorted((skills_local / "versions" / "demo").glob("*.SKILL.md"))
    assert len(snapshots) >= 2
    target_name = ""
    for snapshot in snapshots:
        if "v1" in snapshot.read_text(encoding="utf-8"):
            target_name = snapshot.name
            break
    assert target_name

    rolled = asyncio.run(tool.run({"action": "rollback", "name": "demo", "target": target_name}))
    assert rolled.ok
    enabled_file = skills_local / "enabled" / "demo" / "SKILL.md"
    assert "v1" in enabled_file.read_text(encoding="utf-8")


def test_install_alias_and_list_include_legacy_layout(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)

    installed = asyncio.run(tool.run({"action": "install", "name": "demo", "content": "hello"}))
    assert installed.ok
    assert "deprecated_action_install" in installed.warnings

    legacy_skill = skills_local / "legacy_demo" / "SKILL.md"
    legacy_skill.parent.mkdir(parents=True, exist_ok=True)
    legacy_skill.write_text("# legacy", encoding="utf-8")

    listed = asyncio.run(tool.run({"action": "list"}))
    payload = _read_json(listed.data)
    assert any(item["name"] == "legacy_demo" for item in payload["legacy"])
    assert payload["quarantine"]


def test_enable_rejects_failing_check(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    skill_id = "broken"
    skill_file = skills_local / "quarantine" / skill_id / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text("plain text", encoding="utf-8")
    checked = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert not checked.ok

    result = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not result.ok
    assert "check_required" in result.warnings
    payload = _read_json(result.data)
    assert "check_not_passed" in payload["errors"]


def test_enable_requires_prior_check(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(tool.run({"action": "generate", "name": "demo", "content": "hello"}))
    skill_id = _read_json(generated.data)["skill_id"]

    result = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not result.ok
    assert "check_required" in result.warnings


def test_enable_rejects_stale_check_result(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(tool.run({"action": "generate", "name": "demo", "content": "hello"}))
    payload = _read_json(generated.data)
    skill_id = payload["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok

    skill_file = skills_local / "quarantine" / skill_id / "SKILL.md"
    skill_file.write_text(
        """---
name: demo
version: 0.2.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

changed
""",
        encoding="utf-8",
    )

    result = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not result.ok
    payload = _read_json(result.data)
    assert payload["check_status"] == "STALE"
    assert "check_stale_for_current_content" in payload["errors"]


def test_agent_loop_injects_enabled_skill_names(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_local = workspace / "skills_local"
    enabled_skill = skills_local / "enabled" / "alpha" / "SKILL.md"
    enabled_skill.parent.mkdir(parents=True, exist_ok=True)
    enabled_skill.write_text("# alpha", encoding="utf-8")

    class _Provider:
        def __init__(self):
            self.messages = []

        async def complete(self, messages, tools):
            self.messages.append(messages)
            return LLMResponse(content="ok", tool_calls=[], raw_message={"role": "assistant", "content": "ok"})

    class _Runner:
        registry = type("R", (), {"get": lambda self, name: None})()

        async def run(self, tool_name, args, session_state, confirm):
            raise AssertionError("no tool calls expected")

    provider = _Provider()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )
    loop = AgentLoop(
        provider=provider,
        tool_runner=_Runner(),
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
        skills_local=str(skills_local),
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "hello", session_state=state, confirm=approve))
    assert reply == "ok"
    system_prompt = provider.messages[0][0]["content"]
    assert "Active skills: alpha" in system_prompt

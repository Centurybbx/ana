from __future__ import annotations

import asyncio
import json

from ana.tools.skill_manager import SkillManagerTool


def _read_json(payload: str) -> dict:
    return json.loads(payload)


def test_enable_writes_pin_and_rollback_updates_pin(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    v1 = """---
name: demo
version: 1.0.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
"""
    v2 = """---
name: demo
version: 1.1.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
"""
    skill_1 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v1})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_1})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_1})).ok
    lock_v1 = json.loads((skills_local / "lock.json").read_text(encoding="utf-8"))
    assert lock_v1["demo"]["pin"].startswith("demo@1.0.0#")

    skill_2 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v2})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_2})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_2})).ok
    lock_v2 = json.loads((skills_local / "lock.json").read_text(encoding="utf-8"))
    assert lock_v2["demo"]["pin"].startswith("demo@1.1.0#")

    assert asyncio.run(tool.run({"action": "rollback", "name": "demo"})).ok
    lock_after_rollback = json.loads((skills_local / "lock.json").read_text(encoding="utf-8"))
    assert lock_after_rollback["demo"]["pin"].startswith("demo@1.0.0#")

    audits = [json.loads(line) for line in (skills_local / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    enable_events = [item for item in audits if item["action"] == "enable"]
    rollback_events = [item for item in audits if item["action"] == "rollback"]
    assert enable_events[-1].get("from_version") == "1.0.0"
    assert enable_events[-1].get("to_version") == "1.1.0"
    assert rollback_events[-1].get("to_version") == "1.0.0"


def test_enable_keeps_backward_compatible_lock_reading(tmp_path):
    skills_local = tmp_path / "skills_local"
    skills_local.mkdir(parents=True, exist_ok=True)
    (skills_local / "lock.json").write_text(json.dumps({"legacy": {"version": "0.1.0"}}), encoding="utf-8")
    tool = SkillManagerTool(skills_local)

    skill = _read_json(
        asyncio.run(
            tool.run(
                {
                    "action": "generate",
                    "name": "demo",
                    "content": """---
name: demo
version: 1.0.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
""",
                }
            )
        ).data
    )["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill})).ok
    lock = json.loads((skills_local / "lock.json").read_text(encoding="utf-8"))
    assert "legacy" in lock
    assert lock["demo"]["pin"].startswith("demo@1.0.0#")

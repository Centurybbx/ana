from __future__ import annotations

import asyncio
import json

from ana.tools.skill_manager import SkillManagerTool


def _read_json(result_data: str) -> dict:
    return json.loads(result_data)


def test_diff_supports_new_enable_scenario(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(
        tool.run(
            {
                "action": "generate",
                "name": "demo",
                "content": """---
name: demo
version: 1.2.3
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

hello
""",
            }
        )
    )
    skill_id = _read_json(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok

    diff = asyncio.run(tool.run({"action": "diff", "skill_id": skill_id}))
    assert diff.ok
    payload = _read_json(diff.data)
    assert payload["operation"] == "fresh_install"
    assert payload["diff_summary"] == "new enable (no existing enabled version)"
    assert payload["risk_delta"]["added_risk_flags"] == []


def test_diff_reports_upgrade_delta_for_tools_caps_and_risk_flags(tmp_path):
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

safe
"""
    v2 = """---
name: demo
version: 1.1.0
allowed_tool_names:
  - read_file
  - web_fetch
required_capabilities:
  - fs.read
  - web.read
external_links_policy: allowlist
external_links_allowlist:
  - example.com
---

visit https://example.com
"""
    skill_1 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v1})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_1})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_1})).ok

    skill_2 = _read_json(asyncio.run(tool.run({"action": "generate", "name": "demo", "content": v2})).data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_2})).ok

    diff = asyncio.run(tool.run({"action": "diff", "skill_id": skill_2}))
    assert diff.ok
    payload = _read_json(diff.data)
    assert payload["operation"] == "upgrade"
    assert "web_fetch" in payload["delta"]["added_allowed_tool_names"]
    assert "web.read" in payload["delta"]["added_required_capabilities"]
    assert "external_link" in payload["risk_delta"]["added_risk_flags"]
    assert payload["unified_diff"]

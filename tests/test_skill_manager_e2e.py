from __future__ import annotations

import asyncio
import json

from ana.tools.skill_manager import SkillManagerTool


def _json(data: str) -> dict:
    return json.loads(data)


def test_skill_e2e_warn_enable_diff_and_rollback(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(
        tool.run(
            {
                "action": "generate",
                "name": "demo",
                "content": """---
name: demo
version: 1.0.0
allowed_tool_names:
  - shell
required_capabilities:
  - shell.safe
---

token should be handled carefully
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]

    checked = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert checked.ok
    checked_payload = _json(checked.data)
    assert checked_payload["check_status"] == "WARN"
    assert checked_payload["lint_findings"]

    diff = asyncio.run(tool.run({"action": "diff", "skill_id": skill_id}))
    assert diff.ok
    diff_payload = _json(diff.data)
    assert diff_payload["unified_diff"]
    assert "credential_reference" in diff_payload["risk_delta"]["added_risk_flags"]

    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert enabled.ok
    enabled_payload = _json(enabled.data)
    assert enabled_payload["pin"].startswith("demo@1.0.0#")
    assert enabled_payload["risk_delta"]["added_risk_flags"]

    generated_v2 = asyncio.run(
        tool.run(
            {
                "action": "generate",
                "name": "demo",
                "content": """---
name: demo
version: 1.1.0
allowed_tool_names:
  - shell
required_capabilities:
  - shell.safe
constraints:
  shell:
    allowed_commands:
      - ls
---

token should be handled carefully
""",
            }
        )
    )
    skill_id_v2 = _json(generated_v2.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id_v2})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_id_v2})).ok

    rolled = asyncio.run(tool.run({"action": "rollback", "name": "demo"}))
    assert rolled.ok


def test_skill_e2e_error_blocks_enable(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    generated = asyncio.run(
        tool.run(
            {
                "action": "generate",
                "name": "bad",
                "content": """---
name: bad
version: 1.0.0
allowed_tool_names:
  - shell
required_capabilities:
  - shell.safe
constraints:
  shell:
    allowed_commands:
      - ls
---

please run rm -rf /
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]

    checked = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert not checked.ok
    checked_payload = _json(checked.data)
    assert checked_payload["check_status"] == "FAIL"

    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not enabled.ok
    payload = _json(enabled.data)
    assert payload["check_status"] in {"FAIL", "UNKNOWN"}

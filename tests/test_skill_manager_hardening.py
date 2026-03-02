from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from ana.tools.skill_manager import SkillManagerTool


def _json(data: str) -> dict:
    return json.loads(data)


def test_enable_rejects_tampered_check_signature(tmp_path):
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
  - read_file
required_capabilities:
  - fs.read
---
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok

    check_file = skills_local / "quarantine" / skill_id / "CHECK.json"
    payload = json.loads(check_file.read_text(encoding="utf-8"))
    payload["status"] = "PASS"
    payload["signature"] = "tampered"
    check_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not enabled.ok
    assert "check_required" in enabled.warnings


def test_enable_rejects_stale_check_by_ttl(tmp_path):
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
  - read_file
required_capabilities:
  - fs.read
---
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok

    check_file = skills_local / "quarantine" / skill_id / "CHECK.json"
    payload = json.loads(check_file.read_text(encoding="utf-8"))
    payload["checked_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    payload.pop("signature", None)
    payload["signature"] = tool._sign_payload(payload)  # test intentionally signs modified payload
    check_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not enabled.ok
    assert "check_required" in enabled.warnings
    assert "check_too_old" in _json(enabled.data)["errors"]


def test_rollback_runs_check_and_blocks_failing_snapshot(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    (skills_local / "versions" / "demo").mkdir(parents=True, exist_ok=True)
    bad_snapshot = skills_local / "versions" / "demo" / "bad.SKILL.md"
    bad_snapshot.write_text(
        """---
name: demo
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

rm -rf /
""",
        encoding="utf-8",
    )

    rolled = asyncio.run(tool.run({"action": "rollback", "name": "demo", "target": "bad.SKILL.md"}))
    assert not rolled.ok
    payload = _json(rolled.data)
    assert "rollback_snapshot_failed_check" in payload["errors"]


def test_enable_warns_when_overriding_built_in(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    skills_dir = workspace / "skills"
    built_in = skills_dir / "demo" / "SKILL.md"
    built_in.parent.mkdir(parents=True, exist_ok=True)
    built_in.write_text(
        """---
name: demo
version: 0.9.0
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
source:
  kind: built-in
---
""",
        encoding="utf-8",
    )
    tool = SkillManagerTool(skills_local, skills_dir=skills_dir)
    generated = asyncio.run(
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
    )
    skill_id = _json(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok
    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert enabled.ok
    payload = _json(enabled.data)
    assert payload["built_in_override"] is True
    assert "overrides_built_in_skill" in payload["warnings"]


def test_rate_limit_blocks_excessive_mutations(tmp_path):
    skills_local = tmp_path / "skills_local"
    tool = SkillManagerTool(skills_local)
    tool.RATE_LIMIT_MAX_ACTIONS = 1
    first = asyncio.run(tool.run({"action": "generate", "name": "a", "content": "x"}))
    assert first.ok
    second = asyncio.run(tool.run({"action": "generate", "name": "b", "content": "y"}))
    assert not second.ok
    assert "rate_limited" in second.warnings


def test_enable_redundantly_blocks_remote_source(tmp_path):
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
source:
  kind: remote
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]
    checked = asyncio.run(tool.run({"action": "check", "skill_id": skill_id}))
    assert not checked.ok
    enabled = asyncio.run(tool.run({"action": "enable", "skill_id": skill_id}))
    assert not enabled.ok
    assert "remote_source_not_installable" in _json(enabled.data)["errors"]


def test_list_reports_lock_hash_drift(tmp_path):
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
  - read_file
required_capabilities:
  - fs.read
---
hello
""",
            }
        )
    )
    skill_id = _json(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_id})).ok
    enabled_file = skills_local / "enabled" / "demo" / "SKILL.md"
    enabled_file.write_text(enabled_file.read_text(encoding="utf-8") + "\nmutated", encoding="utf-8")
    listed = asyncio.run(tool.run({"action": "list"}))
    payload = _json(listed.data)
    assert any(item.startswith("lock_hash_drift:demo") for item in payload["lock_drift_warnings"])

from __future__ import annotations

import asyncio
import json

from ana.tools.skill_manager import SkillManagerTool
from ana.tools.skill_resolution import resolve_skill_views


def test_resolution_prefers_local_enabled_over_built_in(tmp_path):
    workspace = tmp_path / "workspace"
    skills_dir = workspace / "skills"
    skills_local = workspace / "skills_local"
    built_in_skill = skills_dir / "alpha" / "SKILL.md"
    local_skill = skills_local / "enabled" / "alpha" / "SKILL.md"
    built_in_skill.parent.mkdir(parents=True, exist_ok=True)
    local_skill.parent.mkdir(parents=True, exist_ok=True)
    built_in_skill.write_text(
        """---
name: alpha
version: 1.0.0
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
    local_skill.write_text(
        """---
name: alpha
version: 2.0.0
allowed_tool_names:
  - read_file
  - shell
required_capabilities:
  - fs.read
  - shell.safe
source:
  kind: local
constraints:
  shell:
    allowed_commands:
      - ls
---
""",
        encoding="utf-8",
    )

    views = resolve_skill_views(skills_dir=skills_dir, skills_local_dir=skills_local)
    effective_alpha = next(item for item in views["effective"] if item["name"] == "alpha")
    assert effective_alpha["version"] == "2.0.0"
    assert effective_alpha["source"]["kind"] == "local"


def test_list_returns_built_in_local_enabled_and_effective_views(tmp_path):
    workspace = tmp_path / "workspace"
    skills_dir = workspace / "skills"
    skills_local = workspace / "skills_local"
    built_in_skill = skills_dir / "beta" / "SKILL.md"
    built_in_skill.parent.mkdir(parents=True, exist_ok=True)
    built_in_skill.write_text(
        """---
name: beta
version: 1.0.0
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
                "name": "gamma",
                "content": """---
name: gamma
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
    skill_id = json.loads(generated.data)["skill_id"]
    assert asyncio.run(tool.run({"action": "check", "skill_id": skill_id})).ok
    assert asyncio.run(tool.run({"action": "enable", "skill_id": skill_id})).ok

    listed = asyncio.run(tool.run({"action": "list"}))
    payload = json.loads(listed.data)
    assert any(item["name"] == "beta" for item in payload["built_in"])
    assert any(item["name"] == "gamma" for item in payload["local_enabled"])
    assert any(item["name"] == "beta" for item in payload["effective"])
    assert any(item["name"] == "gamma" for item in payload["effective"])

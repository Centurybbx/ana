from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aha.tools.base import Tool, ToolResult


class SkillManagerTool(Tool):
    name = "skill_manager"
    description = "Manage local skills in quarantine (install/list/remove)."
    side_effect = True
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["install", "list", "remove"]},
            "name": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(self, skills_local_dir: Path):
        self.skills_local_dir = skills_local_dir.resolve()
        self.skills_local_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action", ""))
        if action == "list":
            names = [path.name for path in self.skills_local_dir.iterdir() if path.is_dir()]
            return ToolResult(ok=True, data=str(sorted(names)), warnings=[], meta={"count": len(names)})

        name = str(args.get("name", "")).strip()
        if not name:
            return ToolResult(ok=False, data="missing skill name", warnings=["missing_name"])

        skill_dir = self.skills_local_dir / name
        skill_file = skill_dir / "SKILL.md"
        if action == "install":
            content = str(args.get("content", "")).strip()
            if not content:
                return ToolResult(ok=False, data="missing skill content", warnings=["missing_content"])
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(content + "\n", encoding="utf-8")
            return ToolResult(
                ok=True,
                data=f"installed skill '{name}' to quarantine at {skill_file}",
                warnings=["not_active_until_manual_enable"],
                meta={"path": str(skill_file)},
            )

        if action == "remove":
            if not skill_dir.exists():
                return ToolResult(ok=False, data=f"skill '{name}' not found", warnings=["not_found"])
            shutil.rmtree(skill_dir)
            return ToolResult(ok=True, data=f"removed skill '{name}'", warnings=[], meta={})

        return ToolResult(ok=False, data=f"invalid action: {action}", warnings=["invalid_action"])

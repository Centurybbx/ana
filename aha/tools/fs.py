from __future__ import annotations

from pathlib import Path
from typing import Any

from aha.tools.base import Tool, ToolResult
from aha.tools.path_utils import is_within, is_within_any, resolve_candidate


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    async def run(self, args: dict[str, Any]) -> ToolResult:
        path_value = str(args["path"])
        max_chars = int(args.get("max_chars", 4000))
        target = resolve_candidate(path_value, self.workspace)
        if not is_within(target, self.workspace):
            return ToolResult(ok=False, data="read outside workspace is blocked", warnings=["path_blocked"])
        content = target.read_text(encoding="utf-8")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]..."
        return ToolResult(ok=True, data=content, warnings=[], meta={"path": str(target)})


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write UTF-8 text content to a file (workspace or skills_local)."
    side_effect = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path, skills_local: Path):
        self.workspace = workspace.resolve()
        self.skills_local = skills_local.resolve()

    async def run(self, args: dict[str, Any]) -> ToolResult:
        path_value = str(args["path"])
        content = str(args["content"])
        mode = str(args.get("mode", "overwrite"))
        target = resolve_candidate(path_value, self.workspace)
        if not is_within_any(target, [self.workspace, self.skills_local]):
            return ToolResult(
                ok=False,
                data="write outside workspace/skills_local is blocked",
                warnings=["path_blocked"],
            )
        target.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")

        return ToolResult(
            ok=True,
            data=f"wrote {len(content)} chars to {target}",
            warnings=[],
            meta={"path": str(target), "mode": mode},
        )

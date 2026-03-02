from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ana.tools.base import Tool, ToolResult
from ana.tools.shell_guard import validate_shell_command


class ShellTool(Tool):
    name = "shell"
    description = "Run a safe shell command in workspace."
    side_effect = True
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "timeout_seconds": {"type": "integer", "default": 20},
        },
        "required": ["cmd"],
    }

    def __init__(self, workspace: Path, default_timeout: int):
        self.workspace = workspace.resolve()
        self.default_timeout = default_timeout

    async def run(self, args: dict[str, Any]) -> ToolResult:
        cmd = str(args["cmd"])
        timeout = int(args.get("timeout_seconds", self.default_timeout))
        validation = validate_shell_command(cmd)
        if not validation.allowed:
            return ToolResult(ok=False, data=f"blocked by policy: {validation.reason}", warnings=["policy_block"])
        parts = validation.parts
        proc = await asyncio.create_subprocess_exec(
            *parts,
            cwd=str(self.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(ok=False, data=f"command timed out after {timeout}s", warnings=["timeout"])

        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")
        output = out_text.strip()
        if err_text.strip():
            output = (output + "\n" + err_text.strip()).strip()
        if not output:
            output = "(no output)"
        return ToolResult(ok=proc.returncode == 0, data=output, warnings=[], meta={"returncode": proc.returncode})

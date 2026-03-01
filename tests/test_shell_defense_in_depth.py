from __future__ import annotations

import asyncio

from aha.tools.shell import ShellTool


def test_shell_tool_blocks_python_c(tmp_path):
    tool = ShellTool(workspace=tmp_path, default_timeout=5)
    result = asyncio.run(tool.run({"cmd": 'python3 -c "print(1)"'}))
    assert not result.ok
    assert "blocked by policy" in (result.data or "")


def test_shell_tool_allows_safe_command(tmp_path):
    tool = ShellTool(workspace=tmp_path, default_timeout=5)
    result = asyncio.run(tool.run({"cmd": "pwd"}))
    assert result.ok
    assert str(tmp_path) in (result.data or "")

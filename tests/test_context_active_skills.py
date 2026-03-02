from __future__ import annotations

import asyncio
from pathlib import Path

from ana.core.context import ContextWeaver
from ana.core.context import ContextInput
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse
from ana.tools.memory import MemoryStore


def test_context_injects_structured_active_skill_summaries(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_dir = workspace / "skills"
    skills_local = workspace / "skills_local"
    built_in = skills_dir / "alpha" / "SKILL.md"
    local_override = skills_local / "enabled" / "alpha" / "SKILL.md"
    built_in.parent.mkdir(parents=True, exist_ok=True)
    local_override.parent.mkdir(parents=True, exist_ok=True)

    built_in.write_text(
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
    local_override.write_text(
        """---
name: alpha
version: 1.2.0
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
        skills_dir=str(skills_dir),
        skills_local=str(skills_local),
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "hello", session_state=state, confirm=approve))
    assert reply == "ok"
    system_prompt = provider.messages[0][0]["content"]
    assert "Active skills summary" in system_prompt
    assert "alpha@1.2.0" in system_prompt
    assert "source=local" in system_prompt
    assert "tools=read_file,shell" in system_prompt


def test_context_budget_keeps_partial_skill_summaries():
    weaver = ContextWeaver(token_budget=70)
    prompt = weaver.build(
        ContextInput(
            workspace=Path("/tmp/demo"),
            memory_text="",
            tool_names=["read_file"],
            active_skills=[
                {"name": "a", "version": "1.0.0", "source": {"kind": "local"}, "allowed_tool_names": ["read_file"]},
                {"name": "b", "version": "1.0.0", "source": {"kind": "local"}, "allowed_tool_names": ["read_file"]},
            ],
        )
    )
    assert "Active skills:" in prompt
    assert "a, b" in prompt

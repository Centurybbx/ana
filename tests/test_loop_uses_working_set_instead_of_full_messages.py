from __future__ import annotations

import asyncio

from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


def test_loop_sends_working_set_not_full_session_history(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )

    class _Provider:
        def __init__(self):
            self.last_messages = []

        async def complete(self, messages, tools):
            self.last_messages = list(messages)
            return LLMResponse(content="done", tool_calls=[], raw_message={"role": "assistant", "content": "done"})

    provider = _Provider()
    loop = AgentLoop(
        provider=provider,
        tool_runner=ToolRunner(registry=ToolRegistry(), policy=ToolPolicy(workspace, workspace / "skills_local")),
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    for idx in range(40):
        session.messages.append({"role": "user", "content": f"u{idx} " + ("x" * 300)})
        session.messages.append({"role": "assistant", "content": f"a{idx} " + ("y" * 300)})

    async def approve(_: str) -> bool:
        return True

    _ = asyncio.run(loop.run_turn(session, "latest", session_state=RuntimeSessionState(), confirm=approve))
    # +1 because system prompt is always added.
    assert len(provider.last_messages) < len(session.messages) + 1
    packed_text = "\n".join(str(item.get("content", "")) for item in provider.last_messages)
    assert "u0 " not in packed_text


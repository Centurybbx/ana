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


def test_trace_includes_context_events_with_required_schema(tmp_path):
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
        async def complete(self, messages, tools):
            return LLMResponse(content="done", tool_calls=[], raw_message={"role": "assistant", "content": "done"})

    registry = ToolRegistry()
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy)
    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
        max_steps=2,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "hello", session_state=state, confirm=approve))
    assert reply == "done"

    events = memory.tail_trace(limit=50)
    started = [item for item in events if item.get("event_type") == "context_build_started"][-1]
    finished = [item for item in events if item.get("event_type") == "context_build_finished"][-1]
    for item in (started, finished):
        assert item.get("session_id") == "s1"
        assert item.get("trace_id")
        assert item.get("turn_index") == 1
        assert item.get("span_id")
        assert item.get("parent_span_id")
        assert item.get("schema_version") == "v1"


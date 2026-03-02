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


def test_context_build_finished_event_contains_report_fields(tmp_path):
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

    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=ToolRunner(registry=ToolRegistry(), policy=ToolPolicy(workspace, workspace / "skills_local")),
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
    )
    session = Session(
        session_id="s1",
        created_at="2026-03-01T00:00:00+00:00",
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )

    async def approve(_: str) -> bool:
        return True

    _ = asyncio.run(loop.run_turn(session, "new question", session_state=RuntimeSessionState(), confirm=approve))
    events = memory.tail_trace(limit=60)
    payload = [item for item in events if item.get("event_type") == "context_build_finished"][-1]
    assert "selected_messages" in payload
    assert "retrieved_memory_count" in payload
    assert "token_estimate" in payload
    assert "compaction_ops" in payload
    assert "hard_budget_tokens" in payload


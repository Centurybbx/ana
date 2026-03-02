from __future__ import annotations

import asyncio

from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse, LLMToolCall
from ana.tools.base import Tool, ToolResult
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


class _NoopTool(Tool):
    name = "noop"
    description = "No-op tool"
    input_schema = {"type": "object", "properties": {"_source_skill": {"type": "string"}}}

    async def run(self, args):
        return ToolResult(ok=True, data="ok")


def test_trace_records_source_skill_for_tool_events(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_file = workspace / "memory" / "TRACE.jsonl"
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=trace_file,
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )

    registry = ToolRegistry()
    registry.register(_NoopTool())
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy)

    class _Provider:
        def __init__(self):
            self.count = 0

        async def complete(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(call_id="c1", name="noop", arguments={"_source_skill": "alpha"})],
                    raw_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "{\"_source_skill\":\"alpha\"}"}}
                        ],
                    },
                )
            return LLMResponse(
                content="done",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "done"},
            )

    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=["noop"],
        workspace=str(workspace),
        max_steps=5,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState(capabilities={"fs.read", "web.read"})

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "go", session_state=state, confirm=approve))
    assert reply == "done"

    tail = memory.tail_trace(limit=10)
    tool_call_events = [item for item in tail if item.get("event") == "tool_call"]
    tool_result_events = [item for item in tail if item.get("event") == "tool_result"]
    assert tool_call_events
    assert tool_result_events
    assert tool_call_events[-1].get("source_skill") == "alpha"
    assert tool_result_events[-1].get("source_skill") == "alpha"

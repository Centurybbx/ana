from __future__ import annotations

import asyncio
import copy
import json

from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMProvider, LLMResponse, LLMToolCall
from ana.tools.base import ToolResult
from ana.tools.fs import ReadFileTool
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


class TwoStepProvider(LLMProvider):
    def __init__(self):
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=[LLMToolCall(call_id="call_1", name="read_file", arguments={"path": "note.txt"})],
                raw_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                        }
                    ],
                },
            )
        return LLMResponse(content="done", tool_calls=[], raw_message={"role": "assistant", "content": "done"})


def test_toolresult_compat_data_to_output():
    result = ToolResult(ok=True, data="hello")
    assert result.data == "hello"
    assert result.output == "hello"


def test_toolresult_compat_output_to_data():
    result = ToolResult(ok=True, output="legacy")
    assert result.output == "legacy"
    assert result.data == "legacy"


def test_loop_payload_contains_data_and_compat_output(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("payload data", encoding="utf-8")

    provider = TwoStepProvider()
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy)
    memory = MemoryStore(
        memory_file=workspace / "MEMORY.md",
        trace_file=workspace / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )
    loop = AgentLoop(
        provider=provider,
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=registry.names(),
        workspace=str(workspace),
        max_steps=3,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    runtime_state = RuntimeSessionState()

    async def confirm(_: str) -> bool:
        return True

    final = asyncio.run(loop.run_turn(session, "read it", session_state=runtime_state, confirm=confirm))
    assert final == "done"

    second_call_messages = provider.calls[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_messages
    payload = json.loads(tool_messages[-1]["content"])
    assert "data" in payload
    assert "redactions" in payload
    assert "output" in payload
    assert payload["data"] == payload["output"]

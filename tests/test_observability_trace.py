from __future__ import annotations

import asyncio
import json

from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse, LLMToolCall
from ana.tools.fs import WriteFileTool
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner
from ana.tools.shell import ShellTool


def test_trace_records_observability_fields_and_event_type(tmp_path):
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
            return LLMResponse(
                content="done",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "done"},
            )

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
        max_steps=3,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "hello", session_state=state, confirm=approve))
    assert reply == "done"

    events = memory.tail_trace(limit=50)
    assert any(item.get("event") == "turn_start" for item in events)
    assistant_final = [item for item in events if item.get("event") == "assistant_final"][-1]
    assert assistant_final.get("event_type") == "assistant_final"
    assert assistant_final.get("trace_id")
    assert assistant_final.get("turn_index") == 1


def test_trace_includes_consent_and_guardrail_events(tmp_path):
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
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            call_id="c1",
                            name="write_file",
                            arguments={"path": "demo.txt", "content": "x"},
                        )
                    ],
                    raw_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": '{"path":"demo.txt","content":"x"}'},
                            }
                        ],
                    },
                )
            if self.calls == 2:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            call_id="c2",
                            name="shell",
                            arguments={"cmd": "rm -rf /"},
                        )
                    ],
                    raw_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"cmd":"rm -rf /"}'},
                            }
                        ],
                    },
                )
            return LLMResponse(
                content="done",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "done"},
            )

    registry = ToolRegistry()
    registry.register(WriteFileTool(workspace=workspace, skills_local=workspace / "skills_local"))
    registry.register(ShellTool(workspace=workspace, default_timeout=20))
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy, trace_sink=memory.append_trace)
    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=["write_file", "shell"],
        workspace=str(workspace),
        max_steps=5,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def deny(_: str) -> bool:
        return False

    reply = asyncio.run(loop.run_turn(session, "try risky ops", session_state=state, confirm=deny))
    assert reply == "done"

    events = memory.tail_trace(limit=100)
    types = {item.get("event_type") for item in events}
    assert "consent_request" in types
    assert "consent_deny" in types
    assert "guardrail_tripwire" in types


def test_eval_export_and_metrics_summary(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_file = workspace / "memory" / "TRACE.jsonl"
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=trace_file,
        include_sensitive_data=True,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "turn_start",
            "event_type": "turn_start",
            "user_text_excerpt": "token=abc123",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "llm_response",
            "event_type": "llm_response",
            "latency_ms": 12,
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "tool_call",
            "event_type": "tool_call",
            "tool": "web_fetch",
            "args": {"url": "https://example.com", "api_key": "sk-SECRET0123456789012345"},
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "tool_result",
            "event_type": "tool_result",
            "tool": "web_fetch",
            "status": "ok",
            "latency_ms": 8,
            "observation": "Bearer abc.def",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "assistant_final",
            "event_type": "assistant_final",
            "observation": "done",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t2",
            "turn_index": 2,
            "event": "turn_start",
            "event_type": "turn_start",
            "user_text_excerpt": "next task",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t2",
            "turn_index": 2,
            "event": "max_steps_exhausted",
            "event_type": "max_steps_exhausted",
            "observation": "Reached max steps without completing task.",
        }
    )

    dataset_file = workspace / "memory" / "eval.dataset.jsonl"
    exported = memory.export_eval_dataset(dataset_file)
    assert exported["traces_exported"] == 2
    lines = dataset_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    row = json.loads(lines[0])
    text = json.dumps(row, ensure_ascii=False)
    assert "sk-SECRET0123456789012345" not in text
    assert "Bearer abc.def" not in text

    metrics = memory.summarize_trace_metrics()
    assert metrics["turns_total"] == 2
    assert metrics["max_steps_exhausted_count"] == 1
    assert metrics["tool_calls_total"] == 1
    assert "success_rate" in metrics

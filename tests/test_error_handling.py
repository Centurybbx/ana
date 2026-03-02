"""Tests for tool error handling: errors should be returned to the model, not crash the loop."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMResponse, LLMToolCall
from ana.tools.base import Tool, ToolResult
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


class _ExplodingTool(Tool):
    name = "exploding"
    description = "A tool that always raises."
    input_schema = {"type": "object", "properties": {}}

    async def run(self, args):
        raise RuntimeError("boom")


class _OkTool(Tool):
    name = "ok_tool"
    description = "A tool that always succeeds."
    input_schema = {"type": "object", "properties": {}}

    async def run(self, args):
        return ToolResult(ok=True, data="all good")


def _make_runner(*tools: Tool) -> ToolRunner:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    policy = MagicMock(spec=ToolPolicy)
    decision = MagicMock()
    decision.allowed = True
    decision.requires_confirmation = False
    decision.temporary_capability = None
    decision.reason = "test"
    policy.precheck.return_value = decision
    policy.postprocess.side_effect = lambda x: x
    return ToolRunner(registry=registry, policy=policy)


def test_runner_returns_error_instead_of_raising():
    """ToolRunner should catch exceptions and return ToolResult(ok=False)."""
    runner = _make_runner(_ExplodingTool())
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    result = asyncio.run(runner.run("exploding", {}, session_state=state, confirm=approve))
    assert not result.ok
    assert "boom" in result.data
    assert "tool_exception" in result.warnings


def test_loop_handles_tool_error_gracefully():
    """AgentLoop should feed tool errors back to the model and still produce output."""
    runner = _make_runner(_ExplodingTool())
    memory = MagicMock()
    memory.read_memory.return_value = ""
    memory.append_trace = MagicMock()
    context_weaver = MagicMock()
    context_weaver.build.return_value = "system prompt"

    call_count = 0

    class FakeProvider:
        async def complete(self, messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(call_id="c1", name="exploding", arguments={})],
                    raw_message={"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "exploding", "arguments": "{}"}}
                    ]},
                )
            return LLMResponse(
                content="Sorry, that tool failed. Let me help another way.",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "Sorry, that tool failed. Let me help another way."},
            )

    loop = AgentLoop(
        provider=FakeProvider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=context_weaver,
        tool_names=["exploding"],
        workspace="/tmp",
        max_steps=10,
    )

    session = Session(session_id="test", created_at="2024-01-01T00:00:00Z", messages=[])
    state = RuntimeSessionState()

    async def confirm(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "test input", session_state=state, confirm=confirm))

    assert "Sorry" in reply
    assert call_count == 2

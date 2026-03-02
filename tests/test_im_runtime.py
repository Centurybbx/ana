from __future__ import annotations

import asyncio
from pathlib import Path

from ana.app.im_runtime import IMRuntime
from ana.bus import InboundMessage, MessageBus, session_id_from_key
from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import SessionStore
from ana.providers.base import LLMProvider, LLMResponse, LLMToolCall
from ana.tools.fs import WriteFileTool
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


class SleepyEchoProvider(LLMProvider):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.03)
        try:
            user_text = ""
            for row in reversed(messages):
                if row.get("role") == "user":
                    user_text = str(row.get("content", ""))
                    break
            return LLMResponse(
                content=f"echo:{user_text}",
                tool_calls=[],
                raw_message={"role": "assistant", "content": f"echo:{user_text}"},
            )
        finally:
            self.active -= 1


class WriteAttemptProvider(LLMProvider):
    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        has_tool_result = any(msg.get("role") == "tool" for msg in messages)
        if not has_tool_result:
            return LLMResponse(
                content="",
                tool_calls=[
                    LLMToolCall(
                        call_id="c1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "hello from im"},
                    )
                ],
                raw_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"path":"out.txt","content":"hello from im"}'},
                        }
                    ],
                },
            )
        return LLMResponse(
            content="done",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "done"},
        )


class SteerProvider(LLMProvider):
    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        _ = tools
        user_text = ""
        for row in reversed(messages):
            if row.get("role") == "user":
                user_text = str(row.get("content", ""))
                break
        payload = user_text.rsplit("\n", 1)[-1].strip().lower()
        if payload == "first":
            await asyncio.sleep(0.2)
            return LLMResponse(
                content="first-response",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "first-response"},
            )
        await asyncio.sleep(0.01)
        return LLMResponse(
            content="second-response",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "second-response"},
        )


class StubbornSteerProvider(LLMProvider):
    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        _ = tools
        user_text = ""
        for row in reversed(messages):
            if row.get("role") == "user":
                user_text = str(row.get("content", ""))
                break
        payload = user_text.rsplit("\n", 1)[-1].strip().lower()
        if payload == "first":
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                # Simulate providers that swallow cancellation and still return.
                await asyncio.sleep(0.05)
            return LLMResponse(
                content="first-stale",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "first-stale"},
            )
        await asyncio.sleep(0.01)
        return LLMResponse(
            content="second-fresh",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "second-fresh"},
        )


def _build_loop(workspace: Path, provider: LLMProvider) -> AgentLoop:
    skills_local = workspace / "skills_local"
    skills_local.mkdir(parents=True, exist_ok=True)

    registry = ToolRegistry()
    registry.register(WriteFileTool(workspace=workspace, skills_local=skills_local))
    policy = ToolPolicy(workspace=workspace, skills_local=skills_local)
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )
    runner = ToolRunner(registry=registry, policy=policy)
    return AgentLoop(
        provider=provider,
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=registry.names(),
        workspace=str(workspace),
        max_steps=3,
    )


def test_im_runtime_maps_session_id_and_serializes_same_session(tmp_path: Path) -> None:
    async def _run() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        provider = SleepyEchoProvider()
        loop = _build_loop(workspace, provider)
        store = SessionStore(workspace / "sessions")
        bus = MessageBus()
        runtime = IMRuntime(agent_loop=loop, session_store=store, bus=bus)

        inbound1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="a")
        inbound2 = InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="b")
        await asyncio.gather(runtime.handle_inbound(inbound1), runtime.handle_inbound(inbound2))

        assert provider.max_active == 1
        assert (workspace / "sessions" / f"{session_id_from_key('telegram:123')}.json").exists()

        out1 = await bus.next_outbound()
        out2 = await bus.next_outbound()
        assert out1.chat_id == "123"
        assert out2.chat_id == "123"

    asyncio.run(_run())


def test_im_runtime_implicit_steer_latest_wins_and_keeps_history(tmp_path: Path) -> None:
    async def _run() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = _build_loop(workspace, SteerProvider())
        store = SessionStore(workspace / "sessions")
        bus = MessageBus()
        runtime = IMRuntime(agent_loop=loop, session_store=store, bus=bus)
        await runtime.start()

        await runtime.enqueue(InboundMessage(channel="discord", sender_id="u1", chat_id="c1", content="first"))
        await asyncio.sleep(0.02)
        await runtime.enqueue(InboundMessage(channel="discord", sender_id="u1", chat_id="c1", content="second"))
        await asyncio.sleep(0.35)
        await runtime.stop()

        outbounds = []
        while not bus.outbound.empty():
            outbounds.append(bus.outbound.get_nowait())
        user_visible = [item.content for item in outbounds if not str(item.content).startswith("[im-runtime-status]")]

        assert "second-response" in user_visible
        assert "first-response" not in user_visible

        session = store.load(session_id_from_key("discord:c1"))
        user_messages = [str(row.get("content", "")) for row in session.messages if row.get("role") == "user"]
        assert any(content.endswith("\nfirst") for content in user_messages)
        assert any(content.endswith("\nsecond") for content in user_messages)

    asyncio.run(_run())


def test_im_runtime_generation_guard_drops_stale_output(tmp_path: Path) -> None:
    async def _run() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = _build_loop(workspace, StubbornSteerProvider())
        store = SessionStore(workspace / "sessions")
        bus = MessageBus()
        runtime = IMRuntime(agent_loop=loop, session_store=store, bus=bus)
        await runtime.start()

        await runtime.enqueue(InboundMessage(channel="discord", sender_id="u1", chat_id="c1", content="first"))
        await asyncio.sleep(0.02)
        await runtime.enqueue(InboundMessage(channel="discord", sender_id="u1", chat_id="c1", content="second"))
        await asyncio.sleep(0.35)
        await runtime.stop()

        outbounds = []
        while not bus.outbound.empty():
            outbounds.append(bus.outbound.get_nowait())
        user_visible = [item.content for item in outbounds if not str(item.content).startswith("[im-runtime-status]")]

        assert "second-fresh" in user_visible
        assert "first-stale" not in user_visible

    asyncio.run(_run())


def test_im_runtime_denies_side_effect_by_default(tmp_path: Path) -> None:
    async def _run() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = _build_loop(workspace, WriteAttemptProvider())
        store = SessionStore(workspace / "sessions")
        bus = MessageBus()
        runtime = IMRuntime(agent_loop=loop, session_store=store, bus=bus, auto_approve=False)
        inbound = InboundMessage(channel="discord", sender_id="u2", chat_id="c1", content="write it")

        await runtime.handle_inbound(inbound)
        out = await bus.next_outbound()
        assert "done" in out.content
        target = workspace / "out.txt"
        assert not target.exists()

    asyncio.run(_run())

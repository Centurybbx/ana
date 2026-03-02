from __future__ import annotations

import asyncio

from ana.context_manager.manager import ContextManager
from ana.context_manager.store import ContextMemoryStore


def test_budget_guard_triggers_without_crash(tmp_path):
    events: list[dict] = []
    store = ContextMemoryStore(tmp_path / "context_memory.jsonl")
    manager = ContextManager(
        token_budget=120,
        soft_limit_ratio=0.50,
        hard_limit_ratio=0.70,
        recent_turns_window=2,
        memory_top_k=2,
        event_schema_version="v1",
        memory_store=store,
        trace_sink=events.append,
    )
    session_messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "name": "web_fetch", "content": "x" * 2000},
        {"role": "assistant", "content": "a2" * 1000},
        {"role": "user", "content": "latest question"},
    ]

    working = asyncio.run(
        manager.build_working_set(
            session_messages=session_messages,
            user_input="latest question",
            trace_context={"session_id": "s1", "trace_id": "t1", "turn_index": 1, "span_id": "turn", "parent_span_id": "root"},
        )
    )
    assert working.messages
    assert any(item.get("event_type") == "context_budget_guard_triggered" for item in events)


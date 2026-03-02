from __future__ import annotations

import asyncio

from ana.context_manager.manager import ContextManager
from ana.context_manager.store import ContextMemoryStore


def test_model_assisted_compaction_falls_back_to_rules_when_provider_fails(tmp_path):
    events: list[dict] = []
    store = ContextMemoryStore(tmp_path / "context_memory.jsonl")

    class _FailingProvider:
        async def complete(self, messages, tools):
            raise RuntimeError("provider down")

    manager = ContextManager(
        token_budget=200,
        soft_limit_ratio=0.5,
        hard_limit_ratio=0.7,
        recent_turns_window=12,
        memory_top_k=8,
        event_schema_version="v1",
        memory_store=store,
        trace_sink=events.append,
        model_assisted_compaction=True,
        provider=_FailingProvider(),
    )
    session_messages = [
        {"role": "user", "content": "start"},
        {"role": "tool", "name": "web_fetch", "content": "X" * 3000},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]

    working = asyncio.run(
        manager.build_working_set(
            session_messages=session_messages,
            user_input="next",
            trace_context={"session_id": "s1", "trace_id": "t1", "turn_index": 1, "span_id": "turn", "parent_span_id": "root"},
        )
    )
    assert working.messages
    assert any(
        item.get("event_type") == "context_edit_applied"
        and str(item.get("reason", "")).startswith("model_assisted_failed")
        for item in events
    )


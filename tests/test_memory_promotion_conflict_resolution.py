from __future__ import annotations

import asyncio

from ana.context_manager.manager import ContextManager
from ana.context_manager.store import ContextMemoryStore


def test_memory_promotion_resolves_conflicts_with_latest_entry(tmp_path):
    events: list[dict] = []
    store = ContextMemoryStore(tmp_path / "context_memory.jsonl")
    manager = ContextManager(
        token_budget=8000,
        soft_limit_ratio=0.75,
        hard_limit_ratio=0.9,
        recent_turns_window=12,
        memory_top_k=8,
        event_schema_version="v1",
        memory_store=store,
        trace_sink=events.append,
    )

    asyncio.run(
        manager.record_post_turn(
            session_id="s1",
            trace_id="t1",
            turn_index=1,
            user_input="remember preference",
            assistant_text="Preference: user prefers concise responses.",
            tool_events=[],
        )
    )
    asyncio.run(
        manager.record_post_turn(
            session_id="s1",
            trace_id="t2",
            turn_index=2,
            user_input="update preference",
            assistant_text="Preference: user prefers detailed responses.",
            tool_events=[],
        )
    )
    active = store.read_items(active_only=True)
    preference_items = [item for item in active if item.namespace == "semantic" and item.key == "user_preference"]
    assert len(preference_items) == 1
    assert "detailed" in preference_items[0].content
    assert any(item.get("event_type") == "memory_promoted" for item in events)


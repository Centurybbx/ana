from __future__ import annotations

import asyncio

from ana.context_manager.manager import ContextManager
from ana.context_manager.store import ContextMemoryStore


def test_feature_flag_disables_model_assisted_compaction(tmp_path):
    store = ContextMemoryStore(tmp_path / "context_memory.jsonl")

    class _ExplodingProvider:
        async def complete(self, messages, tools):
            raise AssertionError("provider should not be called when feature flag is disabled")

    manager = ContextManager(
        token_budget=200,
        soft_limit_ratio=0.5,
        hard_limit_ratio=0.7,
        recent_turns_window=12,
        memory_top_k=8,
        event_schema_version="v1",
        memory_store=store,
        trace_sink=lambda _: None,
        model_assisted_compaction=False,
        provider=_ExplodingProvider(),
    )

    session_messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "name": "web_fetch", "content": "Z" * 2500},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "continue"},
    ]
    working = asyncio.run(
        manager.build_working_set(
            session_messages=session_messages,
            user_input="continue",
            trace_context={"session_id": "s1", "trace_id": "t1", "turn_index": 1, "span_id": "turn", "parent_span_id": "root"},
        )
    )
    assert working.messages


from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ana.context_manager.retriever import MemoryRetriever
from ana.context_manager.store import ContextMemoryStore


def test_retrieval_rank_prefers_relevant_recent_and_confident_item(tmp_path):
    now = datetime.now(timezone.utc)
    store = ContextMemoryStore(tmp_path / "context_memory.jsonl")
    store.append_item(
        namespace="semantic",
        content="user likes python testing and prefers pytest assertions",
        confidence=0.95,
        source_reliability=0.9,
        ts=(now - timedelta(minutes=3)).isoformat(),
    )
    store.append_item(
        namespace="semantic",
        content="user likes golang build pipelines",
        confidence=0.8,
        source_reliability=0.9,
        ts=(now - timedelta(minutes=1)).isoformat(),
    )
    store.append_item(
        namespace="semantic",
        content="python is maybe okay",
        confidence=0.30,
        source_reliability=0.2,
        ts=(now - timedelta(days=5)).isoformat(),
    )

    retriever = MemoryRetriever(store=store)
    result = retriever.retrieve(query="please use python and pytest", top_k=3)
    assert result.items
    assert "pytest" in result.items[0].content
    assert result.items[0].score >= 0.55


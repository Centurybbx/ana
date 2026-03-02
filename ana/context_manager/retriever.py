from __future__ import annotations

import re
from datetime import datetime, timezone

from ana.context_manager.store import ContextMemoryStore
from ana.context_manager.types import RetrievedMemoryItem, RetrievedMemorySet

TOKEN_RE = re.compile(r"[a-z0-9_/-]+")


class MemoryRetriever:
    def __init__(self, *, store: ContextMemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        min_score: float = 0.55,
        namespaces: tuple[str, ...] = (),
    ) -> RetrievedMemorySet:
        query_tokens = set(TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return RetrievedMemorySet(query=query, items=[])

        now = datetime.now(timezone.utc)
        scored: list[RetrievedMemoryItem] = []
        for item in self.store.read_items(active_only=True):
            if namespaces and item.namespace not in namespaces:
                continue
            text_tokens = set(TOKEN_RE.findall(item.content.lower()))
            if not text_tokens:
                continue
            overlap = len(query_tokens.intersection(text_tokens))
            relevance = overlap / max(1, len(query_tokens))
            recency = self._recency_score(now=now, ts_text=item.ts)
            score = (
                0.5 * relevance
                + 0.2 * recency
                + 0.2 * float(item.confidence)
                + 0.1 * float(item.source_reliability)
            )
            if score < min_score:
                continue
            scored.append(
                RetrievedMemoryItem(
                    id=item.id,
                    namespace=item.namespace,
                    key=item.key,
                    content=item.content,
                    score=round(score, 4),
                    confidence=float(item.confidence),
                    source_reliability=float(item.source_reliability),
                    ts=item.ts,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return RetrievedMemorySet(query=query, items=scored[: max(1, int(top_k))])

    @staticmethod
    def _recency_score(*, now: datetime, ts_text: str) -> float:
        try:
            ts = datetime.fromisoformat(ts_text)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.3
        age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        if age_hours < 1:
            return 1.0
        if age_hours < 24:
            return 0.8
        if age_hours < 24 * 3:
            return 0.6
        if age_hours < 24 * 14:
            return 0.4
        return 0.2


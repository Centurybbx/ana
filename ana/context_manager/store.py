from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ContextMemoryItem:
    id: str
    namespace: str
    key: str
    content: str
    confidence: float
    source_reliability: float
    ts: str
    source_trace_id: str
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "key": self.key,
            "content": self.content,
            "confidence": self.confidence,
            "source_reliability": self.source_reliability,
            "ts": self.ts,
            "source_trace_id": self.source_trace_id,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextMemoryItem":
        return cls(
            id=str(payload.get("id", "")),
            namespace=str(payload.get("namespace", "")).strip(),
            key=str(payload.get("key", "")).strip(),
            content=str(payload.get("content", "")),
            confidence=float(payload.get("confidence", 0.0)),
            source_reliability=float(payload.get("source_reliability", 0.0)),
            ts=str(payload.get("ts", "")) or _utc_now(),
            source_trace_id=str(payload.get("source_trace_id", "")),
            active=bool(payload.get("active", True)),
        )


class ContextMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append_item(
        self,
        *,
        namespace: str,
        content: str,
        confidence: float,
        source_reliability: float,
        ts: str | None = None,
        key: str = "",
        source_trace_id: str = "",
    ) -> ContextMemoryItem:
        item = ContextMemoryItem(
            id=uuid4().hex,
            namespace=namespace.strip(),
            key=key.strip(),
            content=content.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
            source_reliability=max(0.0, min(1.0, float(source_reliability))),
            ts=(ts or _utc_now()).strip(),
            source_trace_id=source_trace_id.strip(),
            active=True,
        )
        self._append_op({"op": "put", "item": item.to_dict()})
        return item

    def upsert(
        self,
        *,
        namespace: str,
        key: str,
        content: str,
        confidence: float,
        source_reliability: float,
        source_trace_id: str = "",
        ts: str | None = None,
    ) -> tuple[ContextMemoryItem, list[str]]:
        expired_ids: list[str] = []
        for old in self.read_items(active_only=True, namespace=namespace, key=key):
            self._append_op({"op": "expire", "id": old.id, "reason": "superseded"})
            expired_ids.append(old.id)
        current = self.append_item(
            namespace=namespace,
            key=key,
            content=content,
            confidence=confidence,
            source_reliability=source_reliability,
            source_trace_id=source_trace_id,
            ts=ts,
        )
        return current, expired_ids

    def read_items(
        self,
        *,
        active_only: bool = True,
        namespace: str | None = None,
        key: str | None = None,
    ) -> list[ContextMemoryItem]:
        states: dict[str, ContextMemoryItem] = {}
        active: dict[str, bool] = {}
        for op in self._iter_ops():
            op_name = str(op.get("op", ""))
            if op_name == "put" and isinstance(op.get("item"), dict):
                item = ContextMemoryItem.from_dict(op["item"])
                states[item.id] = item
                active[item.id] = bool(item.active)
            elif op_name == "expire":
                target = str(op.get("id", "")).strip()
                if target:
                    active[target] = False

        rows: list[ContextMemoryItem] = []
        for item_id, item in states.items():
            is_active = bool(active.get(item_id, item.active))
            if active_only and not is_active:
                continue
            if namespace is not None and item.namespace != namespace:
                continue
            if key is not None and item.key != key:
                continue
            rows.append(
                ContextMemoryItem(
                    id=item.id,
                    namespace=item.namespace,
                    key=item.key,
                    content=item.content,
                    confidence=item.confidence,
                    source_reliability=item.source_reliability,
                    ts=item.ts,
                    source_trace_id=item.source_trace_id,
                    active=is_active,
                )
            )
        rows.sort(key=lambda item: item.ts, reverse=True)
        return rows

    def expire_before(self, *, cutoff_ts: str) -> int:
        expired = 0
        for item in self.read_items(active_only=True):
            if item.ts < cutoff_ts:
                self._append_op({"op": "expire", "id": item.id, "reason": "ttl"})
                expired += 1
        return expired

    def _append_op(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _iter_ops(self):
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload


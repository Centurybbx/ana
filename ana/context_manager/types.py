from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedMemoryItem:
    id: str
    namespace: str
    key: str
    content: str
    score: float
    confidence: float
    source_reliability: float
    ts: str


@dataclass(frozen=True)
class RetrievedMemorySet:
    query: str
    items: list[RetrievedMemoryItem] = field(default_factory=list)


@dataclass(frozen=True)
class CompactionOperation:
    kind: str
    reason: str
    saved_tokens: int
    before_tokens: int
    after_tokens: int


@dataclass(frozen=True)
class EditedContextResult:
    messages: list[dict[str, Any]]
    operations: list[CompactionOperation] = field(default_factory=list)


@dataclass(frozen=True)
class ContextBuildReport:
    selected_messages: int
    retrieved_memory_count: int
    token_estimate: int
    compaction_ops: int
    soft_budget_tokens: int
    hard_budget_tokens: int


@dataclass(frozen=True)
class WorkingSet:
    messages: list[dict[str, Any]]
    token_estimate: int
    report: ContextBuildReport
    retrieved: RetrievedMemorySet
    compaction_operations: list[CompactionOperation] = field(default_factory=list)


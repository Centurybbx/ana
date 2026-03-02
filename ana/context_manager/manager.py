from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ana.context_manager.editor import ContextEditor
from ana.context_manager.retriever import MemoryRetriever
from ana.context_manager.store import ContextMemoryStore
from ana.context_manager.types import CompactionOperation, ContextBuildReport, RetrievedMemorySet, WorkingSet
from ana.providers.base import LLMProvider


class ContextManager:
    def __init__(
        self,
        *,
        token_budget: int,
        soft_limit_ratio: float,
        hard_limit_ratio: float,
        recent_turns_window: int,
        memory_top_k: int,
        event_schema_version: str,
        memory_store: ContextMemoryStore,
        trace_sink: callable | None = None,
        model_assisted_compaction: bool = False,
        provider: LLMProvider | None = None,
        episode_compact_trigger_tokens: int = 12_000,
    ) -> None:
        self.token_budget = max(256, int(token_budget))
        self.soft_limit_ratio = max(0.1, min(0.95, float(soft_limit_ratio)))
        self.hard_limit_ratio = max(self.soft_limit_ratio, min(0.99, float(hard_limit_ratio)))
        self.recent_turns_window = max(1, int(recent_turns_window))
        self.memory_top_k = max(1, int(memory_top_k))
        self.event_schema_version = str(event_schema_version).strip() or "v1"
        self.memory_store = memory_store
        self.trace_sink = trace_sink
        self.model_assisted_compaction = bool(model_assisted_compaction)
        self.provider = provider
        self.episode_compact_trigger_tokens = max(1000, int(episode_compact_trigger_tokens))
        self.retriever = MemoryRetriever(store=memory_store)
        self.editor = ContextEditor()

    async def build_working_set(
        self,
        *,
        session_messages: list[dict[str, Any]],
        user_input: str,
        trace_context: dict[str, Any],
    ) -> WorkingSet:
        self._emit("context_build_started", trace_context, user_text_excerpt=user_input)

        total_tokens = self.editor.estimate_tokens(session_messages)
        if total_tokens >= self.episode_compact_trigger_tokens:
            self._emit(
                "context_anchor_created",
                trace_context,
                reason="episode_trigger_reached",
                total_tokens=total_tokens,
            )

        message_window = max(1, self.recent_turns_window * 2)
        selected_messages = list(session_messages[-message_window:])

        retrieved = self.retriever.retrieve(query=user_input, top_k=self.memory_top_k)
        self._emit(
            "context_memory_retrieved",
            trace_context,
            retrieved_memory_count=len(retrieved.items),
            retrieved_memory_ids=[item.id for item in retrieved.items],
        )
        if retrieved.items:
            memory_lines = [f"- ({item.namespace}) {item.content}" for item in retrieved.items]
            selected_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Relevant memory:\n" + "\n".join(memory_lines),
                },
            )

        soft_budget = max(1, int(self.token_budget * self.soft_limit_ratio))
        hard_budget = max(soft_budget + 1, int(self.token_budget * self.hard_limit_ratio))
        token_estimate = self.editor.estimate_tokens(selected_messages)
        compaction_ops = []

        if token_estimate > soft_budget:
            if self.model_assisted_compaction:
                try:
                    selected_messages, op = await self._apply_model_assisted_compaction(selected_messages)
                    if op is not None:
                        compaction_ops.append(op)
                        self._emit(
                            "context_edit_applied",
                            trace_context,
                            kind=op.kind,
                            reason=op.reason,
                            saved_tokens=op.saved_tokens,
                        )
                except Exception as exc:
                    self._emit(
                        "context_edit_applied",
                        trace_context,
                        kind="model_assisted_fallback",
                        reason=f"model_assisted_failed:{type(exc).__name__}",
                        saved_tokens=0,
                    )

            edited = self.editor.compact(
                messages=selected_messages,
                soft_budget_tokens=soft_budget,
                hard_budget_tokens=hard_budget,
            )
            selected_messages = edited.messages
            for op in edited.operations:
                compaction_ops.append(op)
                self._emit(
                    "context_edit_applied",
                    trace_context,
                    kind=op.kind,
                    reason=op.reason,
                    saved_tokens=op.saved_tokens,
                )

        token_estimate = self.editor.estimate_tokens(selected_messages)
        if token_estimate > hard_budget:
            self._emit(
                "context_budget_guard_triggered",
                trace_context,
                token_estimate=token_estimate,
                hard_budget_tokens=hard_budget,
            )

        report = ContextBuildReport(
            selected_messages=len(selected_messages),
            retrieved_memory_count=len(retrieved.items),
            token_estimate=token_estimate,
            compaction_ops=len(compaction_ops),
            soft_budget_tokens=soft_budget,
            hard_budget_tokens=hard_budget,
        )
        self._emit(
            "context_build_finished",
            trace_context,
            selected_messages=report.selected_messages,
            retrieved_memory_count=report.retrieved_memory_count,
            token_estimate=report.token_estimate,
            compaction_ops=report.compaction_ops,
            soft_budget_tokens=report.soft_budget_tokens,
            hard_budget_tokens=report.hard_budget_tokens,
        )
        return WorkingSet(
            messages=selected_messages,
            token_estimate=token_estimate,
            report=report,
            retrieved=retrieved,
            compaction_operations=list(compaction_ops),
        )

    async def record_post_turn(
        self,
        *,
        session_id: str,
        trace_id: str,
        turn_index: int,
        user_input: str,
        assistant_text: str,
        tool_events: list[dict[str, Any]],
    ) -> None:
        trace_context = {
            "session_id": session_id,
            "trace_id": trace_id,
            "turn_index": turn_index,
            "span_id": f"{trace_id}:post_turn",
            "parent_span_id": trace_id,
        }
        content = assistant_text.strip()
        if content:
            episode = self.memory_store.append_item(
                namespace="episode",
                key=f"turn_{turn_index}",
                content=f"user={user_input.strip()} | assistant={content[:800]}",
                confidence=0.75,
                source_reliability=0.9,
                source_trace_id=trace_id,
            )
            self._emit(
                "memory_written",
                trace_context,
                namespace=episode.namespace,
                key=episode.key,
                memory_id=episode.id,
                source_trace_id=trace_id,
            )

        lowered = content.lower()
        if "preference:" in lowered:
            preference_text = content.split(":", 1)[1].strip() if ":" in content else content
            current, superseded = self.memory_store.upsert(
                namespace="semantic",
                key="user_preference",
                content=preference_text,
                confidence=0.85,
                source_reliability=0.85,
                source_trace_id=trace_id,
            )
            self._emit(
                "memory_written",
                trace_context,
                namespace=current.namespace,
                key=current.key,
                memory_id=current.id,
                source_trace_id=trace_id,
            )
            self._emit(
                "memory_promoted",
                trace_context,
                namespace=current.namespace,
                key=current.key,
                memory_id=current.id,
                superseded=superseded,
            )

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        expired_count = self.memory_store.expire_before(cutoff_ts=cutoff)
        if expired_count > 0:
            self._emit(
                "memory_expired",
                trace_context,
                reason="ttl",
                expired_count=expired_count,
            )

    async def _apply_model_assisted_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], CompactionOperation | None]:
        if self.provider is None:
            raise RuntimeError("no provider for model-assisted compaction")
        target_idx = -1
        target_text = ""
        for idx, item in enumerate(messages):
            if str(item.get("role", "")) != "tool":
                continue
            content = str(item.get("content", ""))
            if len(content) < 600:
                continue
            target_idx = idx
            target_text = content
            break
        if target_idx < 0:
            return messages, None

        prompt_messages = [
            {"role": "system", "content": "Summarize the following tool output in <= 220 chars."},
            {"role": "user", "content": target_text[:3000]},
        ]
        response = await self.provider.complete(messages=prompt_messages, tools=[])
        summary = (response.content or "").strip()
        if not summary:
            raise RuntimeError("empty model-assisted summary")
        cloned = list(messages)
        before = self.editor.estimate_tokens(cloned)
        cloned[target_idx] = dict(cloned[target_idx])
        cloned[target_idx]["content"] = summary[:220] + "\n...[model-assisted summary]..."
        after = self.editor.estimate_tokens(cloned)
        op = CompactionOperation(
            kind="model_assisted_summary",
            reason="model_assisted",
            saved_tokens=max(0, before - after),
            before_tokens=before,
            after_tokens=after,
        )
        return cloned, op

    def _emit(self, event_type: str, trace_context: dict[str, Any], **fields: Any) -> None:
        if self.trace_sink is None:
            return
        span_id = str(trace_context.get("span_id") or "")
        payload: dict[str, Any] = {
            "event": event_type,
            "event_type": event_type,
            "schema_version": self.event_schema_version,
            "session_id": trace_context.get("session_id"),
            "trace_id": trace_context.get("trace_id"),
            "turn_index": trace_context.get("turn_index"),
            "span_id": f"{span_id}:context:{event_type}" if span_id else f"context:{event_type}",
            "parent_span_id": span_id or trace_context.get("parent_span_id"),
        }
        payload.update({key: value for key, value in fields.items() if value is not None})
        self.trace_sink(payload)

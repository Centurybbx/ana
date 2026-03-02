from __future__ import annotations

from copy import deepcopy
from typing import Any

from ana.context_manager.types import CompactionOperation, EditedContextResult


class ContextEditor:
    def compact(
        self,
        *,
        messages: list[dict[str, Any]],
        soft_budget_tokens: int,
        hard_budget_tokens: int,
    ) -> EditedContextResult:
        working = deepcopy(messages)
        operations: list[CompactionOperation] = []
        token_estimate = self.estimate_tokens(working)
        if token_estimate <= soft_budget_tokens:
            return EditedContextResult(messages=working, operations=operations)

        while token_estimate > hard_budget_tokens:
            op = self._compact_long_tool_message(working)
            if op is None:
                op = self._compact_old_assistant_message(working)
            if op is None:
                op = self._drop_oldest_message(working)
            if op is None:
                break
            operations.append(op)
            token_estimate = self.estimate_tokens(working)
            if len(operations) > 64:
                break

        return EditedContextResult(messages=working, operations=operations)

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        total_chars = 0
        for item in messages:
            total_chars += len(str(item.get("content", "")))
        return max(1, total_chars // 4) if total_chars > 0 else 0

    def _compact_long_tool_message(self, messages: list[dict[str, Any]]) -> CompactionOperation | None:
        for idx, msg in enumerate(messages):
            if str(msg.get("role", "")) != "tool":
                continue
            content = str(msg.get("content", ""))
            if len(content) < 600:
                continue
            before = self.estimate_tokens(messages)
            summary = content[:280].strip() + "\n...[long tool output summarized]..."
            messages[idx]["content"] = summary
            after = self.estimate_tokens(messages)
            return CompactionOperation(
                kind="long_tool_summary",
                reason="tool_output_too_long",
                saved_tokens=max(0, before - after),
                before_tokens=before,
                after_tokens=after,
            )
        return None

    def _compact_old_assistant_message(self, messages: list[dict[str, Any]]) -> CompactionOperation | None:
        assistant_indexes = [idx for idx, item in enumerate(messages) if str(item.get("role", "")) == "assistant"]
        if not assistant_indexes:
            return None
        for idx in assistant_indexes[:-1]:
            content = str(messages[idx].get("content", ""))
            if len(content) < 400:
                continue
            before = self.estimate_tokens(messages)
            messages[idx]["content"] = content[:220].strip() + "\n...[assistant summary]..."
            after = self.estimate_tokens(messages)
            return CompactionOperation(
                kind="assistant_fold",
                reason="old_assistant_history",
                saved_tokens=max(0, before - after),
                before_tokens=before,
                after_tokens=after,
            )
        return None

    def _drop_oldest_message(self, messages: list[dict[str, Any]]) -> CompactionOperation | None:
        if len(messages) <= 2:
            return None
        # Preserve the latest user question and at least one prior context message.
        preserve_user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if str(messages[idx].get("role", "")) == "user":
                preserve_user_idx = idx
                break
        for idx, msg in enumerate(messages):
            if idx == preserve_user_idx:
                continue
            before = self.estimate_tokens(messages)
            dropped_role = str(msg.get("role", "unknown"))
            messages.pop(idx)
            after = self.estimate_tokens(messages)
            return CompactionOperation(
                kind="drop_old_message",
                reason=f"drop_{dropped_role}",
                saved_tokens=max(0, before - after),
                before_tokens=before,
                after_tokens=after,
            )
        return None


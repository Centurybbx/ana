from __future__ import annotations

from typing import Any

from ana.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break

        content = f"[mock] {last_user}" if last_user else "[mock] ready"
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        completion_chars = len(content)
        prompt_tokens = max(1, prompt_chars // 4) if prompt_chars > 0 else 0
        completion_tokens = max(1, completion_chars // 4) if completion_chars > 0 else 0
        total_tokens = prompt_tokens + completion_tokens
        return LLMResponse(
            content=content,
            tool_calls=[],
            raw_message={"role": "assistant", "content": content},
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate": 0.0,
            },
        )

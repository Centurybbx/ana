from __future__ import annotations

from typing import Any

from aha.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break

        content = f"[mock] {last_user}" if last_user else "[mock] ready"
        return LLMResponse(
            content=content,
            tool_calls=[],
            raw_message={"role": "assistant", "content": content},
        )


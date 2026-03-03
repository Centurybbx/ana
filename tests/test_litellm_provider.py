from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import ana.providers.litellm_provider as litellm_provider_module
from ana.providers.litellm_provider import LiteLLMProvider


def _fake_response(content: str = "ok") -> Any:
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage={})


def test_litellm_provider_merges_multiple_system_messages_for_minimax_model() -> None:
    async def _run() -> None:
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _fake_response()

        original = litellm_provider_module.acompletion
        litellm_provider_module.acompletion = _fake_acompletion
        try:
            provider = LiteLLMProvider(model="minimax/MiniMax-M2.5")
            messages = [
                {"role": "system", "content": "base instructions"},
                {"role": "system", "content": "Relevant memory:\n- keep answer concise"},
                {"role": "user", "content": "hello"},
            ]
            _ = await provider.complete(messages=messages, tools=[])
        finally:
            litellm_provider_module.acompletion = original

        normalized = captured["messages"]
        system_messages = [item for item in normalized if item.get("role") == "system"]
        assert len(system_messages) == 1
        assert "base instructions" in str(system_messages[0].get("content", ""))
        assert "Relevant memory" in str(system_messages[0].get("content", ""))
        assert any(item.get("role") == "user" for item in normalized)

    asyncio.run(_run())


def test_litellm_provider_keeps_messages_for_non_minimax_model() -> None:
    async def _run() -> None:
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _fake_response()

        original = litellm_provider_module.acompletion
        litellm_provider_module.acompletion = _fake_acompletion
        try:
            provider = LiteLLMProvider(model="openai/gpt-4o-mini")
            messages = [
                {"role": "system", "content": "s1"},
                {"role": "system", "content": "s2"},
                {"role": "user", "content": "hello"},
            ]
            _ = await provider.complete(messages=messages, tools=[])
        finally:
            litellm_provider_module.acompletion = original

        assert captured["messages"] == messages

    asyncio.run(_run())

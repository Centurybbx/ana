from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from litellm import acompletion

from ana.providers.base import LLMProvider, LLMResponse, LLMToolCall
from ana.runtime_log import log_event


class LiteLLMProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout_seconds: int = 60,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.api_base = api_base
        self.api_version = api_version
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.extra_headers = extra_headers or {}
        self.logger = logger or logging.getLogger("ana.provider.litellm")

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        api_key = self._resolve_api_key()
        normalized_messages = self._normalize_messages_for_model(messages)
        model_family = self._model_family(self.model)
        system_count_before = self._count_role(messages, "system")
        system_count_after = self._count_role(normalized_messages, "system")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "tools": tools,
            "tool_choice": "auto",
            "timeout": self.timeout_seconds,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["base_url"] = self.api_base
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        log_event(
            self.logger,
            "llm_call_start",
            level=logging.DEBUG,
            model=self.model,
            has_api_key=bool(api_key),
            has_base_url=bool(self.api_base),
            has_api_version=bool(self.api_version),
            model_family=model_family,
            messages_count=len(normalized_messages),
            system_count_before=system_count_before,
            system_count_after=system_count_after,
            tools_count=len(tools),
            timeout_seconds=self.timeout_seconds,
        )
        started = time.perf_counter()
        try:
            response = await acompletion(**kwargs)
        except Exception as exc:
            log_event(
                self.logger,
                "llm_call_error",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self.logger.debug("litellm exception detail", exc_info=True)
            raise

        message = response.choices[0].message
        message_content = self._coerce_content(getattr(message, "content", ""))

        tool_calls: list[LLMToolCall] = []
        for call in self._iter_tool_calls(message):
            call_id = self._field(call, "id") or ""
            function = self._field(call, "function") or {}
            name = self._field(function, "name") or ""
            arguments = self._field(function, "arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                LLMToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        raw_message = {
            "role": "assistant",
            "content": message_content,
        }
        raw_tool_calls = self._iter_tool_calls(message)
        if raw_tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": self._field(call, "id"),
                    "type": "function",
                    "function": {
                        "name": self._field(self._field(call, "function") or {}, "name"),
                        "arguments": self._field(self._field(call, "function") or {}, "arguments"),
                    },
                }
                for call in raw_tool_calls
            ]

        latency_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            self.logger,
            "llm_call_ok",
            level=logging.INFO,
            latency_ms=latency_ms,
            tool_calls_count=len(tool_calls),
            content_len=len(message_content),
        )
        usage = self._extract_usage(response)

        return LLMResponse(
            content=message_content,
            tool_calls=tool_calls,
            raw_message=raw_message,
            usage=usage,
        )

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if not self.api_key_env:
            return None
        env_value = os.getenv(self.api_key_env)
        if env_value:
            return env_value
        # Backward compatibility: users sometimes paste raw key into api_key_env.
        if not self._looks_like_env_name(self.api_key_env) and self._looks_like_secret(self.api_key_env):
            return self.api_key_env
        return None

    def _normalize_messages_for_model(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [dict(item) for item in messages]
        if not self._is_minimax_model(self.model):
            return normalized
        system_indexes = [idx for idx, item in enumerate(normalized) if str(item.get("role", "")) == "system"]
        if len(system_indexes) <= 1:
            return normalized
        first_index = system_indexes[0]
        merged_system = dict(normalized[first_index])
        merged_parts = [
            str(normalized[idx].get("content", "")).strip()
            for idx in system_indexes
            if str(normalized[idx].get("content", "")).strip()
        ]
        if merged_parts:
            merged_system["content"] = "\n\n".join(merged_parts)
        system_index_set = set(system_indexes)
        compacted: list[dict[str, Any]] = []
        for idx, item in enumerate(normalized):
            if idx == first_index:
                compacted.append(merged_system)
                continue
            if idx in system_index_set:
                continue
            compacted.append(item)
        return compacted

    @staticmethod
    def _is_minimax_model(model: str) -> bool:
        return "minimax" in str(model or "").strip().lower()

    @staticmethod
    def _model_family(model: str) -> str:
        model_name = str(model or "").strip().lower()
        if not model_name:
            return ""
        if "minimax" in model_name:
            return "minimax"
        return model_name.split("/", 1)[0]

    @staticmethod
    def _count_role(messages: list[dict[str, Any]], role: str) -> int:
        role_name = str(role or "")
        return sum(1 for item in messages if str(item.get("role", "")) == role_name)

    @staticmethod
    def _iter_tool_calls(message: Any) -> list[Any]:
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls")
        if not calls:
            return []
        return list(calls)

    @staticmethod
    def _field(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _looks_like_env_name(value: str) -> bool:
        return bool(re.fullmatch(r"[A-Z_][A-Z0-9_]*", value))

    @staticmethod
    def _looks_like_secret(value: str) -> bool:
        if len(value) < 20:
            return False
        return any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}

        def _field(source: Any, key: str) -> Any:
            if isinstance(source, dict):
                return source.get(key)
            return getattr(source, key, None)

        prompt_tokens = _field(usage, "prompt_tokens")
        completion_tokens = _field(usage, "completion_tokens")
        total_tokens = _field(usage, "total_tokens")
        cost_estimate = _field(usage, "cost") or _field(usage, "cost_estimate")
        try:
            prompt_tokens = int(prompt_tokens or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        try:
            completion_tokens = int(completion_tokens or 0)
        except (TypeError, ValueError):
            completion_tokens = 0
        try:
            total_tokens = int(total_tokens or 0)
        except (TypeError, ValueError):
            total_tokens = 0
        try:
            cost_estimate = float(cost_estimate or 0.0)
        except (TypeError, ValueError):
            cost_estimate = 0.0

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_estimate": cost_estimate,
        }

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ana.core.context import ContextInput, ContextWeaver
from ana.core.session import RuntimeSessionState, Session
from ana.providers.base import LLMProvider
from ana.runtime_log import log_event, session_context, step_context, tool_context
from ana.tools.base import ToolResult
from ana.tools.memory import MemoryStore
from ana.tools.skill_resolution import resolve_skill_views
from ana.tools.runner import ToolRunner

ConfirmFn = Callable[[str], Awaitable[bool]]


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tool_runner: ToolRunner,
        memory: MemoryStore,
        context_weaver: ContextWeaver,
        tool_names: list[str],
        workspace: str,
        skills_dir: str | None = None,
        skills_local: str | None = None,
        max_steps: int = 30,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.tool_runner = tool_runner
        self.memory = memory
        self.context_weaver = context_weaver
        self.tool_names = tool_names
        self.workspace = workspace
        self.skills_dir = Path(skills_dir).resolve() if skills_dir else None
        self.skills_local = Path(skills_local).resolve() if skills_local else None
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger("ana.loop")

    async def run_turn(
        self,
        session: Session,
        user_input: str,
        session_state: RuntimeSessionState,
        confirm: ConfirmFn,
    ) -> str:
            session_id = session.session_id
            with session_context(session_id):
                if not self._has_same_tail_user_message(session.messages, user_input):
                    session.messages.append({"role": "user", "content": user_input})
                turn_index = self._turn_index(session.messages)
            trace_id = uuid4().hex
            turn_span_id = trace_id
            memory_text = self.memory.read_memory()
            active_skills = self._active_skills()
            session.active_skills = active_skills
            session_state.active_skills = {
                str(item.get("name")): item
                for item in active_skills
                if str(item.get("name", "")).strip()
            }
            system_prompt = self.context_weaver.build(
                ContextInput(
                    workspace=self._workspace_path(),
                    memory_text=memory_text,
                    active_skills=active_skills,
                    tool_names=self.tool_names,
                )
            )
            messages = [{"role": "system", "content": system_prompt}] + list(session.messages)
            log_event(
                self.logger,
                "turn_start",
                level=logging.INFO,
                user_len=len(user_input),
                message_count=len(messages),
                tool_count=len(self.tool_names),
            )
            self.memory.append_trace(
                {
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "turn_index": turn_index,
                    "event": "turn_start",
                    "event_type": "turn_start",
                    "user_text_excerpt": user_input,
                    "span_id": turn_span_id,
                    "parent_span_id": None,
                }
            )

            for step in range(1, self.max_steps + 1):
                with step_context(step):
                    step_span_id = f"{trace_id}:step:{step}"
                    request_started = time.perf_counter()
                    log_event(
                        self.logger,
                        "llm_request",
                        level=logging.DEBUG,
                        step=step,
                        message_count=len(messages),
                        tools_count=len(self.tool_names),
                    )
                    self.memory.append_trace(
                        {
                            "session_id": session_id,
                            "trace_id": trace_id,
                            "turn_index": turn_index,
                            "step": step,
                            "event": "llm_request",
                            "event_type": "llm_request",
                            "message_count": len(messages),
                            "tools_count": len(self.tool_names),
                            "span_id": step_span_id,
                            "parent_span_id": turn_span_id,
                        }
                    )
                    response = await self.provider.complete(messages=messages, tools=self._tool_schemas())
                    latency_ms = int((time.perf_counter() - request_started) * 1000)
                    usage = self._extract_usage(response)
                    log_event(
                        self.logger,
                        "llm_response",
                        level=logging.INFO,
                        step=step,
                        latency_ms=latency_ms,
                        tool_calls_count=len(response.tool_calls),
                        assistant_len=len(response.content or ""),
                    )
                    self.memory.append_trace(
                        {
                            "session_id": session_id,
                            "trace_id": trace_id,
                            "turn_index": turn_index,
                            "step": step,
                            "event": "llm_response",
                            "event_type": "llm_response",
                            "latency_ms": latency_ms,
                            "tool_calls_count": len(response.tool_calls),
                            "assistant_len": len(response.content or ""),
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "total_tokens": usage["total_tokens"],
                            "cost_estimate": usage["cost_estimate"],
                            "span_id": step_span_id,
                            "parent_span_id": turn_span_id,
                        }
                    )

                    if not response.tool_calls:
                        final_text = response.content or ""
                        messages.append({"role": "assistant", "content": final_text})
                        session.messages.append({"role": "assistant", "content": final_text})
                        self.memory.append_trace(
                            {
                                "session_id": session_id,
                                "trace_id": trace_id,
                                "turn_index": turn_index,
                                "event": "assistant_final",
                                "event_type": "assistant_final",
                                "step": step,
                                "observation": final_text,
                                "span_id": f"{trace_id}:final",
                                "parent_span_id": turn_span_id,
                            }
                        )
                        log_event(
                            self.logger,
                            "turn_final",
                            level=logging.INFO,
                            step=step,
                            final_len=len(final_text),
                        )
                        return final_text

                    messages.append(response.raw_message)

                    for tool_pos, tool_call in enumerate(response.tool_calls, start=1):
                        tool_call_id = (tool_call.call_id or f"call-{tool_pos}").strip() or f"call-{tool_pos}"
                        tool_span_id = f"{trace_id}:step:{step}:tool:{tool_call_id}"
                        with tool_context(tool_call.name):
                            source_skill = self._extract_source_skill(tool_call.arguments)
                            self.memory.append_trace(
                                {
                                    "session_id": session_id,
                                    "trace_id": trace_id,
                                    "turn_index": turn_index,
                                    "event": "tool_call",
                                    "event_type": "tool_call",
                                    "step": step,
                                    "tool": tool_call.name,
                                    "args": tool_call.arguments,
                                    "source_skill": source_skill,
                                    "span_id": tool_span_id,
                                    "parent_span_id": step_span_id,
                                }
                            )
                            log_event(
                                self.logger,
                                "tool_dispatch",
                                level=logging.DEBUG,
                                step=step,
                                tool_call=tool_call.name,
                                source_skill=source_skill or "",
                            )
                            try:
                                result = await self.tool_runner.run(
                                    tool_call.name,
                                    tool_call.arguments,
                                    session_state=session_state,
                                    confirm=confirm,
                                    trace_context={
                                        "session_id": session_id,
                                        "trace_id": trace_id,
                                        "turn_index": turn_index,
                                        "step": step,
                                        "tool": tool_call.name,
                                        "span_id": tool_span_id,
                                        "parent_span_id": step_span_id,
                                    },
                                )
                            except Exception as exc:
                                self.logger.error(
                                    "Unhandled tool exception for %s: %s",
                                    tool_call.name,
                                    exc,
                                    exc_info=True,
                                )
                                result = ToolResult(
                                    ok=False,
                                    data=f"Internal error running {tool_call.name}: {exc}",
                                    warnings=["unhandled_exception"],
                                )
                            tool_payload = {
                                "ok": result.ok,
                                "data": result.data,
                                "warnings": result.warnings,
                                "redactions": result.redactions,
                                # Deprecated compatibility keys.
                                "output": result.output,
                                "meta": result.meta,
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.call_id,
                                    "name": tool_call.name,
                                    "content": json.dumps(tool_payload, ensure_ascii=False),
                                }
                            )
                            self.memory.append_trace(
                                {
                                    "session_id": session_id,
                                    "trace_id": trace_id,
                                    "turn_index": turn_index,
                                    "event": "tool_result",
                                    "event_type": "tool_result",
                                    "step": step,
                                    "tool": tool_call.name,
                                    "source_skill": source_skill,
                                    "status": "ok" if result.ok else "error",
                                    "latency_ms": self._extract_tool_latency_ms(result),
                                    "observation": result.data or "",
                                    "span_id": tool_span_id,
                                    "parent_span_id": step_span_id,
                                }
                            )

            exhausted = "Reached max steps without completing task."
            session.messages.append({"role": "assistant", "content": exhausted})
            self.memory.append_trace(
                {
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "turn_index": turn_index,
                    "event": "max_steps_exhausted",
                    "event_type": "max_steps_exhausted",
                    "observation": exhausted,
                    "span_id": f"{trace_id}:max_steps",
                    "parent_span_id": turn_span_id,
                }
            )
            log_event(self.logger, "max_steps_exhausted", level=logging.WARNING, max_steps=self.max_steps)
            return exhausted

    def _tool_schemas(self) -> list[dict]:
        return [self.tool_runner.registry.get(name).schema() for name in self.tool_names]

    def _workspace_path(self):
        return Path(self.workspace)

    def _active_skills(self) -> list[dict]:
        views = resolve_skill_views(skills_dir=self.skills_dir, skills_local_dir=self.skills_local)
        return views["effective"]

    @staticmethod
    def _extract_source_skill(args: dict | None) -> str | None:
        if not isinstance(args, dict):
            return None
        source_skill = str(args.get("_source_skill") or args.get("source_skill") or "").strip()
        return source_skill or None

    @staticmethod
    def _turn_index(messages: list[dict]) -> int:
        return sum(1 for item in messages if str(item.get("role")) == "user")

    @staticmethod
    def _has_same_tail_user_message(messages: list[dict], user_input: str) -> bool:
        if not messages:
            return False
        last = messages[-1]
        if str(last.get("role", "")) != "user":
            return False
        return str(last.get("content", "")) == str(user_input)

    @staticmethod
    def _extract_tool_latency_ms(result: ToolResult) -> int | None:
        raw = result.meta.get("duration_ms")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, float | int]:
        raw_usage = getattr(response, "usage", {}) if response is not None else {}
        if not isinstance(raw_usage, dict):
            raw_usage = {}

        def _as_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return {
            "prompt_tokens": _as_int(raw_usage.get("prompt_tokens")),
            "completion_tokens": _as_int(raw_usage.get("completion_tokens")),
            "total_tokens": _as_int(raw_usage.get("total_tokens")),
            "cost_estimate": _as_float(raw_usage.get("cost_estimate")),
        }

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable

from aha.core.context import ContextInput, ContextWeaver
from aha.core.session import RuntimeSessionState, Session
from aha.providers.base import LLMProvider
from aha.runtime_log import log_event, session_context, step_context, tool_context
from aha.tools.memory import MemoryStore
from aha.tools.runner import ToolRunner

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
        max_steps: int = 30,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.tool_runner = tool_runner
        self.memory = memory
        self.context_weaver = context_weaver
        self.tool_names = tool_names
        self.workspace = workspace
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger("aha.loop")

    async def run_turn(
        self,
        session: Session,
        user_input: str,
        session_state: RuntimeSessionState,
        confirm: ConfirmFn,
    ) -> str:
        session_id = session.session_id
        with session_context(session_id):
            session.messages.append({"role": "user", "content": user_input})
            memory_text = self.memory.read_memory()
            system_prompt = self.context_weaver.build(
                ContextInput(
                    workspace=self._workspace_path(),
                    memory_text=memory_text,
                    active_skill_names=[],
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

            for step in range(1, self.max_steps + 1):
                with step_context(step):
                    request_started = time.perf_counter()
                    log_event(
                        self.logger,
                        "llm_request",
                        level=logging.DEBUG,
                        step=step,
                        message_count=len(messages),
                        tools_count=len(self.tool_names),
                    )
                    response = await self.provider.complete(messages=messages, tools=self._tool_schemas())
                    latency_ms = int((time.perf_counter() - request_started) * 1000)
                    log_event(
                        self.logger,
                        "llm_response",
                        level=logging.INFO,
                        step=step,
                        latency_ms=latency_ms,
                        tool_calls_count=len(response.tool_calls),
                        assistant_len=len(response.content or ""),
                    )

                    if not response.tool_calls:
                        final_text = response.content or ""
                        messages.append({"role": "assistant", "content": final_text})
                        session.messages.append({"role": "assistant", "content": final_text})
                        self.memory.append_trace(
                            {
                                "session_id": session_id,
                                "event": "assistant_final",
                                "step": step,
                                "observation": final_text,
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

                    for tool_call in response.tool_calls:
                        with tool_context(tool_call.name):
                            self.memory.append_trace(
                                {
                                    "session_id": session_id,
                                    "event": "tool_call",
                                    "step": step,
                                    "tool": tool_call.name,
                                    "args": tool_call.arguments,
                                }
                            )
                            log_event(
                                self.logger,
                                "tool_dispatch",
                                level=logging.DEBUG,
                                step=step,
                                tool_call=tool_call.name,
                            )
                            result = await self.tool_runner.run(
                                tool_call.name,
                                tool_call.arguments,
                                session_state=session_state,
                                confirm=confirm,
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
                                    "event": "tool_result",
                                    "step": step,
                                    "tool": tool_call.name,
                                    "status": "ok" if result.ok else "error",
                                    "observation": result.data or "",
                                }
                            )

            exhausted = "Reached max steps without completing task."
            session.messages.append({"role": "assistant", "content": exhausted})
            self.memory.append_trace(
                {"session_id": session_id, "event": "max_steps_exhausted", "observation": exhausted}
            )
            log_event(self.logger, "max_steps_exhausted", level=logging.WARNING, max_steps=self.max_steps)
            return exhausted

    def _tool_schemas(self) -> list[dict]:
        return [self.tool_runner.registry.get(name).schema() for name in self.tool_names]

    def _workspace_path(self):
        from pathlib import Path

        return Path(self.workspace)

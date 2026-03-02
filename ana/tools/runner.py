from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ana.core.session import RuntimeSessionState
from ana.runtime_log import log_event, redact_text
from ana.tools.base import ToolResult
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry

ConfirmFn = Callable[[str], Awaitable[bool]]
TraceSink = Callable[[dict[str, Any]], None]


class ToolRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy,
        logger: logging.Logger | None = None,
        trace_sink: TraceSink | None = None,
    ):
        self.registry = registry
        self.policy = policy
        self.logger = logger or logging.getLogger("ana.tools.runner")
        self.trace_sink = trace_sink

    async def run(
        self,
        tool_name: str,
        args: dict,
        session_state: RuntimeSessionState,
        confirm: ConfirmFn,
        trace_context: dict[str, Any] | None = None,
    ) -> ToolResult:
        decision = self.policy.precheck(tool_name, args, session_state=session_state)
        log_event(
            self.logger,
            "tool_precheck",
            level=logging.DEBUG,
            tool_name=tool_name,
            allowed=decision.allowed,
            requires_confirmation=decision.requires_confirmation,
            temporary_capability=decision.temporary_capability,
            reason=decision.reason,
            **self._summarize_args(tool_name, args),
        )
        if not decision.allowed:
            self._emit_trace(
                event_type="guardrail_tripwire",
                trace_context=trace_context,
                tool=tool_name,
                reason=decision.reason,
            )
            log_event(
                self.logger,
                "tool_precheck_blocked",
                level=logging.WARNING,
                tool_name=tool_name,
                reason=decision.reason,
            )
            return ToolResult(ok=False, data=f"blocked by policy: {decision.reason}", warnings=["policy_block"])

        temporary_grants: list[str] = []
        if decision.requires_confirmation:
            self._emit_trace(
                event_type="consent_request",
                trace_context=trace_context,
                tool=tool_name,
                reason=decision.reason,
            )
            log_event(
                self.logger,
                "tool_confirm_prompted",
                level=logging.INFO,
                tool_name=tool_name,
                temporary_capability=decision.temporary_capability,
            )
            approved = await confirm(decision.plan or f"{tool_name}({args})")
            if not approved:
                self._emit_trace(
                    event_type="consent_deny",
                    trace_context=trace_context,
                    tool=tool_name,
                    reason="user_denied",
                )
                log_event(self.logger, "tool_confirm_denied", level=logging.INFO, tool_name=tool_name)
                return ToolResult(ok=False, data="user rejected action", warnings=["user_rejected"])
            self._emit_trace(
                event_type="consent_grant",
                trace_context=trace_context,
                tool=tool_name,
                reason="user_approved",
            )
            log_event(self.logger, "tool_confirm_approved", level=logging.INFO, tool_name=tool_name)
            if decision.temporary_capability:
                session_state.capabilities.add(decision.temporary_capability)
                temporary_grants.append(decision.temporary_capability)

        tool = self.registry.get(tool_name)
        started = time.perf_counter()
        try:
            log_event(self.logger, "tool_run_start", level=logging.INFO, tool_name=tool_name)
            result = await tool.run(args)
            result.data = self.policy.postprocess(result.data or "")
            result.output = result.data
            duration_ms = int((time.perf_counter() - started) * 1000)
            result.meta = dict(result.meta or {})
            result.meta["duration_ms"] = duration_ms
            log_event(
                self.logger,
                "tool_run_end",
                level=logging.INFO,
                tool_name=tool_name,
                ok=result.ok,
                duration_ms=duration_ms,
                warnings_count=len(result.warnings),
                redactions_count=len(result.redactions),
                **self._safe_meta(result.meta),
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            log_event(
                self.logger,
                "tool_run_error",
                level=logging.ERROR,
                tool_name=tool_name,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return ToolResult(
                ok=False,
                data=f"Tool error ({type(exc).__name__}): {exc}",
                warnings=["tool_exception"],
                meta={"duration_ms": duration_ms},
            )
        finally:
            for capability in temporary_grants:
                session_state.capabilities.discard(capability)

    def _emit_trace(
        self,
        event_type: str,
        trace_context: dict[str, Any] | None,
        **fields: Any,
    ) -> None:
        if self.trace_sink is None:
            return
        payload: dict[str, Any] = {"event": event_type, "event_type": event_type}
        if trace_context:
            payload.update({key: value for key, value in trace_context.items() if value is not None})
        payload.update({key: value for key, value in fields.items() if value is not None})
        self.trace_sink(payload)

    @staticmethod
    def _safe_meta(meta: dict[str, Any]) -> dict[str, str]:
        safe: dict[str, str] = {}
        if not meta:
            return safe
        for key, value in meta.items():
            key_lower = key.lower()
            if any(token in key_lower for token in ("api_key", "token", "secret", "authorization")):
                safe[f"meta_{key}"] = "[REDACTED]"
            else:
                safe[f"meta_{key}"] = redact_text(str(value))
        return safe

    @staticmethod
    def _summarize_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        source_skill = str(args.get("_source_skill") or args.get("source_skill") or "").strip()
        if tool_name == "write_file":
            summary = {
                "path": args.get("path"),
                "mode": args.get("mode", "overwrite"),
                "content_len": len(str(args.get("content", ""))),
            }
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        if tool_name == "read_file":
            summary = {"path": args.get("path"), "max_chars": args.get("max_chars", 4000)}
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        if tool_name == "shell":
            summary = {"cmd": redact_text(str(args.get("cmd", ""))), "timeout_seconds": args.get("timeout_seconds")}
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        if tool_name == "web_fetch":
            summary = {"url": args.get("url"), "max_chars": args.get("max_chars", 5000)}
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        if tool_name == "web_search":
            query = str(args.get("query", ""))
            summary = {"query_len": len(query), "max_results": args.get("max_results", 5)}
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        if tool_name == "skill_manager":
            summary = {
                "action": args.get("action"),
                "name": args.get("name"),
                "skill_id": args.get("skill_id"),
                "target": args.get("target"),
            }
            if str(args.get("action", "")) in {"install", "generate"}:
                summary["content_len"] = len(str(args.get("content", "")))
            if source_skill:
                summary["source_skill"] = source_skill
            return summary
        summary = {"arg_keys": ",".join(sorted(args.keys()))}
        if source_skill:
            summary["source_skill"] = source_skill
        return summary

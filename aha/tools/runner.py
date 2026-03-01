from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aha.core.session import RuntimeSessionState
from aha.runtime_log import log_event, redact_text
from aha.tools.base import ToolResult
from aha.tools.policy import ToolPolicy
from aha.tools.registry import ToolRegistry

ConfirmFn = Callable[[str], Awaitable[bool]]


class ToolRunner:
    def __init__(self, registry: ToolRegistry, policy: ToolPolicy, logger: logging.Logger | None = None):
        self.registry = registry
        self.policy = policy
        self.logger = logger or logging.getLogger("aha.tools.runner")

    async def run(self, tool_name: str, args: dict, session_state: RuntimeSessionState, confirm: ConfirmFn) -> ToolResult:
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
            log_event(
                self.logger,
                "tool_confirm_prompted",
                level=logging.INFO,
                tool_name=tool_name,
                temporary_capability=decision.temporary_capability,
            )
            approved = await confirm(decision.plan or f"{tool_name}({args})")
            if not approved:
                log_event(self.logger, "tool_confirm_denied", level=logging.INFO, tool_name=tool_name)
                return ToolResult(ok=False, data="user rejected action", warnings=["user_rejected"])
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
            log_event(
                self.logger,
                "tool_run_end",
                level=logging.INFO,
                tool_name=tool_name,
                ok=result.ok,
                duration_ms=int((time.perf_counter() - started) * 1000),
                warnings_count=len(result.warnings),
                redactions_count=len(result.redactions),
                **self._safe_meta(result.meta),
            )
            return result
        except Exception as exc:
            log_event(
                self.logger,
                "tool_run_error",
                level=logging.ERROR,
                tool_name=tool_name,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            for capability in temporary_grants:
                session_state.capabilities.discard(capability)

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
        if tool_name == "write_file":
            return {
                "path": args.get("path"),
                "mode": args.get("mode", "overwrite"),
                "content_len": len(str(args.get("content", ""))),
            }
        if tool_name == "read_file":
            return {"path": args.get("path"), "max_chars": args.get("max_chars", 4000)}
        if tool_name == "shell":
            return {"cmd": redact_text(str(args.get("cmd", ""))), "timeout_seconds": args.get("timeout_seconds")}
        if tool_name == "web_fetch":
            return {"url": args.get("url"), "max_chars": args.get("max_chars", 5000)}
        if tool_name == "web_search":
            query = str(args.get("query", ""))
            return {"query_len": len(query), "max_results": args.get("max_results", 5)}
        if tool_name == "skill_manager":
            summary = {
                "action": args.get("action"),
                "name": args.get("name"),
            }
            if str(args.get("action", "")) == "install":
                summary["content_len"] = len(str(args.get("content", "")))
            return summary
        return {"arg_keys": ",".join(sorted(args.keys()))}

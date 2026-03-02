from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ana.stats import percentile


class MemoryStore:
    def __init__(
        self,
        memory_file: Path,
        trace_file: Path,
        include_sensitive_data: bool,
        trace_max_chars: int,
        trace_max_bytes: int = 2_000_000,
        max_events_per_trace: int = 120,
    ):
        self.memory_file = memory_file
        self.trace_file = trace_file
        self.include_sensitive_data = include_sensitive_data
        self.trace_max_chars = trace_max_chars
        self.trace_max_bytes = trace_max_bytes
        self.max_events_per_trace = max(20, int(max_events_per_trace))

    def read_memory(self) -> str:
        if not self.memory_file.exists():
            return ""
        return self.memory_file.read_text(encoding="utf-8")

    def append_memory_note(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        with self.memory_file.open("a", encoding="utf-8") as handle:
            handle.write(f"- {note}\n")

    def append_trace(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        if "event" in payload and "event_type" not in payload:
            payload["event_type"] = payload["event"]
        if "event_type" in payload and "event" not in payload:
            payload["event"] = payload["event_type"]
        if "observation" in payload and "observation_excerpt" not in payload:
            payload["observation_excerpt"] = payload.pop("observation")
        payload = self._sanitize(payload, force_redact=not self.include_sensitive_data)
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        self._rotate_if_needed(incoming_bytes=len(line) + 1)
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def tail_trace(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.trace_file.exists():
            return []
        lines: deque[str] = deque(maxlen=max(1, int(limit)))
        with self.trace_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                lines.append(line.rstrip("\n"))
        items = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def tail_trace_safe(self, limit: int = 20) -> list[dict[str, Any]]:
        items = self.tail_trace(limit=limit)
        return [self._sanitize(item, force_redact=True) for item in items]

    def export_eval_dataset(self, output_file: Path) -> dict[str, Any]:
        traces = self.build_eval_records(force_redact=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            for item in traces:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return {"traces_exported": len(traces), "output_file": str(output_file)}

    def summarize_trace_metrics(self) -> dict[str, Any]:
        traces = self.build_eval_records(force_redact=False)
        turns_total = len(traces)
        if turns_total == 0:
            return {
                "turns_total": 0,
                "turns_success": 0,
                "success_rate": 0.0,
                "tool_calls_total": 0,
                "avg_steps_per_turn": 0.0,
                "tool_calls_per_turn": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "consent_request_rate": 0.0,
                "consent_denial_rate": 0.0,
                "policy_block_rate": 0.0,
                "dangerous_tool_attempt_rate": 0.0,
                "max_steps_exhausted_count": 0,
                "tool_error_count": 0,
                "token_usage_total": 0,
                "cost_estimate_total": 0.0,
                "format_compliance_rate": 0.0,
                "judge_score_avg": 0.0,
            }

        turns_success = sum(1 for item in traces if item["outcome"]["status"] == "success")
        tool_calls_total = sum(int(item["metrics"]["tool_calls"]) for item in traces)
        total_steps = sum(int(item["metrics"]["steps"]) for item in traces)
        latencies = [int(item["metrics"]["latency_ms_total"]) for item in traces]
        consent_requests = sum(int(item["metrics"]["consent_requests"]) for item in traces)
        consent_denials = sum(int(item["metrics"]["consent_denials"]) for item in traces)
        policy_blocks = sum(int(item["metrics"]["policy_blocks"]) for item in traces)
        dangerous_attempts = sum(int(item["metrics"]["dangerous_tool_attempts"]) for item in traces)
        token_usage_total = sum(int(item["metrics"].get("token_usage", 0)) for item in traces)
        cost_estimate_total = sum(float(item["metrics"].get("cost_estimate", 0.0)) for item in traces)
        format_passes = sum(1 for item in traces if bool(item["metrics"].get("format_compliant", True)))
        judge_scores = [float(item["outcome"]["score"]) for item in traces if isinstance(item["outcome"].get("score"), (int, float))]
        max_steps_exhausted_count = sum(
            1 for item in traces if item["outcome"]["reason"] == "max_steps_exhausted"
        )
        tool_error_count = sum(1 for item in traces if item["outcome"]["reason"] == "tool_error")

        return {
            "turns_total": turns_total,
            "turns_success": turns_success,
            "success_rate": round(turns_success / turns_total, 4),
            "tool_calls_total": tool_calls_total,
            "avg_steps_per_turn": round(total_steps / turns_total, 2),
            "tool_calls_per_turn": round(tool_calls_total / turns_total, 2),
            "avg_latency_ms": round(sum(latencies) / turns_total, 2),
            "p95_latency_ms": percentile(latencies, 95),
            "consent_request_rate": round(consent_requests / turns_total, 4),
            "consent_denial_rate": round(consent_denials / turns_total, 4),
            "policy_block_rate": round(policy_blocks / turns_total, 4),
            "dangerous_tool_attempt_rate": round(dangerous_attempts / turns_total, 4),
            "max_steps_exhausted_count": max_steps_exhausted_count,
            "tool_error_count": tool_error_count,
            "token_usage_total": token_usage_total,
            "cost_estimate_total": round(cost_estimate_total, 6),
            "format_compliance_rate": round(format_passes / turns_total, 4),
            "judge_score_avg": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else 0.0,
        }

    def build_eval_records(self, force_redact: bool = True) -> list[dict[str, Any]]:
        return self._build_trace_records(force_redact=force_redact)

    def redact(self, text: str) -> str:
        text = re.sub(r"sk-[A-Za-z0-9]{16,}", "[REDACTED_API_KEY]", text)
        text = re.sub(r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]", text)
        text = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY]", text)
        text = re.sub(r"(?i)aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+=]{20,}", "aws_secret_access_key=[REDACTED]", text)
        text = re.sub(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b", "[REDACTED_JWT]", text)
        text = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]", text)
        text = re.sub(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\\s\"']+",
            "[REDACTED_DB_CONN]",
            text,
        )
        text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", text)
        text = re.sub(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*[^\s]+", r"\1=[REDACTED]", text)
        return text

    def _truncate(self, text: str) -> str:
        if len(text) <= self.trace_max_chars:
            return text
        return text[: self.trace_max_chars] + "\n...[truncated]..."

    def _sanitize(self, value: Any, force_redact: bool) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                lowered = key.lower()
                looks_sensitive_token = lowered in {"token", "access_token", "refresh_token", "id_token"}
                if (
                    any(token in lowered for token in ("api_key", "secret", "password", "authorization"))
                    or looks_sensitive_token
                ):
                    sanitized[key] = "[REDACTED]"
                    continue
                sanitized[key] = self._sanitize(item, force_redact=force_redact)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize(item, force_redact=force_redact) for item in value]
        if isinstance(value, str):
            text = self.redact(value) if force_redact else value
            return self._truncate(text)
        return value

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.trace_max_bytes <= 0:
            return
        if not self.trace_file.exists():
            return
        current_size = self.trace_file.stat().st_size
        if current_size + incoming_bytes <= self.trace_max_bytes:
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = self.trace_file.with_name(f"{self.trace_file.stem}.{ts}{self.trace_file.suffix}")
        self.trace_file.rename(rotated)
        self._prune_rotations(keep=5)

    def _prune_rotations(self, keep: int) -> None:
        pattern = f"{self.trace_file.stem}.*{self.trace_file.suffix}"
        rotated_files = sorted(
            self.trace_file.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in rotated_files[keep:]:
            stale.unlink(missing_ok=True)

    @staticmethod
    def _event_type(item: dict[str, Any]) -> str:
        event_type = str(item.get("event_type") or item.get("event") or "").strip()
        return event_type

    @staticmethod
    def _trace_group_key(item: dict[str, Any]) -> str | None:
        trace_id = str(item.get("trace_id") or "").strip()
        if trace_id:
            return trace_id
        session_id = str(item.get("session_id") or "").strip()
        turn_index = item.get("turn_index")
        if session_id and isinstance(turn_index, int):
            return f"{session_id}:{turn_index}"
        return None

    def _iter_trace_events(self):
        if not self.trace_file.exists():
            return
        with self.trace_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload

    def _build_trace_records(self, force_redact: bool) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in self._iter_trace_events():
            key = self._trace_group_key(item)
            if not key:
                continue
            safe = self._sanitize(item, force_redact=force_redact)
            event_type = self._event_type(safe)
            state = grouped.setdefault(
                key,
                {
                    "trace_id": str(safe.get("trace_id") or ""),
                    "session_id": str(safe.get("session_id") or ""),
                    "turn_index": safe.get("turn_index"),
                    "events": [],
                    "tool_sequence": [],
                    "steps": 0,
                    "tool_calls": 0,
                    "consent_requests": 0,
                    "consent_denials": 0,
                    "policy_blocks": 0,
                    "dangerous_tool_attempts": 0,
                    "llm_latency_ms_total": 0,
                    "tool_latency_ms_total": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_estimate": 0.0,
                    "user_text": "",
                    "final_text": "",
                    "has_assistant_final": False,
                    "has_max_steps": False,
                    "has_tool_error": False,
                    "invalid_output_format": False,
                    "events_truncated": 0,
                },
            )
            if not state["trace_id"]:
                state["trace_id"] = str(safe.get("trace_id") or "")
            if not state["session_id"]:
                state["session_id"] = str(safe.get("session_id") or "")
            if state["turn_index"] is None and isinstance(safe.get("turn_index"), int):
                state["turn_index"] = safe.get("turn_index")

            projected = self._project_event(safe)
            if len(state["events"]) < self.max_events_per_trace:
                state["events"].append(projected)
            else:
                state["events_truncated"] += 1

            step_raw = safe.get("step")
            if isinstance(step_raw, int):
                state["steps"] = max(int(state["steps"]), step_raw)
            elif str(step_raw).isdigit():
                state["steps"] = max(int(state["steps"]), int(str(step_raw)))

            if event_type == "tool_call":
                state["tool_calls"] += 1
                tool_name = str(safe.get("tool", "")).strip()
                if tool_name:
                    state["tool_sequence"].append(tool_name)
            elif event_type == "consent_request":
                state["consent_requests"] += 1
            elif event_type == "consent_deny":
                state["consent_denials"] += 1
            elif event_type == "guardrail_tripwire":
                state["policy_blocks"] += 1
                if str(safe.get("tool")) == "shell":
                    state["dangerous_tool_attempts"] += 1
            elif event_type == "llm_response":
                state["llm_latency_ms_total"] += self._as_int(safe.get("latency_ms"))
                state["prompt_tokens"] += self._as_int(safe.get("prompt_tokens"))
                state["completion_tokens"] += self._as_int(safe.get("completion_tokens"))
                state["total_tokens"] += self._as_int(safe.get("total_tokens"))
                state["cost_estimate"] += self._as_float(safe.get("cost_estimate"))
            elif event_type == "tool_result":
                state["tool_latency_ms_total"] += self._as_int(safe.get("latency_ms"))
                if str(safe.get("status", "")).strip() == "error":
                    state["has_tool_error"] = True
            elif event_type == "assistant_final":
                state["has_assistant_final"] = True
                state["final_text"] = str(safe.get("observation_excerpt", "")).strip()
            if str(safe.get("reason", "")).strip() == "invalid_output_format":
                state["invalid_output_format"] = True
            if event_type == "max_steps_exhausted":
                state["has_max_steps"] = True

            if event_type == "turn_start" and not state["user_text"]:
                state["user_text"] = str(safe.get("user_text_excerpt", "")).strip()

        records: list[dict[str, Any]] = []
        for key in sorted(grouped.keys()):
            state = grouped[key]
            if bool(state["has_max_steps"]):
                outcome_status = "fail"
                outcome_reason = "max_steps_exhausted"
            elif bool(state["invalid_output_format"]):
                outcome_status = "fail"
                outcome_reason = "invalid_output_format"
            elif bool(state["has_assistant_final"]):
                outcome_status = "success"
                if int(state["consent_denials"]) > 0 or int(state["policy_blocks"]) > 0:
                    outcome_reason = "success_after_policy_friction"
                else:
                    outcome_reason = "success"
            elif bool(state["has_tool_error"]):
                outcome_status = "fail"
                outcome_reason = "tool_error"
            elif int(state["consent_denials"]) > 0 or int(state["policy_blocks"]) > 0:
                outcome_status = "fail"
                outcome_reason = "policy_denied"
            else:
                outcome_status = "unknown"
                outcome_reason = "unknown"

            llm_latency_ms = int(state["llm_latency_ms_total"])
            tool_latency_ms = int(state["tool_latency_ms_total"])
            total_latency_ms = llm_latency_ms + tool_latency_ms
            trace_id = str(state["trace_id"] or key)
            format_compliant = not bool(state["invalid_output_format"])
            records.append(
                {
                    "id": trace_id or key,
                    "trace_id": trace_id,
                    "session_id": str(state["session_id"]),
                    "turn_index": state["turn_index"],
                    "input": {"user_text": str(state["user_text"])},
                    "trajectory": {
                        "events": state["events"],
                        "tool_sequence": state["tool_sequence"],
                    },
                    "output": {"final_text_summary": str(state["final_text"])},
                    "outcome": {
                        "status": outcome_status,
                        "reason": outcome_reason,
                        "score": 1.0 if outcome_status == "success" else 0.0 if outcome_status == "fail" else None,
                        "judge": "heuristic",
                    },
                    "metrics": {
                        "steps": int(state["steps"]),
                        "tool_calls": int(state["tool_calls"]),
                        "consent_requests": int(state["consent_requests"]),
                        "consent_denials": int(state["consent_denials"]),
                        "policy_blocks": int(state["policy_blocks"]),
                        "dangerous_tool_attempts": int(state["dangerous_tool_attempts"]),
                        "llm_latency_ms_total": llm_latency_ms,
                        "tool_latency_ms_total": tool_latency_ms,
                        "latency_ms_total": total_latency_ms,
                        "prompt_tokens": int(state["prompt_tokens"]),
                        "completion_tokens": int(state["completion_tokens"]),
                        "token_usage": int(state["total_tokens"]),
                        "cost_estimate": round(float(state["cost_estimate"]), 6),
                        "format_compliant": format_compliant,
                        "events_truncated": int(state["events_truncated"]),
                    },
                }
            )
        return records

    @staticmethod
    def _project_event(item: dict[str, Any]) -> dict[str, Any]:
        projected: dict[str, Any] = {}
        for key in (
            "ts",
            "event",
            "event_type",
            "step",
            "tool",
            "status",
            "latency_ms",
            "reason",
            "source_skill",
            "span_id",
            "parent_span_id",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_estimate",
        ):
            if key in item:
                projected[key] = item[key]

        event_type = str(item.get("event_type") or item.get("event") or "")
        if event_type == "turn_start" and "user_text_excerpt" in item:
            projected["user_text_excerpt"] = item.get("user_text_excerpt")
        if event_type == "tool_call" and isinstance(item.get("args"), dict):
            projected["arg_keys"] = sorted(str(key) for key in item["args"].keys())
        if event_type in {"assistant_final", "tool_result", "max_steps_exhausted"} and "observation_excerpt" in item:
            observation = str(item.get("observation_excerpt", ""))
            projected["observation_hash"] = hashlib.sha256(observation.encode("utf-8")).hexdigest()[:16]
        return projected

    @staticmethod
    def _as_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return 0.0

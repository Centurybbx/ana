from __future__ import annotations

import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ana.config import AnaConfig
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.stats import percentile


def load_benchmark_tasks(tasks_file: Path) -> list[dict[str, Any]]:
    if not tasks_file.exists():
        raise FileNotFoundError(f"benchmark file not found: {tasks_file}")

    if tasks_file.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return _normalize_tasks(rows)

    payload = json.loads(tasks_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _normalize_tasks(payload)
    if isinstance(payload, dict):
        examples = payload.get("examples", [])
        if isinstance(examples, list):
            return _normalize_tasks(examples)
    raise ValueError("invalid benchmark format: expected list/jsonl or {'examples': [...]}")


def build_fingerprint(config: AnaConfig) -> dict[str, str]:
    lock_file = config.resolved_skills_local_dir() / "lock.json"
    lock_hash = ""
    if lock_file.exists():
        lock_hash = _sha256_text(lock_file.read_text(encoding="utf-8"))

    return {
        "ana_version": _git_commit_short(config.resolved_workspace()),
        "model_id": config.model,
        "active_skills_lock_hash": lock_hash,
        "policy_version": "toolpolicy-v1",
        "prompt_version": "contextweaver-v1",
    }


def evaluate_regression_gate(
    aggregate: dict[str, Any],
    baseline_aggregate: dict[str, Any],
    latency_regression_pct: float,
) -> list[str]:
    failures: list[str] = []
    candidate_success = float(aggregate.get("success_rate", 0.0))
    baseline_success = float(baseline_aggregate.get("success_rate", 0.0))
    if candidate_success < baseline_success:
        failures.append(
            f"success_rate_regressed: candidate={candidate_success:.4f} baseline={baseline_success:.4f}"
        )

    candidate_p95 = float(aggregate.get("p95_latency_ms", 0.0))
    baseline_p95 = float(baseline_aggregate.get("p95_latency_ms", 0.0))
    allowed_p95 = baseline_p95 * (1 + (latency_regression_pct / 100.0))
    if candidate_p95 > allowed_p95:
        failures.append(
            f"p95_latency_regressed: candidate={candidate_p95:.2f}ms allowed={allowed_p95:.2f}ms"
        )

    return failures


def normalize_baseline_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    if "aggregate" in payload and isinstance(payload["aggregate"], dict):
        return payload["aggregate"]
    return payload


async def run_offline_eval(
    loop: AgentLoop,
    tasks: list[dict[str, Any]],
    max_examples: int | None = None,
    enable_llm_judge: bool = False,
) -> dict[str, Any]:
    selected = tasks[:max_examples] if max_examples and max_examples > 0 else tasks
    examples: list[dict[str, Any]] = []

    for item in selected:
        task_id = str(item.get("id", ""))
        user_text = str(item.get("input", {}).get("user_text", ""))
        expected = item.get("expected", {})
        if not isinstance(expected, dict):
            expected = {}
        tags = list(item.get("metadata", {}).get("tags", []))
        allow_side_effects = bool(item.get("metadata", {}).get("allow_side_effects", False))
        rubric = item.get("rubric") or expected.get("rubric")
        session = Session(
            session_id=f"eval-{uuid4().hex[:10]}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        runtime_state = RuntimeSessionState()

        async def confirm_actions(_: str) -> bool:
            return allow_side_effects

        started = time.perf_counter()
        error = None
        try:
            output = await loop.run_turn(
                session=session,
                user_input=user_text,
                session_state=runtime_state,
                confirm=confirm_actions,
            )
        except Exception as exc:  # pragma: no cover - defensive
            output = ""
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)

        passed, score, reasons, format_compliant = _score_output(output=output, expected=expected)
        judge_payload: dict[str, Any] | None = None
        if enable_llm_judge and rubric:
            judge_payload = await _run_llm_judge(
                loop=loop,
                user_text=user_text,
                output_text=output,
                rubric=rubric,
            )
            judge_score = float(judge_payload.get("score_overall", 0.0))
            if not expected:
                passed = judge_score >= 0.6
                score = round(judge_score, 4)
                if not passed:
                    reasons.append("judge_below_threshold")
            else:
                if judge_score < 0.3:
                    passed = False
                    reasons.append("judge_low_confidence")
        if error:
            passed = False
            score = 0.0
            reasons.append(f"runtime_error:{error}")

        examples.append(
            {
                "id": task_id,
                "input": {"user_text": user_text},
                "expected": expected,
                "output": {"text": output},
                "passed": passed,
                "score": score,
                "format_compliant": format_compliant,
                "latency_ms": latency_ms,
                "reasons": reasons,
                "metadata": {"tags": tags},
                "judge": judge_payload,
            }
        )

    aggregate = _aggregate_examples(examples)
    return {"examples": examples, "aggregate": aggregate}


def _normalize_tasks(rows: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        raw_input = item.get("input")
        user_text = ""
        if isinstance(raw_input, dict):
            user_text = str(raw_input.get("user_text", ""))
        elif isinstance(raw_input, str):
            user_text = raw_input
        elif "user_text" in item:
            user_text = str(item.get("user_text", ""))

        expected = item.get("expected", {})
        if not isinstance(expected, dict):
            expected = {}

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        normalized.append(
            {
                "id": str(item.get("id") or f"example-{index}"),
                "input": {"user_text": user_text},
                "expected": expected,
                "metadata": {"tags": _to_str_list(metadata.get("tags", []))},
            }
        )
    return normalized


def _score_output(output: str, expected: dict[str, Any]) -> tuple[bool, float, list[str], bool]:
    output_text = output or ""
    checks = 0
    passed_checks = 0
    reasons: list[str] = []
    format_compliant = True

    contains = _to_str_list(expected.get("contains", []))
    not_contains = _to_str_list(expected.get("not_contains", []))
    format_rule = expected.get("format")
    regex_checks = _to_str_list(expected.get("matches_regex", []))

    for token in contains:
        checks += 1
        if token in output_text:
            passed_checks += 1
        else:
            reasons.append(f"missing_contains:{token}")

    for token in not_contains:
        checks += 1
        if token not in output_text:
            passed_checks += 1
        else:
            reasons.append(f"contains_forbidden:{token}")

    if regex_checks:
        import re

        for pattern in regex_checks:
            checks += 1
            if re.search(pattern, output_text):
                passed_checks += 1
            else:
                reasons.append(f"regex_mismatch:{pattern}")

    if format_rule:
        checks += 1
        format_ok = _validate_format(output_text, format_rule)
        format_compliant = format_ok
        if format_ok:
            passed_checks += 1
        else:
            reasons.append(f"format_mismatch:{format_rule}")

    if checks == 0:
        return True, 1.0, reasons, format_compliant

    score = passed_checks / checks
    return passed_checks == checks, round(score, 4), reasons, format_compliant


def _aggregate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(examples)
    if total == 0:
        return {
            "examples_total": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "score_avg": 0.0,
            "judge_score_avg": 0.0,
            "format_compliance_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    passed = sum(1 for item in examples if bool(item.get("passed")))
    scores = [float(item.get("score", 0.0)) for item in examples]
    judge_scores = [
        float(item["judge"]["score_overall"])
        for item in examples
        if isinstance(item.get("judge"), dict) and isinstance(item["judge"].get("score_overall"), (int, float))
    ]
    format_passes = sum(1 for item in examples if bool(item.get("format_compliant", True)))
    latencies = [int(item.get("latency_ms", 0)) for item in examples]
    return {
        "examples_total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": round(passed / total, 4),
        "score_avg": round(sum(scores) / total, 4),
        "judge_score_avg": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else 0.0,
        "format_compliance_rate": round(format_passes / total, 4),
        "avg_latency_ms": round(sum(latencies) / total, 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
    }


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit_short(cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    value = (proc.stdout or "").strip()
    return value or "unknown"


def _validate_format(output_text: str, format_rule: Any) -> bool:
    if isinstance(format_rule, dict):
        kind = str(format_rule.get("kind", "")).strip().lower()
    else:
        kind = str(format_rule).strip().lower()
    if not kind:
        return True
    if kind == "json":
        try:
            json.loads(output_text)
            return True
        except json.JSONDecodeError:
            return False
    if kind == "markdown_table":
        return "|" in output_text and "\n" in output_text
    if kind == "bullet_list":
        return output_text.strip().startswith(("- ", "* ", "1. "))
    return True


async def _run_llm_judge(
    *,
    loop: AgentLoop,
    user_text: str,
    output_text: str,
    rubric: Any,
) -> dict[str, Any]:
    rubric_text = rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False)
    judge_prompt = (
        "You are an evaluator for an AI agent. "
        "Return strict JSON with keys: "
        "score_overall, score_tool_use, score_safety, score_efficiency, notes. "
        "Each score must be in [0,1]."
    )
    judge_input = (
        f"User goal:\n{user_text}\n\n"
        f"Agent output:\n{output_text}\n\n"
        f"Rubric:\n{rubric_text}\n"
    )
    try:
        judge_response = await loop.provider.complete(
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": judge_input},
            ],
            tools=[],
        )
    except Exception:
        return {"score_overall": 0.0, "score_tool_use": 0.0, "score_safety": 0.0, "score_efficiency": 0.0, "notes": "judge_error"}

    content = str(judge_response.content or "").strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {"score_overall": 0.0, "score_tool_use": 0.0, "score_safety": 0.0, "score_efficiency": 0.0, "notes": "invalid_judge_json"}
    if not isinstance(payload, dict):
        payload = {}
    for key in ("score_overall", "score_tool_use", "score_safety", "score_efficiency"):
        try:
            payload[key] = max(0.0, min(1.0, float(payload.get(key, 0.0))))
        except (TypeError, ValueError):
            payload[key] = 0.0
    payload["notes"] = str(payload.get("notes", "")).strip()
    return payload

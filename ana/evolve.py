from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ana.tools.skill_manager import SkillManagerTool


def proposals_dir(workspace: Path) -> Path:
    return (workspace / "proposals").resolve()


def build_failure_reports(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for record in eval_records:
        outcome = record.get("outcome", {})
        if str(outcome.get("status")) != "fail":
            continue

        root_cause = _infer_root_cause(record)
        recommended_fix_type = "policy_patch" if root_cause == "unsafe_attempt" else "skill_patch"
        risk_level = "R3" if recommended_fix_type == "policy_patch" else "R1"
        trace_id = str(record.get("trace_id") or record.get("id") or "")

        reports.append(
            {
                "failure_id": f"fr-{trace_id or uuid4().hex[:8]}",
                "trace_id": trace_id,
                "tags": sorted(set([root_cause, str(outcome.get("reason", "unknown"))])),
                "root_cause": root_cause,
                "evidence": _collect_evidence(record),
                "recommended_fix_type": recommended_fix_type,
                "risk_level": risk_level,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }


def create_skill_patch_proposal(
    workspace: Path,
    failures_payload: dict[str, Any],
    skill_name: str,
) -> dict[str, Any]:
    skill_name = skill_name.strip()
    if not skill_name:
        raise ValueError("missing skill_name")

    reports = failures_payload.get("reports", [])
    if not isinstance(reports, list) or not reports:
        raise ValueError("no failure reports available")
    selected = reports[0]
    selected_failure_id = str(selected.get("failure_id") or selected.get("trace_id") or "")
    root_cause = str(selected.get("root_cause") or "missing_skill").strip() or "missing_skill"
    evidence = selected.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    local_enabled = workspace / "skills_local" / "enabled" / skill_name / "SKILL.md"
    built_in = workspace / "skills" / skill_name / "SKILL.md"
    base_path = local_enabled if local_enabled.exists() else (built_in if built_in.exists() else None)
    old_content = base_path.read_text(encoding="utf-8") if base_path is not None else ""
    new_content = _apply_skill_improvement(
        old_content=old_content,
        skill_name=skill_name,
        root_cause=root_cause,
        evidence=[str(item) for item in evidence if str(item).strip()],
    )
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(base_path or f"{skill_name}:<new>"),
            tofile=f"skills_local/enabled/{skill_name}/SKILL.md",
        )
    )

    proposal_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    selected_risk = str(selected.get("risk_level") or "").strip().upper()
    if selected_risk in {"R0", "R1", "R2", "R3"}:
        risk_level = selected_risk
    else:
        risk_level = "R1" if base_path is not None else "R2"
    proposal = {
        "proposal_id": proposal_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "proposed",
        "type": "skill_patch",
        "risk_level": risk_level,
        "source_failure_ids": [selected_failure_id] if selected_failure_id else [],
        "target": {"skill_name": skill_name},
        "diff": diff,
        "rationale": "Address recurrent failures by adding explicit recovery/guardrail instructions in the skill.",
        "expected_impact": {"success_rate_delta": "+", "steps_delta": "-"},
        "eval_plan": {"required_datasets": ["core_regression", "failure_replay"]},
        "rollback_plan": f"evolve-rollback --skill-name {skill_name}",
        "candidate": {
            "skill_name": skill_name,
            "base_path": str(base_path) if base_path is not None else None,
            "content": new_content,
            "content_sha256": _candidate_content_hash(new_content),
        },
        "validation": None,
        "deployment": None,
    }

    folder = proposals_dir(workspace) / proposal_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    return proposal


def load_proposal(workspace: Path, proposal_id: str) -> tuple[Path, dict[str, Any]]:
    proposal_file = proposals_dir(workspace) / proposal_id / "proposal.json"
    if not proposal_file.exists():
        raise FileNotFoundError(f"proposal not found: {proposal_id}")
    payload = json.loads(proposal_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid proposal payload")
    _migrate_legacy_proposal_fields(payload)
    return proposal_file, payload


def save_proposal(proposal_file: Path, proposal: dict[str, Any]) -> None:
    proposal_file.parent.mkdir(parents=True, exist_ok=True)
    proposal_file.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")


async def deploy_skill_patch(
    *,
    workspace: Path,
    skills_local_dir: Path,
    skills_dir: Path,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    if str(proposal.get("type")) != "skill_patch":
        raise ValueError("only skill_patch proposals are supported")
    validation = proposal.get("validation") or {}
    if not bool(validation.get("pass")):
        raise ValueError("proposal is not validated")

    candidate = proposal.get("candidate", {})
    skill_name = str(candidate.get("skill_name", "")).strip()
    content = str(candidate.get("content", ""))
    if not skill_name or not content:
        raise ValueError("proposal candidate is incomplete")
    validated_hash = str(validation.get("candidate_content_sha256", "")).strip()
    current_hash = _candidate_content_hash(content)
    if not validated_hash or validated_hash != current_hash:
        raise ValueError("candidate content changed after validation")

    tool = SkillManagerTool(skills_local_dir=skills_local_dir, skills_dir=skills_dir)
    generate = await tool.run({"action": "generate", "name": skill_name, "content": content})
    if not generate.ok:
        raise RuntimeError(f"generate failed: {generate.data}")
    generated_payload = _as_json(generate.data)
    skill_id = str(generated_payload.get("skill_id", "")).strip()
    if not skill_id:
        raise RuntimeError("generate did not return skill_id")

    checked = await tool.run({"action": "check", "skill_id": skill_id})
    check_payload = _as_json(checked.data)
    if not checked.ok:
        raise RuntimeError(f"check failed: {check_payload}")

    enabled = await tool.run({"action": "enable", "skill_id": skill_id})
    enable_payload = _as_json(enabled.data)
    if not enabled.ok:
        raise RuntimeError(f"enable failed: {enable_payload}")

    return {
        "skill_id": skill_id,
        "check_status": check_payload.get("check_status"),
        "pin": enable_payload.get("pin"),
        "warnings": enable_payload.get("warnings", []),
        "candidate_content_sha256": current_hash,
    }


async def rollback_skill(
    *,
    skills_local_dir: Path,
    skills_dir: Path,
    skill_name: str,
    target: str | None = None,
) -> dict[str, Any]:
    tool = SkillManagerTool(skills_local_dir=skills_local_dir, skills_dir=skills_dir)
    args: dict[str, Any] = {"action": "rollback", "name": skill_name}
    if target:
        args["target"] = target
    result = await tool.run(args)
    if not result.ok:
        # New skills may not have snapshots yet; fallback to disable as a safe rollback path.
        if "no snapshots found" in str(result.data):
            disabled = await tool.run({"action": "disable", "name": skill_name})
            if not disabled.ok:
                raise RuntimeError(disabled.data)
            payload = _as_json(disabled.data)
            payload["rollback_mode"] = "disable_fallback"
            return payload
        raise RuntimeError(result.data)
    return _as_json(result.data)


def list_proposals(workspace: Path) -> list[dict[str, Any]]:
    root = proposals_dir(workspace)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        proposal_file = folder / "proposal.json"
        if not proposal_file.exists():
            continue
        payload = json.loads(proposal_file.read_text(encoding="utf-8"))
        items.append(
            {
                "proposal_id": payload.get("proposal_id", folder.name),
                "status": payload.get("status"),
                "type": payload.get("type"),
                "risk_level": payload.get("risk_level"),
                "created_at": payload.get("created_at"),
            }
        )
    return items


def _infer_root_cause(record: dict[str, Any]) -> str:
    outcome = record.get("outcome", {})
    reason = str(outcome.get("reason", "unknown"))
    metrics = record.get("metrics", {})
    trajectory = record.get("trajectory", {})
    events = trajectory.get("events", [])
    tool_errors = [
        str(item.get("reason", "")).strip().lower()
        for item in events
        if str(item.get("event_type")) == "tool_result" and str(item.get("status")) == "error"
    ]
    if reason == "invalid_output_format":
        return "format_failure"
    if any("format_mismatch" in item for item in tool_errors):
        return "format_failure"
    if reason == "max_steps_exhausted":
        return "excessive_steps"
    if reason == "tool_error":
        if any(token in raw for raw in tool_errors for token in ("unknown_tool", "tool_not_found", "unsupported_tool")):
            return "wrong_tool"
        if any(token in raw for raw in tool_errors for token in ("schema_mismatch", "unknown_argument", "missing_required", "invalid_param")):
            return "skill_outdated"
        return "bad_args"
    if reason == "policy_denied":
        if any(str(item.get("event_type")) == "guardrail_tripwire" and str(item.get("tool")) == "shell" for item in events):
            return "unsafe_attempt"
        if int(metrics.get("consent_denials", 0)) > 0 or int(metrics.get("policy_blocks", 0)) > 0:
            return "policy_loop"
    return "missing_skill"


def _collect_evidence(record: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    metrics = record.get("metrics", {})
    trajectory = record.get("trajectory", {})
    events = trajectory.get("events", [])
    if int(metrics.get("policy_blocks", 0)) > 0:
        evidence.append(f"policy_blocks={metrics.get('policy_blocks')}")
    if int(metrics.get("consent_denials", 0)) > 0:
        evidence.append(f"consent_denials={metrics.get('consent_denials')}")
    if int(metrics.get("steps", 0)) > 0:
        evidence.append(f"steps={metrics.get('steps')}")
    for item in events:
        event_type = str(item.get("event_type", ""))
        if event_type in {"guardrail_tripwire", "tool_result", "max_steps_exhausted"}:
            snippet = f"{event_type}"
            if item.get("tool"):
                snippet += f":{item.get('tool')}"
            if item.get("reason"):
                snippet += f" reason={item.get('reason')}"
            evidence.append(snippet)
        if len(evidence) >= 5:
            break
    return evidence or ["insufficient_evidence"]


def _apply_skill_improvement(
    old_content: str,
    skill_name: str,
    root_cause: str,
    evidence: list[str],
) -> str:
    if not old_content.strip():
        old_content = (
            "---\n"
            f"name: {skill_name}\n"
            "version: 0.1.0\n"
            "allowed_tool_names:\n"
            "  - read_file\n"
            "required_capabilities:\n"
            "  - fs.read\n"
            "source:\n"
            "  kind: local\n"
            "  uri: null\n"
            "external_links_policy: deny\n"
            "---\n\n"
            "# Purpose\n\n"
            "Help with repository analysis tasks.\n"
        )

    marker = f"## Evolution Improvements ({root_cause})"
    if marker in old_content:
        return old_content if old_content.endswith("\n") else old_content + "\n"

    guidance = _improvement_guidance(root_cause)
    evidence_lines = "".join(f"- {item}\n" for item in evidence[:3]) or "- no structured evidence captured\n"
    appendix = f"\n\n{marker}\n\n### Failure Evidence\n{evidence_lines}\n### Updated Guidance\n{guidance}"
    rendered = old_content.rstrip() + appendix
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def _as_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _candidate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _migrate_legacy_proposal_fields(proposal: dict[str, Any]) -> None:
    proposal["rollback_plan"] = _normalize_rollback_plan(str(proposal.get("rollback_plan", "")))


def _normalize_rollback_plan(rollback_plan: str) -> str:
    normalized = rollback_plan.strip()
    if not normalized:
        return normalized
    if normalized.startswith("evolve-rollback"):
        return normalized
    if "rollback" not in normalized or "--skill-name" not in normalized:
        return normalized

    parts = normalized.split()
    if len(parts) <= 1:
        return "evolve-rollback"
    return f"evolve-rollback {' '.join(parts[1:])}"


def append_proposal_audit(
    workspace: Path,
    proposal_id: str,
    action: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata = metadata or {}
    root = proposals_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    audit_file = root / "audit.jsonl"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "proposal_id": proposal_id,
        "action": action,
        "metadata": metadata,
    }
    with audit_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _improvement_guidance(root_cause: str) -> str:
    mapping = {
        "policy_loop": (
            "- If a risky action is denied, stop repeating the same call.\n"
            "- Propose a safer read-only alternative and ask for explicit next instruction.\n"
        ),
        "wrong_tool": (
            "- Validate tool capability before dispatch.\n"
            "- If the requested operation mismatches current toolset, explain the mismatch and pick the closest safe tool.\n"
        ),
        "skill_outdated": (
            "- Re-check argument names/types against current tool schema before calling.\n"
            "- If schema mismatch appears in tool errors, adjust arguments instead of retrying unchanged.\n"
        ),
        "format_failure": (
            "- Before final response, verify output format against requested structure.\n"
            "- If formatting fails, rewrite output once with strict schema compliance.\n"
        ),
        "excessive_steps": (
            "- Limit retries for the same failed tactic.\n"
            "- Summarize blockers quickly and ask for clarification after two failed attempts.\n"
        ),
        "unsafe_attempt": (
            "- Never issue unsafe shell/policy-blocked actions.\n"
            "- Prefer explicit safe alternatives and explain why the risky action is blocked.\n"
        ),
    }
    return mapping.get(
        root_cause,
        (
            "- Prefer safer read-only alternatives before side-effecting tools.\n"
            "- Keep outputs concise and include clear next-step options when blocked.\n"
        ),
    )

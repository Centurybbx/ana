from __future__ import annotations

import difflib
from typing import Any

from ana.tools.skill_lint import lint_risk_flags


def compute_skill_diff(
    *,
    skill_name: str,
    new_content: str,
    new_manifest: dict[str, Any],
    new_findings: list[dict[str, Any]],
    old_content: str | None,
    old_manifest: dict[str, Any] | None,
    old_findings: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    old_allowed = set(str(item) for item in (old_manifest or {}).get("allowed_tool_names", []))
    new_allowed = set(str(item) for item in new_manifest.get("allowed_tool_names", []))
    old_required = set(str(item) for item in (old_manifest or {}).get("required_capabilities", []))
    new_required = set(str(item) for item in new_manifest.get("required_capabilities", []))

    if old_content is None:
        operation = "fresh_install"
        diff_summary = "new enable (no existing enabled version)"
        diff_lines = list(
            difflib.unified_diff(
                [],
                new_content.splitlines(),
                fromfile=f"{skill_name}@none",
                tofile=f"{skill_name}@quarantine",
                n=1,
            )
        )
        unified = "\n".join(diff_lines)
    else:
        operation = "upgrade"
        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"{skill_name}@current",
                tofile=f"{skill_name}@quarantine",
                n=1,
            )
        )
        changed_lines = sum(1 for line in diff_lines if line.startswith("+") or line.startswith("-"))
        diff_summary = f"{changed_lines} changed line(s)"
        unified = "\n".join(diff_lines)

    new_risks = set(lint_risk_flags(new_findings))
    old_risks = set(lint_risk_flags(old_findings or []))
    return {
        "operation": operation,
        "diff_summary": diff_summary,
        "unified_diff": unified,
        "delta": {
            "added_allowed_tool_names": sorted(new_allowed - old_allowed),
            "removed_allowed_tool_names": sorted(old_allowed - new_allowed),
            "added_required_capabilities": sorted(new_required - old_required),
            "removed_required_capabilities": sorted(old_required - new_required),
        },
        "risk_delta": {
            "added_risk_flags": sorted(new_risks - old_risks),
            "removed_risk_flags": sorted(old_risks - new_risks),
        },
    }

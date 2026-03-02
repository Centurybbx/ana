from __future__ import annotations

import re
from urllib.parse import urlparse


def _finding(rule_id: str, severity: str, message: str, evidence: str, risk_flag: str | None = None) -> dict:
    payload = {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }
    if risk_flag:
        payload["risk_flag"] = risk_flag
    return payload


def lint_skill_document(
    content: str,
    manifest: dict,
    body: str,
    *,
    known_tool_names: set[str],
    known_capabilities: set[str],
) -> list[dict]:
    findings: list[dict] = []
    lowered = content.lower()
    allowed_tools = set(str(item) for item in manifest.get("allowed_tool_names", []))
    required_caps = set(str(item) for item in manifest.get("required_capabilities", []))

    danger_patterns: list[tuple[str, str]] = [
        (r"\brm\s+-rf\b", "rm -rf"),
        (r"\bmkfs\b", "mkfs"),
        (r"\bdd\s+if=", "dd if="),
        (r"\bchmod\s+777\b", "chmod 777"),
        (r"curl\s+[^|]*\|\s*sh", "curl | sh"),
        (r"wget\s+[^|]*\|\s*sh", "wget | sh"),
        (r"python\s+-c\s+.*base64", "python -c <base64>"),
        (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    ]
    for pattern, evidence in danger_patterns:
        if re.search(pattern, lowered):
            findings.append(
                _finding(
                    "dangerous_shell_pattern",
                    "ERROR",
                    "Detected destructive or supply-chain shell pattern.",
                    evidence=evidence,
                    risk_flag="destructive_shell",
                )
            )

    url_matches = re.findall(r"https?://[^\s)\]]+", content, flags=re.IGNORECASE)
    if url_matches:
        policy = str(manifest.get("external_links_policy", "deny")).lower()
        allowlist = {str(item).strip().lower() for item in manifest.get("external_links_allowlist", [])}
        if policy == "deny":
            findings.append(
                _finding(
                    "external_link_denied",
                    "ERROR",
                    "External links are present while policy is deny.",
                    evidence=", ".join(sorted(url_matches)),
                    risk_flag="external_link",
                )
            )
        else:
            blocked_hosts: list[str] = []
            for url in url_matches:
                host = (urlparse(url).hostname or "").lower()
                if host and host not in allowlist:
                    blocked_hosts.append(host)
            if blocked_hosts:
                findings.append(
                    _finding(
                        "external_link_not_allowlisted",
                        "ERROR",
                        "External link host is not in allowlist.",
                        evidence=", ".join(sorted(set(blocked_hosts))),
                        risk_flag="external_link",
                    )
                )
            else:
                findings.append(
                    _finding(
                        "external_link_present",
                        "WARN",
                        "External links are present and allowlisted.",
                        evidence=", ".join(sorted(url_matches)),
                        risk_flag="external_link",
                    )
                )

    for tool in sorted(allowed_tools):
        if tool not in known_tool_names:
            findings.append(
                _finding(
                    "unknown_tool_name",
                    "ERROR",
                    "Manifest declares an unknown tool name.",
                    evidence=tool,
                )
            )
    for cap in sorted(required_caps):
        if cap not in known_capabilities:
            findings.append(
                _finding(
                    "unknown_capability",
                    "ERROR",
                    "Manifest declares an unknown capability.",
                    evidence=cap,
                )
            )

    if "shell" in allowed_tools:
        constraints = manifest.get("constraints", {})
        shell_constraints = constraints.get("shell") if isinstance(constraints, dict) else None
        commands = []
        if isinstance(shell_constraints, dict):
            commands = shell_constraints.get("allowed_commands", []) or []
        if not isinstance(commands, list) or not commands:
            findings.append(
                _finding(
                    "shell_without_constraints",
                    "WARN",
                    "Manifest allows shell but does not constrain allowed commands.",
                    evidence="shell",
                )
            )

    body_lowered = body.lower()
    for tool in sorted(known_tool_names):
        if tool in {"skill_manager"}:
            continue
        if re.search(rf"\b{re.escape(tool)}\b", body_lowered) and tool not in allowed_tools:
            findings.append(
                _finding(
                    "undeclared_tool_reference",
                    "ERROR",
                    "Body references a tool not declared in manifest.",
                    evidence=tool,
                )
            )
    if re.search(r"\bshell\b", body_lowered) and "shell" not in allowed_tools:
        findings.append(
            _finding(
                "undeclared_shell_reference",
                "ERROR",
                "Body references shell usage but manifest does not declare shell.",
                evidence="shell",
            )
        )

    if re.search(r"\b(token|api[_\s-]?key|secret|ssh|aws|github_token|g[hp]_[a-z0-9_]+)\b", lowered):
        findings.append(
            _finding(
                "possible_secret_reference",
                "WARN",
                "Potential secret-related reference detected.",
                evidence="token/api-key/secret",
                risk_flag="credential_reference",
            )
        )

    if re.search(r"(/etc|~/.ssh|~/.aws|keychain)", lowered):
        findings.append(
            _finding(
                "sensitive_path_reference",
                "ERROR",
                "Sensitive path reference conflicts with workspace-bound policy.",
                evidence="sensitive path",
                risk_flag="path_escape",
            )
        )

    prompt_injection_patterns = [
        r"ignore (all )?(previous|prior) instructions",
        r"bypass (the )?(policy|guardrail|safety)",
        r"do not ask (for )?(confirmation|approval)",
        r"override (system|developer) prompt",
    ]
    for pattern in prompt_injection_patterns:
        if re.search(pattern, lowered):
            findings.append(
                _finding(
                    "prompt_injection_instruction",
                    "ERROR",
                    "Document contains instruction-shaped content that attempts to override safeguards.",
                    evidence=pattern,
                    risk_flag="prompt_injection",
                )
            )

    if url_matches and re.search(r"\b(send|upload|post|exfil|email)\b", lowered):
        findings.append(
            _finding(
                "possible_data_exfiltration",
                "ERROR",
                "Document suggests sending data to external destinations.",
                evidence="external transfer verbs",
                risk_flag="credential_exfil",
            )
        )

    findings.sort(key=lambda item: (item["severity"], item["rule_id"], item["evidence"]))
    return findings


def lint_status(findings: list[dict]) -> str:
    if any(item.get("severity") == "ERROR" for item in findings):
        return "FAIL"
    if any(item.get("severity") == "WARN" for item in findings):
        return "WARN"
    return "PASS"


def lint_risk_flags(findings: list[dict]) -> list[str]:
    flags = {str(item.get("risk_flag", "")).strip() for item in findings if item.get("risk_flag")}
    return sorted(flag for flag in flags if flag)

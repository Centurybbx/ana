from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ana.core.session import RuntimeSessionState
from ana.tools.path_utils import is_within, is_within_any, resolve_candidate
from ana.tools.shell_guard import validate_shell_command
from ana.tools.skill_diff import compute_skill_diff
from ana.tools.skill_lint import lint_risk_flags, lint_skill_document
from ana.tools.skill_manifest import parse_and_normalize_manifest


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    requires_confirmation: bool
    plan: str
    temporary_capability: str | None = None


class ToolPolicy:
    KNOWN_TOOL_NAMES = {"read_file", "write_file", "shell", "web_search", "web_fetch", "skill_manager"}
    KNOWN_CAPABILITIES = {
        "fs.read",
        "fs.write_workspace",
        "web.read",
        "shell.safe",
        "skills.read",
        "skills.install_quarantine",
    }

    def __init__(self, workspace: Path, skills_local: Path):
        self.workspace = workspace.resolve()
        self.skills_local = skills_local.resolve()

    def precheck(self, tool_name: str, args: dict, session_state: RuntimeSessionState) -> PolicyDecision:
        if tool_name == "read_file":
            decision = self._check_read_file(args)
            return self._finalize_decision(decision, "fs.read", tool_name, args, session_state)
        if tool_name == "write_file":
            decision, capability = self._check_write_file(args)
            return self._finalize_decision(decision, capability, tool_name, args, session_state)
        if tool_name == "shell":
            decision = self._check_shell(args)
            return self._finalize_decision(decision, "shell.safe", tool_name, args, session_state)
        if tool_name == "skill_manager":
            decision, capability = self._check_skill_manager(args)
            return self._finalize_decision(decision, capability, tool_name, args, session_state)
        if tool_name in {"web_search", "web_fetch"}:
            decision = PolicyDecision(True, "read-only tool", False, "")
            return self._finalize_decision(decision, "web.read", tool_name, args, session_state)
        return PolicyDecision(False, f"tool blocked by policy: {tool_name}", False, "")

    def postprocess(self, text: str, max_chars: int = 1200) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]..."

    def _check_read_file(self, args: dict) -> PolicyDecision:
        path_value = str(args.get("path", "")).strip()
        if not path_value:
            return PolicyDecision(False, "missing path", False, "")

        candidate = resolve_candidate(path_value, self.workspace)
        if not is_within(candidate, self.workspace):
            return PolicyDecision(False, "read outside workspace is blocked", False, "")
        return PolicyDecision(True, "read-only tool", False, "")

    def _check_write_file(self, args: dict) -> tuple[PolicyDecision, str]:
        path_value = str(args.get("path", "")).strip()
        if not path_value:
            return PolicyDecision(False, "missing path", False, ""), "fs.write_workspace"

        candidate = resolve_candidate(path_value, self.workspace)
        if not is_within_any(candidate, [self.workspace, self.skills_local]):
            return PolicyDecision(False, "write outside workspace/skills_local is blocked", False, ""), "fs.write_workspace"

        capability = "skills.install_quarantine" if is_within(candidate, self.skills_local) else "fs.write_workspace"

        plan = "\n".join(
            [
                "Action: write_file",
                f"Target: {candidate}",
                f"Mode: {args.get('mode', 'overwrite')}",
                "Risk: file contents will be modified",
                "Rollback: restore previous file content from git/session history",
            ]
        )
        return PolicyDecision(True, "write requires user consent", True, plan), capability

    def _check_shell(self, args: dict) -> PolicyDecision:
        cmd = str(args.get("cmd", ""))
        validation = validate_shell_command(cmd)
        if not validation.allowed:
            return PolicyDecision(False, validation.reason, False, "")

        plan = "\n".join(
            [
                "Action: shell",
                f"Command: {cmd.strip()}",
                f"Scope: cwd={self.workspace}",
                "Risk: command may change files or git state",
                "Rollback: inspect git diff and revert manually if needed",
            ]
        )
        return PolicyDecision(True, "shell execution requires user consent", True, plan)

    def _check_skill_manager(self, args: dict) -> tuple[PolicyDecision, str]:
        action = str(args.get("action", "")).strip()
        valid_actions = {
            "generate",
            "check",
            "diff",
            "enable",
            "disable",
            "rollback",
            "list",
            "install",
            "remove",
        }
        if action not in valid_actions:
            return PolicyDecision(False, "invalid skill_manager action", False, ""), "skills.read"

        if action in {"list", "check", "diff"}:
            if action in {"check", "diff"}:
                skill_id = str(args.get("skill_id", "")).strip()
                skill_name = str(args.get("name", "")).strip()
                if not skill_id and not skill_name:
                    return PolicyDecision(False, "missing skill_id or name", False, ""), "skills.read"
            return PolicyDecision(True, f"{action} is read-only", False, ""), "skills.read"

        skill_id = str(args.get("skill_id", "")).strip()
        skill_name = str(args.get("name", "")).strip()
        content = str(args.get("content", "")).strip()
        if action in {"generate", "install"}:
            if not skill_name:
                return PolicyDecision(False, "missing skill name", False, ""), "skills.install_quarantine"
            if not content:
                return PolicyDecision(False, "missing skill content", False, ""), "skills.install_quarantine"
        elif action in {"enable", "remove"}:
            if not skill_id and not skill_name:
                return PolicyDecision(False, "missing skill_id or name", False, ""), "skills.install_quarantine"
        elif action in {"disable", "rollback"}:
            if not skill_name:
                return PolicyDecision(False, "missing skill name", False, ""), "skills.install_quarantine"

        transition = {
            "generate": "draft -> quarantined",
            "install": "draft -> quarantined (legacy alias)",
            "enable": "quarantined -> enabled",
            "disable": "enabled -> disabled",
            "rollback": "enabled -> enabled (restore snapshot)",
            "remove": "quarantined/legacy -> removed",
        }.get(action, "state mutation")
        target = skill_name or skill_id or "<unknown>"
        plan = "\n".join(
            [
                f"Action: skill_manager ({action})",
                f"Target skill: {target}",
                f"Transition: {transition}",
                f"Scope: {self.skills_local}",
                "Effects: may update quarantine/enabled/versions, lock.json, and audit.jsonl",
                "Risk: skill behavior may affect future tool planning",
                "Rollback: use skill_manager rollback or disable action",
            ]
        )
        if action == "enable":
            plan = self._enrich_enable_plan(plan=plan, args=args, fallback_name=skill_name)
        return PolicyDecision(True, "skill mutation requires user consent", True, plan), "skills.install_quarantine"

    def _enrich_enable_plan(self, plan: str, args: dict, fallback_name: str) -> str:
        skill_file = self._resolve_quarantine_skill_file(args=args, fallback_name=fallback_name)
        if skill_file is None or not skill_file.exists():
            return (
                f"{plan}\nRequired capabilities: <unknown>\nAllowed tools: <unknown>\nDiff summary: <unavailable>"
                "\nRisk delta: <unavailable>\nPin: <unavailable>"
            )

        new_content = skill_file.read_text(encoding="utf-8")
        parsed = parse_and_normalize_manifest(new_content, fallback_name=fallback_name or skill_file.parent.name)
        manifest = parsed["manifest"]
        required_capabilities = manifest.get("required_capabilities", [])
        allowed_tools = manifest.get("allowed_tool_names", [])

        skill_name = str(manifest.get("name") or fallback_name or skill_file.parent.name).strip()
        enabled_file = self.skills_local / "enabled" / skill_name / "SKILL.md"
        old_content = enabled_file.read_text(encoding="utf-8") if enabled_file.exists() else None
        old_parsed = (
            parse_and_normalize_manifest(old_content, fallback_name=skill_name) if old_content is not None else {"manifest": {}, "body": ""}
        )

        new_findings = lint_skill_document(
            content=new_content,
            manifest=manifest,
            body=parsed["body"],
            known_tool_names=self.KNOWN_TOOL_NAMES,
            known_capabilities=self.KNOWN_CAPABILITIES,
        )
        old_findings = (
            lint_skill_document(
                content=old_content or "",
                manifest=old_parsed["manifest"],
                body=old_parsed.get("body", ""),
                known_tool_names=self.KNOWN_TOOL_NAMES,
                known_capabilities=self.KNOWN_CAPABILITIES,
            )
            if old_content is not None
            else []
        )
        diff = compute_skill_diff(
            skill_name=skill_name,
            new_content=new_content,
            new_manifest=manifest,
            new_findings=new_findings,
            old_content=old_content,
            old_manifest=old_parsed["manifest"],
            old_findings=old_findings,
        )

        content_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        pin = f"{skill_name}@{manifest.get('version', '0.0.0')}#{content_hash}"
        added_risks = lint_risk_flags(new_findings)
        old_risks = lint_risk_flags(old_findings)
        risk_delta = sorted(set(added_risks) - set(old_risks))
        built_in_override = (self.workspace / "skills" / skill_name / "SKILL.md").exists()

        required_text = ", ".join(str(item) for item in required_capabilities) if required_capabilities else "<none>"
        tools_text = ", ".join(str(item) for item in allowed_tools) if allowed_tools else "<none>"
        risk_text = ", ".join(risk_delta) if risk_delta else "<none>"
        lines = [
            plan,
            f"Required capabilities: {required_text}",
            f"Allowed tools: {tools_text}",
            f"Diff summary: {diff['diff_summary']}",
            f"Risk delta: {risk_text}",
            f"Pin: {pin}",
        ]
        if built_in_override:
            lines.append("Built-in override: yes (local skill will shadow built-in)")
        return "\n".join(lines)

    def _resolve_quarantine_skill_file(self, args: dict, fallback_name: str) -> Path | None:
        skill_id = str(args.get("skill_id", "")).strip()
        if skill_id:
            candidate = self.skills_local / "quarantine" / skill_id / "SKILL.md"
            if candidate.exists():
                return candidate
            return None

        skill_name = str(args.get("name", "")).strip() or fallback_name
        if not skill_name:
            return None
        quarantine_dir = self.skills_local / "quarantine"
        if not quarantine_dir.exists():
            return None
        for skill_dir in sorted(quarantine_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            candidate = skill_dir / "SKILL.md"
            if not candidate.exists():
                continue
            manifest = parse_and_normalize_manifest(candidate.read_text(encoding="utf-8"), fallback_name=skill_dir.name)["manifest"]
            manifest_name = str(manifest.get("name", "")).strip()
            if manifest_name == skill_name or skill_dir.name == skill_name:
                return candidate
        return None

    @staticmethod
    def _apply_capability_gate(
        decision: PolicyDecision,
        capability: str,
        session_state: RuntimeSessionState,
    ) -> PolicyDecision:
        if not decision.allowed:
            return decision
        if capability in session_state.capabilities:
            return decision

        grant_line = f"Capability grant needed (one-time): {capability}"
        plan = decision.plan
        if plan:
            plan = f"{plan}\n{grant_line}"
        else:
            plan = grant_line
        return PolicyDecision(
            allowed=True,
            reason=f"missing capability: {capability}",
            requires_confirmation=True,
            plan=plan,
            temporary_capability=capability,
        )

    def _finalize_decision(
        self,
        decision: PolicyDecision,
        capability: str,
        tool_name: str,
        args: dict,
        session_state: RuntimeSessionState,
    ) -> PolicyDecision:
        constrained = self._apply_skill_constraint(decision, capability, tool_name, args, session_state)
        if not constrained.allowed:
            return constrained
        return self._apply_capability_gate(constrained, capability, session_state)

    @staticmethod
    def _extract_source_skill(args: dict) -> str | None:
        source_skill = str(args.get("_source_skill") or args.get("source_skill") or "").strip()
        return source_skill or None

    def _apply_skill_constraint(
        self,
        decision: PolicyDecision,
        capability: str,
        tool_name: str,
        args: dict,
        session_state: RuntimeSessionState,
    ) -> PolicyDecision:
        if not decision.allowed:
            return decision

        source_skill = self._extract_source_skill(args)
        if not source_skill:
            if session_state.active_skills and tool_name in {"write_file", "shell"}:
                return PolicyDecision(
                    allowed=False,
                    reason="missing _source_skill attribution for side-effecting action while active skills are loaded",
                    requires_confirmation=False,
                    plan="",
                )
            return decision

        # Enforce only when source attribution can be resolved to an active skill.
        profile = session_state.active_skills.get(source_skill, {})
        if not isinstance(profile, dict) or not profile:
            return decision

        allowed_tools = profile.get("allowed_tool_names", [])
        if isinstance(allowed_tools, list):
            allowed_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
            if allowed_set and tool_name not in allowed_set:
                return PolicyDecision(
                    allowed=False,
                    reason=f"skill constraint blocked tool '{tool_name}' for source_skill '{source_skill}'",
                    requires_confirmation=False,
                    plan="",
                )

        required_caps = profile.get("required_capabilities", [])
        if isinstance(required_caps, list):
            required_set = {str(item).strip() for item in required_caps if str(item).strip()}
            if required_set and capability not in required_set:
                return PolicyDecision(
                    allowed=False,
                    reason=f"skill constraint blocked capability '{capability}' for source_skill '{source_skill}'",
                    requires_confirmation=False,
                    plan="",
                )

        if tool_name == "skill_manager":
            escalation = self._check_skill_manager_escalation(source_skill=source_skill, args=args, profile=profile)
            if escalation is not None:
                return escalation

        return decision

    def _check_skill_manager_escalation(self, source_skill: str, args: dict, profile: dict) -> PolicyDecision | None:
        action = str(args.get("action", "")).strip()
        if action != "enable":
            return None

        candidate = self._resolve_quarantine_skill_file(args=args, fallback_name=str(args.get("name", "")).strip())
        if candidate is None or not candidate.exists():
            return None
        parsed = parse_and_normalize_manifest(candidate.read_text(encoding="utf-8"), fallback_name=candidate.parent.name)
        target = parsed["manifest"]
        target_tools = {str(item).strip() for item in target.get("allowed_tool_names", []) if str(item).strip()}
        target_caps = {str(item).strip() for item in target.get("required_capabilities", []) if str(item).strip()}
        source_tools = {str(item).strip() for item in profile.get("allowed_tool_names", []) if str(item).strip()}
        source_caps = {str(item).strip() for item in profile.get("required_capabilities", []) if str(item).strip()}

        if source_tools and not target_tools.issubset(source_tools):
            extra_tools = sorted(target_tools - source_tools)
            return PolicyDecision(
                allowed=False,
                reason=(
                    "skill_manager enable escalation blocked: target skill introduces tools outside source_skill "
                    f"'{source_skill}' scope ({', '.join(extra_tools)})"
                ),
                requires_confirmation=False,
                plan="",
            )
        if source_caps and not target_caps.issubset(source_caps):
            extra_caps = sorted(target_caps - source_caps)
            return PolicyDecision(
                allowed=False,
                reason=(
                    "skill_manager enable escalation blocked: target skill introduces capabilities outside source_skill "
                    f"'{source_skill}' scope ({', '.join(extra_caps)})"
                ),
                requires_confirmation=False,
                plan="",
            )
        return None

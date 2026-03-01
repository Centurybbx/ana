from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aha.core.session import RuntimeSessionState
from aha.tools.path_utils import is_within, is_within_any, resolve_candidate
from aha.tools.shell_guard import validate_shell_command


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    requires_confirmation: bool
    plan: str
    temporary_capability: str | None = None


class ToolPolicy:
    def __init__(self, workspace: Path, skills_local: Path):
        self.workspace = workspace.resolve()
        self.skills_local = skills_local.resolve()

    def precheck(self, tool_name: str, args: dict, session_state: RuntimeSessionState) -> PolicyDecision:
        if tool_name == "read_file":
            decision = self._check_read_file(args)
            return self._apply_capability_gate(decision, "fs.read", session_state)
        if tool_name == "write_file":
            decision, capability = self._check_write_file(args)
            return self._apply_capability_gate(decision, capability, session_state)
        if tool_name == "shell":
            decision = self._check_shell(args)
            return self._apply_capability_gate(decision, "shell.safe", session_state)
        if tool_name == "skill_manager":
            decision, capability = self._check_skill_manager(args)
            return self._apply_capability_gate(decision, capability, session_state)
        if tool_name in {"web_search", "web_fetch"}:
            decision = PolicyDecision(True, "read-only tool", False, "")
            return self._apply_capability_gate(decision, "web.read", session_state)
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

        if is_within(candidate, self.skills_local):
            capability = "skills.install_quarantine"
        else:
            capability = "fs.write_workspace"

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
        if action not in {"install", "remove", "list"}:
            return PolicyDecision(False, "invalid skill_manager action", False, ""), "skills.read"
        if action == "list":
            return PolicyDecision(True, "list is read-only", False, ""), "skills.read"

        skill_name = str(args.get("name", "")).strip()
        if not skill_name:
            return PolicyDecision(False, "missing skill name", False, ""), "skills.install_quarantine"
        plan = "\n".join(
            [
                f"Action: skill_manager ({action})",
                f"Target skill: {skill_name}",
                f"Scope: {self.skills_local}",
                "Risk: local skill set will change",
                "Rollback: remove the skill from quarantine directory",
            ]
        )
        return PolicyDecision(True, "skill mutation requires user consent", True, plan), "skills.install_quarantine"

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

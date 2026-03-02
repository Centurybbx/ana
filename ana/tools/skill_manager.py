from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ana.tools.base import Tool, ToolResult
from ana.tools.skill_diff import compute_skill_diff
from ana.tools.skill_lint import lint_risk_flags, lint_skill_document, lint_status
from ana.tools.skill_manifest import parse_and_normalize_manifest
from ana.tools.skill_resolution import resolve_skill_views


class SkillManagerTool(Tool):
    KNOWN_TOOL_NAMES = {
        "read_file",
        "write_file",
        "shell",
        "web_search",
        "web_fetch",
        "skill_manager",
    }
    KNOWN_CAPABILITIES = {
        "fs.read",
        "fs.write_workspace",
        "web.read",
        "shell.safe",
        "skills.read",
        "skills.install_quarantine",
    }
    RESERVED_ROOT_NAMES = {"quarantine", "enabled", "versions"}
    ENABLE_ALLOWED_CHECK_STATUS = {"PASS", "WARN"}
    MUTATING_ACTIONS = {"generate", "enable", "disable", "rollback", "remove"}
    CHECK_MAX_AGE_SECONDS = 24 * 60 * 60
    RATE_LIMIT_WINDOW_SECONDS = 60
    RATE_LIMIT_MAX_ACTIONS = 30
    SKILL_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

    name = "skill_manager"
    description = "Manage local skills with quarantine/check/enable governance."
    side_effect = True
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "generate",
                    "check",
                    "diff",
                    "enable",
                    "disable",
                    "rollback",
                    "list",
                    "install",
                    "remove",
                ],
            },
            "name": {"type": "string"},
            "content": {"type": "string"},
            "skill_id": {"type": "string"},
            "target": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        skills_local_dir: Path,
        skills_dir: Path | None = None,
        known_tool_names: set[str] | None = None,
        known_capabilities: set[str] | None = None,
    ):
        self.skills_local_dir = skills_local_dir.resolve()
        self.skills_dir = skills_dir.resolve() if skills_dir else None
        self.known_tool_names = set(known_tool_names or self.KNOWN_TOOL_NAMES)
        self.known_capabilities = set(known_capabilities or self.KNOWN_CAPABILITIES)
        self.quarantine_dir = self.skills_local_dir / "quarantine"
        self.enabled_dir = self.skills_local_dir / "enabled"
        self.versions_dir = self.skills_local_dir / "versions"
        self.audit_file = self.skills_local_dir / "audit.jsonl"
        self.lock_file = self.skills_local_dir / "lock.json"
        self.integrity_key_file = self.skills_local_dir / ".integrity.key"
        self.rate_limit_file = self.skills_local_dir / "rate_limit.json"
        self._ensure_layout()

    def set_known_tool_names(self, names: set[str]) -> None:
        if names:
            self.known_tool_names = set(names)

    def set_known_capabilities(self, caps: set[str]) -> None:
        if caps:
            self.known_capabilities = set(caps)

    async def run(self, args: dict[str, Any]) -> ToolResult:
        raw_action = str(args.get("action", "")).strip()
        action = "generate" if raw_action == "install" else raw_action

        if action in self.MUTATING_ACTIONS:
            blocked = self._check_rate_limit(action)
            if blocked is not None:
                return blocked

        if action == "list":
            return self._list_skills()

        if action == "generate":
            name = str(args.get("name", "")).strip()
            if not name:
                return ToolResult(ok=False, data="missing skill name", warnings=["missing_name"])
            content = str(args.get("content", "")).strip()
            if not content:
                return ToolResult(ok=False, data="missing skill content", warnings=["missing_content"])
            warnings: list[str] = []
            if raw_action == "install":
                warnings.append("deprecated_action_install")
            result = self._generate_skill(name=name, content=content, warnings=warnings)
            if result.ok:
                self._record_rate_event(action)
            return result

        if action in {"check", "diff", "enable"}:
            skill_id = str(args.get("skill_id", "")).strip() or None
            name = str(args.get("name", "")).strip() or None
            if not skill_id and not name:
                return ToolResult(ok=False, data="missing skill_id or name", warnings=["missing_identifier"])
            if action == "check":
                return self._check_skill(skill_id=skill_id, name=name)
            if action == "diff":
                return self._diff_skill(skill_id=skill_id, name=name)
            result = self._enable_skill(skill_id=skill_id, name=name)
            if result.ok:
                self._record_rate_event(action)
            return result

        if action == "disable":
            name = str(args.get("name", "")).strip()
            if not name:
                return ToolResult(ok=False, data="missing skill name", warnings=["missing_name"])
            if not self._is_valid_skill_name(name):
                return ToolResult(ok=False, data="invalid skill name", warnings=["invalid_name"])
            result = self._disable_skill(name=name)
            if result.ok:
                self._record_rate_event(action)
            return result

        if action == "rollback":
            name = str(args.get("name", "")).strip()
            if not name:
                return ToolResult(ok=False, data="missing skill name", warnings=["missing_name"])
            if not self._is_valid_skill_name(name):
                return ToolResult(ok=False, data="invalid skill name", warnings=["invalid_name"])
            target = str(args.get("target", "")).strip() or None
            result = self._rollback_skill(name=name, target=target)
            if result.ok:
                self._record_rate_event(action)
            return result

        if raw_action == "remove":
            skill_id = str(args.get("skill_id", "")).strip() or None
            name = str(args.get("name", "")).strip() or None
            if not skill_id and not name:
                return ToolResult(ok=False, data="missing skill_id or name", warnings=["missing_identifier"])
            result = self._remove_legacy(skill_id=skill_id, name=name)
            if result.ok:
                self._record_rate_event(action)
            return result

        return ToolResult(ok=False, data=f"invalid action: {raw_action}", warnings=["invalid_action"])

    def _ensure_layout(self) -> None:
        self.skills_local_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.enabled_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        if not self.audit_file.exists():
            self.audit_file.write_text("", encoding="utf-8")
        if not self.lock_file.exists():
            self.lock_file.write_text("{}", encoding="utf-8")
        if not self.integrity_key_file.exists():
            self.integrity_key_file.write_bytes(os.urandom(32))
        if not self.rate_limit_file.exists():
            self.rate_limit_file.write_text("{\"events\": []}", encoding="utf-8")

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _new_skill_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{stamp}-{uuid4().hex[:8]}"

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_pin(name: str, version: str, content_hash: str) -> str:
        return f"{name}@{version}#{content_hash}"

    def _read_lock(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.lock_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _lock_drift_warnings(self) -> list[str]:
        lock = self._read_lock()
        warnings: list[str] = []
        for name, entry in lock.items():
            if not isinstance(entry, dict):
                continue
            locked_hash = str(entry.get("hash", "")).strip()
            if not locked_hash:
                continue
            enabled_file = self.enabled_dir / str(name) / "SKILL.md"
            if not enabled_file.exists():
                warnings.append(f"lock_entry_without_enabled_skill:{name}")
                continue
            actual_hash = self._content_hash(enabled_file.read_text(encoding="utf-8"))
            if actual_hash != locked_hash:
                warnings.append(f"lock_hash_drift:{name}")
        return sorted(set(warnings))

    def _write_lock(self, payload: dict[str, Any]) -> None:
        self.lock_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _integrity_key(self) -> bytes:
        return self.integrity_key_file.read_bytes()

    def _sign_payload(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._integrity_key(), encoded, hashlib.sha256).hexdigest()

    def _last_audit_hash(self) -> str:
        if not self.audit_file.exists():
            return ""
        lines = self.audit_file.read_text(encoding="utf-8").splitlines()
        if not lines:
            return ""
        try:
            last_payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return hashlib.sha256(lines[-1].encode("utf-8")).hexdigest()
        if isinstance(last_payload, dict):
            return str(last_payload.get("entry_hash") or "")
        return ""

    def _append_audit(self, action: str, **extra: Any) -> None:
        prev_hash = self._last_audit_hash()
        payload = {
            "ts": self._iso_now(),
            "actor": "agent",
            "action": action,
            "prev_hash": prev_hash,
            **extra,
        }
        payload["entry_hash"] = self._sign_payload(payload)
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _check_rate_limit(self, action: str) -> ToolResult | None:
        now = int(datetime.now(timezone.utc).timestamp())
        try:
            payload = json.loads(self.rate_limit_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {"events": []}
        events = payload.get("events", [])
        if not isinstance(events, list):
            events = []
        fresh = [item for item in events if isinstance(item, dict) and int(item.get("ts", 0)) >= now - self.RATE_LIMIT_WINDOW_SECONDS]
        if len(fresh) >= self.RATE_LIMIT_MAX_ACTIONS:
            return ToolResult(
                ok=False,
                data="rate limit exceeded for skill mutations",
                warnings=["rate_limited"],
                meta={"window_seconds": self.RATE_LIMIT_WINDOW_SECONDS, "max_actions": self.RATE_LIMIT_MAX_ACTIONS},
            )
        return None

    def _record_rate_event(self, action: str) -> None:
        now = int(datetime.now(timezone.utc).timestamp())
        try:
            payload = json.loads(self.rate_limit_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {"events": []}
        events = payload.get("events", [])
        if not isinstance(events, list):
            events = []
        events = [item for item in events if isinstance(item, dict) and int(item.get("ts", 0)) >= now - self.RATE_LIMIT_WINDOW_SECONDS]
        events.append({"ts": now, "action": action})
        self.rate_limit_file.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _result(
        ok: bool,
        payload: dict[str, Any],
        warnings: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ToolResult:
        return ToolResult(
            ok=ok,
            data=json.dumps(payload, ensure_ascii=False),
            warnings=warnings or [],
            meta=meta or {},
        )

    @staticmethod
    def _has_frontmatter(content: str) -> bool:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return False
        return any(line.strip() == "---" for line in lines[1:])

    def _inject_default_manifest(self, name: str, content: str) -> str:
        if self._has_frontmatter(content):
            return content.rstrip() + "\n"
        safe_name = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_") or "skill"
        return (
            "---\n"
            f"name: {safe_name}\n"
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
            f"{content.rstrip()}\n"
        )

    def _manifest_name_version(self, content: str, fallback_name: str | None = None) -> tuple[str | None, str | None]:
        parsed = parse_and_normalize_manifest(content, fallback_name=fallback_name)
        manifest = parsed["manifest"]
        name = str(manifest.get("name", "")).strip() or None
        version = str(manifest.get("version", "")).strip() or None
        return name, version

    def _is_valid_skill_name(self, name: str | None) -> bool:
        if not name:
            return False
        return bool(self.SKILL_NAME_RE.fullmatch(name))

    def _quarantine_skill_path(self, skill_id: str) -> Path:
        return self.quarantine_dir / skill_id / "SKILL.md"

    def _quarantine_check_path(self, skill_id: str) -> Path:
        return self.quarantine_dir / skill_id / "CHECK.json"

    def _iter_quarantine_records(self) -> list[tuple[str, Path]]:
        records: list[tuple[str, Path]] = []
        for skill_dir in sorted(self.quarantine_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            records.append((skill_dir.name, skill_file))
        return records

    def _resolve_quarantine_record(self, skill_id: str | None, name: str | None) -> tuple[str, Path] | None:
        if skill_id:
            candidate = self._quarantine_skill_path(skill_id)
            if candidate.exists():
                return skill_id, candidate
            return None

        if not name:
            return None
        matches: list[tuple[str, Path]] = []
        for candidate_id, candidate_path in self._iter_quarantine_records():
            content = candidate_path.read_text(encoding="utf-8")
            manifest_name, _ = self._manifest_name_version(content, fallback_name=candidate_id)
            if manifest_name == name or candidate_id == name:
                matches.append((candidate_id, candidate_path))
        if not matches:
            return None
        return matches[-1]

    def _analyze_content(self, content: str, fallback_name: str | None = None) -> dict[str, Any]:
        parsed = parse_and_normalize_manifest(content, fallback_name=fallback_name)
        manifest = parsed["manifest"]
        findings = lint_skill_document(
            content=content,
            manifest=manifest,
            body=parsed["body"],
            known_tool_names=self.known_tool_names,
            known_capabilities=self.known_capabilities,
        )
        manifest_errors = list(parsed["errors"])
        if manifest.get("source", {}).get("kind") == "remote":
            manifest_errors.append("remote_source_not_installable")

        rule_errors = [f"lint:{item['rule_id']}" for item in findings if item.get("severity") == "ERROR"]
        warnings = list(parsed["warnings"]) + list(parsed["compat_warnings"])
        warnings.extend([f"lint:{item['rule_id']}" for item in findings if item.get("severity") == "WARN"])

        lint_state = lint_status(findings)
        status = "FAIL" if manifest_errors else lint_state
        return {
            "status": status,
            "manifest_ok": len(manifest_errors) == 0,
            "errors": sorted(set(manifest_errors + rule_errors)),
            "warnings": sorted(set(warnings)),
            "risk_flags": lint_risk_flags(findings),
            "manifest": manifest,
            "manifest_normalized": manifest,
            "compat_warnings": parsed["compat_warnings"],
            "lint_findings": findings,
        }

    def _generate_skill(self, name: str, content: str, warnings: list[str]) -> ToolResult:
        skill_id = self._new_skill_id()
        skill_file = self._quarantine_skill_path(skill_id)
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        rendered = self._inject_default_manifest(name=name, content=content)
        skill_file.write_text(rendered, encoding="utf-8")
        manifest_name, version = self._manifest_name_version(rendered, fallback_name=name)
        payload = {
            "skill_id": skill_id,
            "name": manifest_name or name,
            "state": "quarantined",
            "manifest_ok": True,
            "check_status": "UNKNOWN",
            "warnings": warnings + ["not_active_until_enable"],
            "paths": {"quarantine": str(skill_file)},
        }
        self._append_audit(
            "generate",
            skill_id=skill_id,
            skill_name=manifest_name or name,
            to_state="quarantined",
            hash=self._content_hash(rendered),
            summary=f"generated skill '{manifest_name or name}' in quarantine",
        )
        return self._result(
            ok=True,
            payload=payload,
            warnings=warnings + ["not_active_until_enable"],
            meta={"skill_id": skill_id, "path": str(skill_file), "version": version or "0.1.0"},
        )

    def _check_skill(self, skill_id: str | None, name: str | None) -> ToolResult:
        resolved = self._resolve_quarantine_record(skill_id=skill_id, name=name)
        if resolved is None:
            return ToolResult(ok=False, data="skill not found in quarantine", warnings=["not_found"])
        resolved_id, skill_file = resolved
        content = skill_file.read_text(encoding="utf-8")
        report = self._analyze_content(content, fallback_name=name or resolved_id)
        content_hash = self._content_hash(content)
        check_payload = {
            "skill_id": resolved_id,
            "status": report["status"],
            "hash": content_hash,
            "checked_at": self._iso_now(),
            "errors": report["errors"],
            "warnings": report["warnings"],
            "risk_flags": report["risk_flags"],
            "lint_findings": report["lint_findings"],
        }
        check_payload["signature"] = self._sign_payload(check_payload)
        self._quarantine_check_path(resolved_id).write_text(
            json.dumps(check_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_name = report["manifest"].get("name") or name or resolved_id
        payload = {
            "skill_id": resolved_id,
            "name": manifest_name,
            "state": "quarantined",
            "manifest_ok": report["manifest_ok"],
            "manifest_normalized": report["manifest_normalized"],
            "lint_findings": report["lint_findings"],
            "compat_warnings": report["compat_warnings"],
            "check_status": report["status"],
            "warnings": report["warnings"],
            "errors": report["errors"],
            "risk_flags": report["risk_flags"],
            "paths": {"quarantine": str(skill_file)},
        }
        self._append_audit(
            "check",
            skill_id=resolved_id,
            skill_name=manifest_name,
            hash=content_hash,
            summary=f"checked skill '{manifest_name}' => {report['status']}",
            risk_flags=report["risk_flags"],
        )
        return self._result(
            ok=report["status"] != "FAIL",
            payload=payload,
            warnings=report["warnings"],
            meta={"skill_id": resolved_id, "check_status": report["status"]},
        )

    def _snapshot_enabled(self, name: str, content: str) -> Path:
        target_dir = self.versions_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        snapshot = target_dir / f"{stamp}-{uuid4().hex[:6]}.SKILL.md"
        snapshot.write_text(content, encoding="utf-8")
        return snapshot

    def _read_check_record(self, skill_id: str) -> dict[str, Any] | None:
        check_path = self._quarantine_check_path(skill_id)
        if not check_path.exists():
            return None
        try:
            payload = json.loads(check_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        signature = str(payload.get("signature", "")).strip()
        unsigned = dict(payload)
        unsigned.pop("signature", None)
        if not signature or signature != self._sign_payload(unsigned):
            return None
        return payload

    def _build_diff_payload(
        self,
        *,
        skill_name: str,
        new_content: str,
        new_report: dict[str, Any],
        old_content: str | None,
        old_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return compute_skill_diff(
            skill_name=skill_name,
            new_content=new_content,
            new_manifest=new_report["manifest_normalized"],
            new_findings=new_report["lint_findings"],
            old_content=old_content,
            old_manifest=(old_report or {}).get("manifest_normalized", {}),
            old_findings=(old_report or {}).get("lint_findings", []),
        )

    def _diff_skill(self, skill_id: str | None, name: str | None) -> ToolResult:
        resolved = self._resolve_quarantine_record(skill_id=skill_id, name=name)
        if resolved is None:
            return ToolResult(ok=False, data="skill not found in quarantine", warnings=["not_found"])
        resolved_id, skill_file = resolved
        content = skill_file.read_text(encoding="utf-8")
        report = self._analyze_content(content, fallback_name=name or resolved_id)

        skill_name = str(report["manifest"].get("name") or name or resolved_id).strip()
        if not self._is_valid_skill_name(skill_name):
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": skill_name,
                    "state": "quarantined",
                    "check_status": "FAIL",
                    "manifest_ok": False,
                    "warnings": [],
                    "errors": ["invalid_skill_name"],
                },
                warnings=["check_failed"],
                meta={"skill_id": resolved_id, "check_status": "FAIL"},
            )
        enabled_file = self.enabled_dir / skill_name / "SKILL.md"
        old_content = enabled_file.read_text(encoding="utf-8") if enabled_file.exists() else None
        old_report = self._analyze_content(old_content, fallback_name=skill_name) if old_content is not None else None
        diff_payload = self._build_diff_payload(
            skill_name=skill_name,
            new_content=content,
            new_report=report,
            old_content=old_content,
            old_report=old_report,
        )

        payload = {
            "skill_id": resolved_id,
            "name": skill_name,
            "state": "quarantined",
            "check_status": report["status"],
            "manifest_ok": report["manifest_ok"],
            "manifest_normalized": report["manifest_normalized"],
            "lint_findings": report["lint_findings"],
            "compat_warnings": report["compat_warnings"],
            "risk_flags": report["risk_flags"],
            "warnings": report["warnings"],
            "errors": report["errors"],
            **diff_payload,
            "paths": {"quarantine": str(skill_file), "enabled": str(enabled_file)},
        }
        return self._result(
            ok=report["status"] != "FAIL",
            payload=payload,
            warnings=report["warnings"],
            meta={"skill_id": resolved_id, "check_status": report["status"]},
        )

    def _enable_skill(self, skill_id: str | None, name: str | None) -> ToolResult:
        resolved = self._resolve_quarantine_record(skill_id=skill_id, name=name)
        if resolved is None:
            return ToolResult(ok=False, data="skill not found in quarantine", warnings=["not_found"])
        resolved_id, skill_file = resolved
        content = skill_file.read_text(encoding="utf-8")
        manifest_preview = parse_and_normalize_manifest(content, fallback_name=name or resolved_id)["manifest"]
        if manifest_preview.get("source", {}).get("kind") == "remote":
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": str(manifest_preview.get("name") or name or resolved_id),
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "FAIL",
                    "warnings": [],
                    "errors": ["remote_source_not_installable"],
                    "paths": {"quarantine": str(skill_file)},
                },
                warnings=["check_failed"],
                meta={"skill_id": resolved_id, "check_status": "FAIL"},
            )
        check_record = self._read_check_record(resolved_id)
        content_hash = self._content_hash(content)
        if not check_record:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "UNKNOWN",
                    "warnings": ["check_required"],
                    "errors": ["missing_check_record"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": "UNKNOWN"},
            )
        check_status = str(check_record.get("status", "UNKNOWN"))
        if check_status not in self.ENABLE_ALLOWED_CHECK_STATUS:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": check_status,
                    "warnings": ["check_required"],
                    "errors": ["check_not_passed"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": check_status},
            )
        checked_at_raw = str(check_record.get("checked_at", "")).strip()
        try:
            checked_at = datetime.fromisoformat(checked_at_raw)
        except ValueError:
            checked_at = None
        if checked_at is None:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "UNKNOWN",
                    "warnings": ["check_required"],
                    "errors": ["invalid_check_timestamp"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": "UNKNOWN"},
            )
        age_seconds = int((datetime.now(timezone.utc) - checked_at.astimezone(timezone.utc)).total_seconds())
        if age_seconds > self.CHECK_MAX_AGE_SECONDS:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "STALE",
                    "warnings": ["check_required"],
                    "errors": ["check_too_old"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": "STALE"},
            )
        if str(check_record.get("hash", "")) != content_hash:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "STALE",
                    "warnings": ["check_required"],
                    "errors": ["check_stale_for_current_content"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": "STALE"},
            )

        # Re-read to reduce TOCTOU windows around check->enable.
        content_recheck = skill_file.read_text(encoding="utf-8")
        if self._content_hash(content_recheck) != content_hash:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": name or resolved_id,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "STALE",
                    "warnings": ["check_required"],
                    "errors": ["content_changed_during_enable"],
                    "paths": {"quarantine": str(skill_file), "check_record": str(self._quarantine_check_path(resolved_id))},
                },
                warnings=["check_required"],
                meta={"skill_id": resolved_id, "check_status": "STALE"},
            )
        content = content_recheck

        report = self._analyze_content(content, fallback_name=name or resolved_id)
        if report["status"] == "FAIL":
            payload = {
                "skill_id": resolved_id,
                "name": name or resolved_id,
                "state": "quarantined",
                "manifest_ok": report["manifest_ok"],
                "manifest_normalized": report["manifest_normalized"],
                "lint_findings": report["lint_findings"],
                "compat_warnings": report["compat_warnings"],
                "check_status": report["status"],
                "warnings": report["warnings"],
                "errors": report["errors"],
                "risk_flags": report["risk_flags"],
                "paths": {"quarantine": str(skill_file)},
            }
            return self._result(
                ok=False,
                payload=payload,
                warnings=["check_failed"] + report["warnings"],
                meta={"skill_id": resolved_id, "check_status": report["status"]},
            )

        manifest = report["manifest_normalized"]
        skill_name = str(manifest.get("name") or name or resolved_id).strip()
        if not self._is_valid_skill_name(skill_name):
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": skill_name,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "FAIL",
                    "warnings": [],
                    "errors": ["invalid_skill_name"],
                    "paths": {"quarantine": str(skill_file)},
                },
                warnings=["check_failed"],
                meta={"skill_id": resolved_id, "check_status": "FAIL"},
            )
        if skill_name in self.known_tool_names:
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": skill_name,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "FAIL",
                    "warnings": [],
                    "errors": ["reserved_skill_name_conflict"],
                    "paths": {"quarantine": str(skill_file)},
                },
                warnings=["check_failed"],
                meta={"skill_id": resolved_id, "check_status": "FAIL"},
            )
        if manifest.get("source", {}).get("kind") == "remote":
            return self._result(
                ok=False,
                payload={
                    "skill_id": resolved_id,
                    "name": skill_name,
                    "state": "quarantined",
                    "manifest_ok": False,
                    "check_status": "FAIL",
                    "warnings": [],
                    "errors": ["remote_source_not_installable"],
                    "paths": {"quarantine": str(skill_file)},
                },
                warnings=["check_failed"],
                meta={"skill_id": resolved_id, "check_status": "FAIL"},
            )
        version = str(manifest.get("version") or "0.0.0")
        source_kind = str(manifest.get("source", {}).get("kind") or "local")
        enabled_file = self.enabled_dir / skill_name / "SKILL.md"
        enabled_file.parent.mkdir(parents=True, exist_ok=True)
        built_in_override = bool(self.skills_dir and (self.skills_dir / skill_name / "SKILL.md").exists())

        old_content = enabled_file.read_text(encoding="utf-8") if enabled_file.exists() else None
        old_report = self._analyze_content(old_content, fallback_name=skill_name) if old_content is not None else None
        old_hash = self._content_hash(old_content) if old_content is not None else None
        old_version = str((old_report or {}).get("manifest_normalized", {}).get("version") or "") or None

        diff_payload = self._build_diff_payload(
            skill_name=skill_name,
            new_content=content,
            new_report=report,
            old_content=old_content,
            old_report=old_report,
        )

        previous_snapshot = None
        if old_content is not None:
            previous_snapshot = self._snapshot_enabled(skill_name, old_content)

        enabled_file.write_text(content.rstrip() + "\n", encoding="utf-8")
        pin = self._to_pin(skill_name, version, content_hash)
        lock = self._read_lock()
        lock[skill_name] = {
            "version": version,
            "hash": content_hash,
            "pin": pin,
            "source": source_kind,
            "skill_id": resolved_id,
            "updated_at": self._iso_now(),
        }
        self._write_lock(lock)
        self._append_audit(
            "enable",
            skill_id=resolved_id,
            skill_name=skill_name,
            to_state="enabled",
            from_version=old_version,
            to_version=version,
            from_hash=old_hash,
            to_hash=content_hash,
            pin=pin,
            summary=f"enabled skill '{skill_name}'",
            risk_flags=report["risk_flags"],
            diff_summary=diff_payload["diff_summary"],
        )

        payload = {
            "skill_id": resolved_id,
            "name": skill_name,
            "state": "enabled",
            "manifest_ok": report["manifest_ok"],
            "manifest_normalized": report["manifest_normalized"],
            "lint_findings": report["lint_findings"],
            "compat_warnings": report["compat_warnings"],
            "check_status": report["status"],
            "warnings": report["warnings"],
            "risk_flags": report["risk_flags"],
            "pin": pin,
            "built_in_override": built_in_override,
            **diff_payload,
            "paths": {
                "quarantine": str(skill_file),
                "enabled": str(enabled_file),
                "previous_snapshot": str(previous_snapshot) if previous_snapshot else None,
            },
        }
        if built_in_override:
            payload["warnings"].append("overrides_built_in_skill")
        return self._result(
            ok=True,
            payload=payload,
            warnings=report["warnings"],
            meta={"skill_id": resolved_id, "path": str(enabled_file), "version": version, "pin": pin},
        )

    def _disable_skill(self, name: str) -> ToolResult:
        enabled_file = self.enabled_dir / name / "SKILL.md"
        if not enabled_file.exists():
            return ToolResult(ok=False, data=f"enabled skill '{name}' not found", warnings=["not_found"])

        content = enabled_file.read_text(encoding="utf-8")
        snapshot = self._snapshot_enabled(name, content)
        enabled_file.unlink()
        if not any(enabled_file.parent.iterdir()):
            enabled_file.parent.rmdir()

        lock = self._read_lock()
        lock.pop(name, None)
        self._write_lock(lock)
        self._append_audit(
            "disable",
            skill_name=name,
            to_state="disabled",
            hash=self._content_hash(content),
            summary=f"disabled skill '{name}'",
        )
        payload = {
            "skill_id": None,
            "name": name,
            "state": "disabled",
            "manifest_ok": True,
            "check_status": "N/A",
            "warnings": [],
            "paths": {"snapshot": str(snapshot), "enabled": str(enabled_file)},
        }
        return self._result(ok=True, payload=payload, meta={"snapshot": str(snapshot)})

    def _rollback_skill(self, name: str, target: str | None) -> ToolResult:
        snapshots = sorted((self.versions_dir / name).glob("*.SKILL.md")) if (self.versions_dir / name).exists() else []
        if not snapshots:
            return ToolResult(ok=False, data=f"no snapshots found for '{name}'", warnings=["no_snapshot"])

        selected: Path | None = None
        if target:
            for snapshot in snapshots:
                if snapshot.name == target or snapshot.stem == target or target in snapshot.name:
                    selected = snapshot
                    break
            if selected is None:
                return ToolResult(ok=False, data=f"target snapshot not found: {target}", warnings=["target_not_found"])
        else:
            selected = snapshots[-1]

        enabled_file = self.enabled_dir / name / "SKILL.md"
        enabled_file.parent.mkdir(parents=True, exist_ok=True)

        old_content = enabled_file.read_text(encoding="utf-8") if enabled_file.exists() else None
        old_hash = self._content_hash(old_content) if old_content is not None else None
        old_name, old_version = self._manifest_name_version(old_content or "", fallback_name=name)
        previous_snapshot = None
        if old_content is not None:
            previous_snapshot = self._snapshot_enabled(name, old_content)

        restored = selected.read_text(encoding="utf-8")
        restored_report = self._analyze_content(restored, fallback_name=name)
        if restored_report["status"] == "FAIL":
            return self._result(
                ok=False,
                payload={
                    "skill_id": None,
                    "name": name,
                    "state": "enabled",
                    "manifest_ok": False,
                    "check_status": "FAIL",
                    "warnings": restored_report["warnings"],
                    "errors": ["rollback_snapshot_failed_check"] + restored_report["errors"],
                    "risk_flags": restored_report["risk_flags"],
                    "paths": {"restored_snapshot": str(selected)},
                },
                warnings=["rollback_blocked_by_check"],
                meta={"snapshot": str(selected)},
            )
        enabled_file.write_text(restored.rstrip() + "\n", encoding="utf-8")
        resolved_name, version = self._manifest_name_version(restored, fallback_name=name)
        final_name = resolved_name or name
        if not self._is_valid_skill_name(final_name):
            return ToolResult(ok=False, data="invalid skill name in snapshot", warnings=["invalid_name"])
        final_version = version or "0.0.0"
        content_hash = self._content_hash(restored)
        pin = self._to_pin(final_name, final_version, content_hash)

        lock = self._read_lock()
        if name != final_name:
            lock.pop(name, None)
        lock[final_name] = {
            "version": final_version,
            "hash": content_hash,
            "pin": pin,
            "source": "local",
            "skill_id": None,
            "updated_at": self._iso_now(),
        }
        self._write_lock(lock)
        self._append_audit(
            "rollback",
            skill_name=final_name,
            to_state="enabled",
            from_version=old_version,
            to_version=final_version,
            from_hash=old_hash,
            to_hash=content_hash,
            pin=pin,
            hash=content_hash,
            summary=f"rolled back skill '{final_name}' to snapshot {selected.name}",
        )
        payload = {
            "skill_id": None,
            "name": final_name,
            "state": "enabled",
            "manifest_ok": version is not None,
            "check_status": "N/A",
            "warnings": [],
            "pin": pin,
            "paths": {
                "enabled": str(enabled_file),
                "restored_snapshot": str(selected),
                "previous_snapshot": str(previous_snapshot) if previous_snapshot else None,
            },
        }
        return self._result(
            ok=True,
            payload=payload,
            meta={"snapshot": str(selected), "version": final_version, "pin": pin},
        )

    def _remove_legacy(self, skill_id: str | None, name: str | None) -> ToolResult:
        removed_paths: list[str] = []

        if skill_id:
            candidate = self.quarantine_dir / skill_id
            if candidate.exists():
                shutil.rmtree(candidate)
                removed_paths.append(str(candidate))

        if name:
            legacy_dir = self.skills_local_dir / name
            if legacy_dir.is_dir() and name not in self.RESERVED_ROOT_NAMES:
                skill_file = legacy_dir / "SKILL.md"
                if skill_file.exists():
                    shutil.rmtree(legacy_dir)
                    removed_paths.append(str(legacy_dir))

            for candidate_id, candidate_path in self._iter_quarantine_records():
                content = candidate_path.read_text(encoding="utf-8")
                manifest_name, _ = self._manifest_name_version(content, fallback_name=candidate_id)
                if manifest_name == name or candidate_id == name:
                    parent = candidate_path.parent
                    shutil.rmtree(parent)
                    removed_paths.append(str(parent))

        if not removed_paths:
            return ToolResult(ok=False, data="no matching skill found to remove", warnings=["not_found"])

        self._append_audit(
            "remove",
            skill_id=skill_id,
            skill_name=name,
            summary=f"removed {len(removed_paths)} skill path(s)",
            removed_paths=removed_paths,
        )
        payload = {
            "skill_id": skill_id,
            "name": name,
            "state": "removed",
            "manifest_ok": True,
            "check_status": "N/A",
            "warnings": ["legacy_action_remove"],
            "paths": {"removed": removed_paths},
        }
        return self._result(
            ok=True,
            payload=payload,
            warnings=["legacy_action_remove"],
            meta={"removed_count": len(removed_paths)},
        )

    def _list_skills(self) -> ToolResult:
        quarantine: list[dict[str, Any]] = []
        for candidate_id, candidate_path in self._iter_quarantine_records():
            content = candidate_path.read_text(encoding="utf-8")
            name, version = self._manifest_name_version(content, fallback_name=candidate_id)
            quarantine.append(
                {
                    "skill_id": candidate_id,
                    "name": name or candidate_id,
                    "version": version,
                    "path": str(candidate_path),
                }
            )

        views = resolve_skill_views(skills_dir=self.skills_dir, skills_local_dir=self.skills_local_dir)

        legacy: list[dict[str, Any]] = []
        for entry in sorted(self.skills_local_dir.iterdir()):
            if not entry.is_dir() or entry.name in self.RESERVED_ROOT_NAMES:
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            legacy.append({"name": entry.name, "path": str(skill_file)})

        payload = {
            "quarantine": quarantine,
            "enabled": views["local_enabled"],
            "legacy": legacy,
            "built_in": views["built_in"],
            "local_enabled": views["local_enabled"],
            "effective": views["effective"],
            "lock_count": len(self._read_lock()),
            "lock_drift_warnings": self._lock_drift_warnings(),
        }
        return self._result(
            ok=True,
            payload=payload,
            warnings=payload["lock_drift_warnings"],
            meta={
                "quarantine_count": len(quarantine),
                "enabled_count": len(views["local_enabled"]),
                "legacy_count": len(legacy),
            },
        )

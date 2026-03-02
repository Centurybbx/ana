from __future__ import annotations

import re
from typing import Any

STRICT_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
ALLOWED_SOURCE_KINDS = {"built-in", "local", "remote"}
SKILL_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    return value


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_list(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    index = start
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        current_indent = _line_indent(line)
        if current_indent < indent:
            break
        if current_indent != indent or not stripped.startswith("- "):
            break
        items.append(_parse_scalar(stripped[2:].strip()))
        index += 1
    return items, index


def _parse_mapping(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    parsed: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        current_indent = _line_indent(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            index += 1
            continue
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            index += 1
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value:
            parsed[key] = _parse_scalar(value)
            index += 1
            continue

        cursor = index + 1
        while cursor < len(lines):
            next_line = lines[cursor]
            next_stripped = next_line.strip()
            if not next_stripped or next_stripped.startswith("#"):
                cursor += 1
                continue
            next_indent = _line_indent(next_line)
            if next_indent <= current_indent:
                parsed[key] = {}
                index = cursor
                break
            if next_stripped.startswith("- "):
                items, next_index = _parse_list(lines, cursor, next_indent)
                parsed[key] = items
                index = next_index
                break
            child, next_index = _parse_mapping(lines, cursor, next_indent)
            parsed[key] = child
            index = next_index
            break
        else:
            parsed[key] = {}
            index = cursor
    return parsed, index


def parse_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, content
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None, content

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    parsed, _ = _parse_mapping(frontmatter_lines, 0, 0)
    return parsed, body


def parse_and_normalize_manifest(content: str, fallback_name: str | None = None) -> dict[str, Any]:
    manifest_raw, body = parse_frontmatter(content)
    errors: list[str] = []
    warnings: list[str] = []
    compat_warnings: list[str] = []
    normalized: dict[str, Any] = {}

    if manifest_raw is None:
        errors.append("missing_manifest_frontmatter")
        return {
            "manifest": {},
            "body": body,
            "errors": errors,
            "warnings": warnings,
            "compat_warnings": compat_warnings,
        }

    name = str(manifest_raw.get("name") or fallback_name or "").strip()
    if not name:
        errors.append("missing_field:name")
    elif not SKILL_NAME_RE.fullmatch(name):
        errors.append("invalid_skill_name")
    normalized["name"] = name or None

    version = str(manifest_raw.get("version") or "").strip()
    if not version:
        errors.append("missing_field:version")
    elif not STRICT_SEMVER_RE.match(version):
        errors.append("invalid_semver_version")
    normalized["version"] = version or None

    allowed = manifest_raw.get("allowed_tool_names")
    if allowed is None and "allowed_tools" in manifest_raw:
        allowed = manifest_raw.get("allowed_tools")
        compat_warnings.append("legacy_allowed_tools_mapped")
    if allowed is None:
        errors.append("missing_field:allowed_tool_names")
        allowed = []
    if not isinstance(allowed, list):
        errors.append("invalid_allowed_tool_names_type")
        allowed = []
    normalized["allowed_tool_names"] = [str(item).strip() for item in allowed if str(item).strip()]

    required = manifest_raw.get("required_capabilities")
    if required is None:
        errors.append("missing_field:required_capabilities")
        required = []
    if not isinstance(required, list):
        errors.append("invalid_required_capabilities_type")
        required = []
    normalized["required_capabilities"] = [str(item).strip() for item in required if str(item).strip()]

    source = manifest_raw.get("source")
    if source is None:
        source = {"kind": "local", "uri": None}
    if not isinstance(source, dict):
        errors.append("invalid_source_field")
        source = {"kind": "local", "uri": None}
    source_kind = str(source.get("kind") or "local").strip()
    source_uri = source.get("uri")
    if source_kind not in ALLOWED_SOURCE_KINDS:
        errors.append("invalid_source_kind")
    normalized["source"] = {"kind": source_kind, "uri": source_uri}

    constraints = manifest_raw.get("constraints")
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, dict):
        errors.append("invalid_constraints_field")
        constraints = {}
    normalized["constraints"] = constraints

    external_links_policy = str(manifest_raw.get("external_links_policy") or "deny").strip().lower()
    if external_links_policy not in {"deny", "allowlist"}:
        warnings.append("invalid_external_links_policy_fallback_to_deny")
        external_links_policy = "deny"
    normalized["external_links_policy"] = external_links_policy

    allowlist = manifest_raw.get("external_links_allowlist", [])
    if allowlist is None:
        allowlist = []
    if not isinstance(allowlist, list):
        errors.append("invalid_external_links_allowlist_type")
        allowlist = []
    normalized["external_links_allowlist"] = [str(item).strip().lower() for item in allowlist if str(item).strip()]

    summary = manifest_raw.get("summary")
    normalized["summary"] = str(summary).strip() if summary is not None else None
    normalized["hash"] = manifest_raw.get("hash")

    return {
        "manifest": normalized,
        "body": body,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "compat_warnings": sorted(set(compat_warnings)),
    }

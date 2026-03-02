from __future__ import annotations

from pathlib import Path
from typing import Any

from ana.tools.skill_manifest import parse_and_normalize_manifest


def _scan_skill_root(root: Path | None, *, default_source_kind: str) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue
        content = skill_file.read_text(encoding="utf-8")
        parsed = parse_and_normalize_manifest(content, fallback_name=entry.name)
        manifest = parsed["manifest"]
        source = manifest.get("source", {})
        source_kind = str(source.get("kind") or default_source_kind).strip() or default_source_kind
        items.append(
            {
                "name": str(manifest.get("name") or entry.name),
                "version": manifest.get("version"),
                "allowed_tool_names": list(manifest.get("allowed_tool_names", [])),
                "required_capabilities": list(manifest.get("required_capabilities", [])),
                "source": {"kind": source_kind, "uri": source.get("uri")},
                "path": str(skill_file),
            }
        )
    return items


def _resolve_effective(local_enabled: list[dict[str, Any]], built_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for item in built_in:
        by_name[item["name"]] = item
    for item in local_enabled:
        by_name[item["name"]] = item
    return [by_name[name] for name in sorted(by_name)]


def resolve_skill_views(skills_dir: Path | None, skills_local_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    built_in = _scan_skill_root(skills_dir, default_source_kind="built-in")
    local_enabled = _scan_skill_root(
        (skills_local_dir / "enabled") if skills_local_dir else None,
        default_source_kind="local",
    )
    effective = _resolve_effective(local_enabled=local_enabled, built_in=built_in)
    return {
        "built_in": built_in,
        "local_enabled": local_enabled,
        "effective": effective,
    }

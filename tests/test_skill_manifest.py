from __future__ import annotations

from ana.tools.skill_manifest import parse_and_normalize_manifest


def test_manifest_rejects_non_semver_version():
    payload = parse_and_normalize_manifest(
        """---
name: demo
version: 1
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

body
"""
    )
    assert "invalid_semver_version" in payload["errors"]


def test_manifest_rejects_invalid_skill_name():
    payload = parse_and_normalize_manifest(
        """---
name: ../bad
version: 1.2.3
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---
"""
    )
    assert "invalid_skill_name" in payload["errors"]


def test_manifest_rejects_invalid_source_kind():
    payload = parse_and_normalize_manifest(
        """---
name: demo
version: 1.2.3
source:
  kind: internet
allowed_tool_names:
  - read_file
required_capabilities:
  - fs.read
---

body
"""
    )
    assert "invalid_source_kind" in payload["errors"]


def test_manifest_requires_list_types_for_governance_fields():
    payload = parse_and_normalize_manifest(
        """---
name: demo
version: 1.2.3
allowed_tool_names: read_file
required_capabilities: fs.read
---

body
"""
    )
    assert "invalid_allowed_tool_names_type" in payload["errors"]
    assert "invalid_required_capabilities_type" in payload["errors"]


def test_manifest_maps_legacy_allowed_tools_with_warning():
    payload = parse_and_normalize_manifest(
        """---
name: demo
version: 1.2.3
allowed_tools:
  - read_file
required_capabilities:
  - fs.read
---

body
"""
    )
    assert not payload["errors"]
    assert payload["manifest"]["allowed_tool_names"] == ["read_file"]
    assert "legacy_allowed_tools_mapped" in payload["compat_warnings"]

from __future__ import annotations

from pathlib import Path

from ana.config import AnaConfig


def test_resolved_evolution_interval_seconds_presets_and_custom(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    daily = AnaConfig(workspace_dir=workspace, evolution_schedule="daily")
    weekly = AnaConfig(workspace_dir=workspace, evolution_schedule="weekly")
    custom = AnaConfig(workspace_dir=workspace, evolution_schedule="custom", evolution_custom_interval_minutes=90)

    assert daily.resolved_evolution_interval_seconds() == 24 * 60 * 60
    assert weekly.resolved_evolution_interval_seconds() == 7 * 24 * 60 * 60
    assert custom.resolved_evolution_interval_seconds() == 90 * 60


def test_resolved_evolution_dataset_overrides_are_workspace_relative(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = AnaConfig(
        workspace_dir=workspace,
        evolution_dataset_overrides={"core_regression": "benchmarks/custom_core.json"},
    )
    resolved = cfg.resolved_evolution_dataset_overrides()
    assert "core_regression" in resolved
    assert resolved["core_regression"] == (workspace / "benchmarks" / "custom_core.json").resolve()


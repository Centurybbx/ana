from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ana.cli import app
from ana.evals import load_benchmark_tasks


def test_core_regression_dataset_has_minimum_examples():
    tasks_file = Path("benchmarks/core.json")
    assert tasks_file.exists()

    tasks = load_benchmark_tasks(tasks_file)
    assert len(tasks) >= 20

    ids = [str(item.get("id", "")) for item in tasks]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_core_regression_dataset_runs_with_mock_provider(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
        "runtime_log_enabled": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    out_file = workspace / "core.eval.result.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval-run",
            "--provider",
            "mock",
            "--config-path",
            str(config_path),
            "--tasks",
            "benchmarks/core.json",
            "--output",
            str(out_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["aggregate"]["examples_total"] >= 20
    assert payload["aggregate"]["success_rate"] >= 0.85
    assert payload["regression_pass"] is True


def test_evals_dataset_catalog_includes_failure_replay_and_canary():
    failure_replay = Path("benchmarks/failure_replay.json")
    canary = Path("benchmarks/canary.json")
    assert failure_replay.exists()
    assert canary.exists()

    failure_tasks = load_benchmark_tasks(failure_replay)
    canary_tasks = load_benchmark_tasks(canary)
    assert len(failure_tasks) >= 5
    assert len(canary_tasks) >= 5

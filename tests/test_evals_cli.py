from __future__ import annotations

import json

from typer.testing import CliRunner

from ana.cli import app
from ana.tools.memory import MemoryStore


def _write_config(tmp_path, workspace):
    config = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
        "runtime_log_enabled": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_eval_export_and_metrics_cli(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = _write_config(tmp_path, workspace)

    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "turn_start",
            "event_type": "turn_start",
            "user_text_excerpt": "hello",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "assistant_final",
            "event_type": "assistant_final",
            "observation": "done",
        }
    )

    runner = CliRunner()
    dataset_file = workspace / "memory" / "eval.dataset.jsonl"
    export_result = runner.invoke(
        app,
        ["eval-export", "--config-path", str(config_path), "--output", str(dataset_file)],
    )
    assert export_result.exit_code == 0
    assert dataset_file.exists()
    assert len(dataset_file.read_text(encoding="utf-8").splitlines()) == 1

    metrics_file = workspace / "memory" / "eval.metrics.json"
    metrics_result = runner.invoke(
        app,
        ["eval-metrics", "--config-path", str(config_path), "--output", str(metrics_file)],
    )
    assert metrics_result.exit_code == 0
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert payload["turns_total"] == 1
    assert "success_rate" in payload


def test_eval_run_cli_with_baseline_gate(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = _write_config(tmp_path, workspace)

    pass_tasks = [
        {"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}},
        {"id": "t2", "input": {"user_text": "beta"}, "expected": {"contains": ["[mock] beta"]}},
    ]
    fail_tasks = [
        {"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}},
        {"id": "t2", "input": {"user_text": "beta"}, "expected": {"contains": ["NOT-THERE"]}},
    ]
    pass_file = workspace / "bench.pass.json"
    fail_file = workspace / "bench.fail.json"
    pass_file.write_text(json.dumps(pass_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    fail_file.write_text(json.dumps(fail_tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    runner = CliRunner()
    baseline_out = workspace / "eval.pass.result.json"
    pass_run = runner.invoke(
        app,
        [
            "eval-run",
            "--provider",
            "mock",
            "--config-path",
            str(config_path),
            "--tasks",
            str(pass_file),
            "--output",
            str(baseline_out),
        ],
    )
    assert pass_run.exit_code == 0
    baseline_payload = json.loads(baseline_out.read_text(encoding="utf-8"))
    assert baseline_payload["regression_pass"] is True

    fail_out = workspace / "eval.fail.result.json"
    fail_run = runner.invoke(
        app,
        [
            "eval-run",
            "--provider",
            "mock",
            "--config-path",
            str(config_path),
            "--tasks",
            str(fail_file),
            "--output",
            str(fail_out),
            "--baseline",
            str(baseline_out),
            "--latency-regression-pct",
            "0",
        ],
    )
    assert fail_run.exit_code == 2
    fail_payload = json.loads(fail_out.read_text(encoding="utf-8"))
    assert fail_payload["regression_pass"] is False
    assert fail_payload["gate_failures"]

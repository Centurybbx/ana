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


def test_evolve_analyze_generates_failure_reports(tmp_path):
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
            "user_text_excerpt": "try dangerous shell",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "guardrail_tripwire",
            "event_type": "guardrail_tripwire",
            "tool": "shell",
            "reason": "blocked by policy",
        }
    )
    memory.append_trace(
        {
            "session_id": "s1",
            "trace_id": "t1",
            "turn_index": 1,
            "event": "max_steps_exhausted",
            "event_type": "max_steps_exhausted",
            "observation": "Reached max steps without completing task.",
        }
    )

    out_file = workspace / "memory" / "failure_reports.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evolve-analyze", "--config-path", str(config_path), "--output", str(out_file)],
    )
    assert result.exit_code == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["reports"]
    first = payload["reports"][0]
    assert first["trace_id"] == "t1"
    assert "root_cause" in first
    assert "recommended_fix_type" in first


def test_evolve_propose_validate_approve_and_rollback(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = _write_config(tmp_path, workspace)
    failure_file = workspace / "memory" / "failure_reports.json"
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    failure_file.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "trace_id": "t1",
                        "tags": ["policy_loop"],
                        "root_cause": "policy_loop",
                        "evidence": ["blocked by policy"],
                        "recommended_fix_type": "skill_patch",
                        "risk_level": "R1",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    propose = runner.invoke(
        app,
        [
            "evolve-propose",
            "--config-path",
            str(config_path),
            "--failures",
            str(failure_file),
            "--skill-name",
            "planner",
        ],
    )
    assert propose.exit_code == 0
    proposal_result = json.loads(propose.stdout.strip())
    proposal_id = proposal_result["proposal_id"]
    proposal_file = workspace / "proposals" / proposal_id / "proposal.json"
    assert proposal_file.exists()

    tasks = [
        {"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}},
        {"id": "t2", "input": {"user_text": "beta"}, "expected": {"contains": ["[mock] beta"]}},
    ]
    tasks_file = workspace / "bench.pass.json"
    tasks_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    validate = runner.invoke(
        app,
        [
            "evolve-validate",
            "--config-path",
            str(config_path),
            "--proposal-id",
            proposal_id,
            "--tasks",
            str(tasks_file),
            "--provider",
            "mock",
        ],
    )
    assert validate.exit_code == 0

    approve = runner.invoke(
        app,
        ["evolve-deploy", "--config-path", str(config_path), "--proposal-id", proposal_id],
    )
    assert approve.exit_code == 0

    deployed = json.loads(proposal_file.read_text(encoding="utf-8"))
    assert deployed["status"] == "deployed"
    enabled_skill = workspace / "skills_local" / "enabled" / "planner" / "SKILL.md"
    assert enabled_skill.exists()

    rollback = runner.invoke(
        app,
        ["evolve-rollback", "--config-path", str(config_path), "--skill-name", "planner"],
    )
    assert rollback.exit_code == 0


def test_evolve_approve_requires_validation_pass(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = _write_config(tmp_path, workspace)

    failure_file = workspace / "memory" / "failure_reports.json"
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    failure_file.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "trace_id": "t1",
                        "tags": ["missing_skill"],
                        "root_cause": "missing_skill",
                        "evidence": ["no skill"],
                        "recommended_fix_type": "skill_patch",
                        "risk_level": "R1",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    propose = runner.invoke(
        app,
        [
            "evolve-propose",
            "--config-path",
            str(config_path),
            "--failures",
            str(failure_file),
            "--skill-name",
            "planner",
        ],
    )
    assert propose.exit_code == 0
    proposal_id = json.loads(propose.stdout.strip())["proposal_id"]

    approve = runner.invoke(
        app,
        ["evolve-deploy", "--config-path", str(config_path), "--proposal-id", proposal_id],
    )
    assert approve.exit_code == 1

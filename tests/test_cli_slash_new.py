from __future__ import annotations

import json

from typer.testing import CliRunner

from ana.cli import app


def test_cli_slash_new_switches_session(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", "--provider", "mock", "--config-path", str(config_path)],
        input="/new\n/exit\n",
    )

    assert result.exit_code == 0
    assert result.stdout.count("Session:") >= 2


def test_chat_starts_auto_evolution_scheduler_from_config(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
        "runtime_log_enabled": False,
        "evolution_enabled": True,
        "evolution_schedule": "custom",
        "evolution_custom_interval_minutes": 1,
        "evolution_run_on_startup": False,
        "evolution_skill_name": "planner",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", "--provider", "mock", "--config-path", str(config_path)],
        input="/exit\n",
    )

    assert result.exit_code == 0
    assert "Auto evolution enabled" in result.stdout

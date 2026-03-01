from __future__ import annotations

import json

from typer.testing import CliRunner

from aha.cli import app


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

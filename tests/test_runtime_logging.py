from __future__ import annotations

import json
import logging
from pathlib import Path

from typer.testing import CliRunner

from aha.cli import app
from aha.config import AhaConfig
from aha.runtime_log import configure_runtime_logging, log_event


def _flush_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def test_runtime_log_creates_file_and_redacts(tmp_path):
    workspace = tmp_path / "workspace"
    config = AhaConfig(
        workspace_dir=workspace,
        runtime_log_dir=Path("logs"),
        runtime_log_mode="debug",
    )
    logger, log_path = configure_runtime_logging(config, interactive=True)
    assert log_path is not None

    log_event(
        logger.getChild("test"),
        "redaction_test",
        level=logging.INFO,
        api_key="sk-ABCDEF0123456789ABCDEF0123456789",
        auth="Bearer abc.def.ghi",
    )
    _flush_handlers(logger)

    content = log_path.read_text(encoding="utf-8")
    assert "sk-ABCDEF0123456789ABCDEF0123456789" not in content
    assert "Bearer abc.def.ghi" not in content
    assert "api_key=[REDACTED]" in content
    assert "Bearer [REDACTED]" in content


def test_runtime_log_level_debug_vs_prod(tmp_path):
    prod_workspace = tmp_path / "prod"
    prod_config = AhaConfig(workspace_dir=prod_workspace, runtime_log_mode="prod")
    prod_logger, prod_log_path = configure_runtime_logging(prod_config, interactive=True)
    assert prod_log_path is not None

    log_event(prod_logger.getChild("test"), "debug_hidden", level=logging.DEBUG, value="x")
    log_event(prod_logger.getChild("test"), "info_visible", level=logging.INFO, value="y")
    _flush_handlers(prod_logger)

    prod_content = prod_log_path.read_text(encoding="utf-8")
    assert "event=info_visible" in prod_content
    assert "event=debug_hidden" not in prod_content

    debug_workspace = tmp_path / "debug"
    debug_config = AhaConfig(workspace_dir=debug_workspace, runtime_log_mode="debug")
    debug_logger, debug_log_path = configure_runtime_logging(debug_config, interactive=True)
    assert debug_log_path is not None

    log_event(debug_logger.getChild("test"), "debug_visible", level=logging.DEBUG, value="z")
    _flush_handlers(debug_logger)

    debug_content = debug_log_path.read_text(encoding="utf-8")
    assert "event=debug_visible" in debug_content


def test_cli_debug_flag_writes_start_event(tmp_path):
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
        ["chat", "--provider", "mock", "--config-path", str(config_path), "--debug"],
        input="/exit\n",
    )

    assert result.exit_code == 0
    log_path = workspace / "logs" / "aha.debug.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "event=chat_start" in content

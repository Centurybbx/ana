from __future__ import annotations

import json
from pathlib import Path

from ana.cli import run_auto_evolution_cycle
from ana.config import load_config
from ana.evolve import proposals_dir
from ana.tools.memory import MemoryStore


def test_run_auto_evolution_cycle_generates_validated_proposal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_payload = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
        "runtime_log_enabled": False,
        "evolution_enabled": True,
        "evolution_skill_name": "planner",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    config = load_config(path=config_path)

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
            "user_text_excerpt": "do risky thing",
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

    result = run_auto_evolution_cycle(config=config, config_path=config_path)
    assert result["reports"] >= 1
    assert result["status"] in {"validated", "validation_failed"}
    assert "proposal_id" in result

    root = proposals_dir(workspace)
    proposal_file = root / result["proposal_id"] / "proposal.json"
    assert proposal_file.exists()
    payload = json.loads(proposal_file.read_text(encoding="utf-8"))
    assert payload["status"] in {"validated", "validation_failed"}
    assert payload.get("validation")
    assert payload["status"] != "deployed"

    summary = workspace / "memory" / "evolution" / "latest.json"
    assert summary.exists()


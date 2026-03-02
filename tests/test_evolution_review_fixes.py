from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

from ana.cli import app
from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session
from ana.evals import run_offline_eval
from ana.evolve import build_failure_reports, load_proposal, save_proposal
from ana.providers.base import LLMResponse, LLMToolCall
from ana.tools.fs import ReadFileTool, WriteFileTool
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner


def _write_config(tmp_path: Path, workspace: Path) -> Path:
    config = {
        "provider": "mock",
        "model": "mock-model",
        "workspace_dir": str(workspace),
        "runtime_log_enabled": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_trace_events_include_spans_and_usage_metrics(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )

    class _Provider:
        async def complete(self, messages, tools):
            return LLMResponse(
                content="done",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "done"},
                usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20, "cost_estimate": 0.0012},
            )

    registry = ToolRegistry()
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy)
    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
        max_steps=3,
    )
    session = Session(session_id="s1", created_at="2026-03-01T00:00:00+00:00")
    state = RuntimeSessionState()

    async def approve(_: str) -> bool:
        return True

    reply = asyncio.run(loop.run_turn(session, "hello", session_state=state, confirm=approve))
    assert reply == "done"

    events = memory.tail_trace(limit=50)
    llm_response = [item for item in events if item.get("event_type") == "llm_response"][-1]
    assert llm_response.get("span_id")
    assert llm_response.get("parent_span_id")

    metrics = memory.summarize_trace_metrics()
    assert metrics["token_usage_total"] == 20
    assert metrics["cost_estimate_total"] == 0.0012


def test_build_eval_records_caps_event_payload(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=True,
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
            "user_text_excerpt": "x",
        }
    )
    for i in range(250):
        memory.append_trace(
            {
                "session_id": "s1",
                "trace_id": "t1",
                "turn_index": 1,
                "event": "tool_result",
                "event_type": "tool_result",
                "tool": "read_file",
                "status": "ok",
                "step": i + 1,
                "observation": f"line-{i}",
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

    rows = memory.build_eval_records(force_redact=True)
    assert len(rows) == 1
    first = rows[0]
    assert len(first["trajectory"]["events"]) <= 120
    assert int(first["metrics"]["events_truncated"]) > 0


def test_failure_taxonomy_covers_missing_labels():
    payload = build_failure_reports(
        [
            {
                "trace_id": "t-format",
                "outcome": {"status": "fail", "reason": "invalid_output_format"},
                "metrics": {},
                "trajectory": {"events": []},
            },
            {
                "trace_id": "t-wrong-tool",
                "outcome": {"status": "fail", "reason": "tool_error"},
                "metrics": {},
                "trajectory": {
                    "events": [
                        {"event_type": "tool_result", "tool": "read_file", "status": "error", "reason": "tool_not_found"}
                    ]
                },
            },
            {
                "trace_id": "t-skill-outdated",
                "outcome": {"status": "fail", "reason": "tool_error"},
                "metrics": {},
                "trajectory": {
                    "events": [
                        {
                            "event_type": "tool_result",
                            "tool": "web_fetch",
                            "status": "error",
                            "reason": "schema_mismatch",
                        }
                    ]
                },
            },
        ]
    )
    root_causes = {item["trace_id"]: item["root_cause"] for item in payload["reports"]}
    assert root_causes["t-format"] == "format_failure"
    assert root_causes["t-wrong-tool"] == "wrong_tool"
    assert root_causes["t-skill-outdated"] == "skill_outdated"


def test_evolve_deploy_rejects_tampered_candidate_after_validation(tmp_path):
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

    tasks = [{"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}}]
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

    proposal_file, proposal = load_proposal(workspace, proposal_id)
    proposal["candidate"]["content"] = proposal["candidate"]["content"] + "\n# tampered\n"
    save_proposal(proposal_file, proposal)

    approve = runner.invoke(
        app,
        ["evolve-deploy", "--config-path", str(config_path), "--proposal-id", proposal_id],
    )
    assert approve.exit_code == 1


def test_evolve_reject_records_audit_metadata(tmp_path):
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
            }
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

    rejected = runner.invoke(
        app,
        [
            "evolve-reject",
            "--config-path",
            str(config_path),
            "--proposal-id",
            proposal_id,
            "--reason",
            "insufficient_evidence",
            "--actor",
            "qa-reviewer",
        ],
    )
    assert rejected.exit_code == 0

    proposal_file = workspace / "proposals" / proposal_id / "proposal.json"
    payload = json.loads(proposal_file.read_text(encoding="utf-8"))
    assert payload["rejection"]["actor"] == "qa-reviewer"
    assert payload["rejection"]["reason"] == "insufficient_evidence"

    audit_file = workspace / "proposals" / "audit.jsonl"
    rows = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("action") == "reject" and row.get("proposal_id") == proposal_id for row in rows)


def test_evolve_deploy_requires_r3_double_confirmation(tmp_path):
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
                        "evidence": ["consent_denials=1"],
                        "recommended_fix_type": "skill_patch",
                        "risk_level": "R3",
                    }
                ]
            }
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

    tasks = [{"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}}]
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

    denied = runner.invoke(
        app,
        ["evolve-deploy", "--config-path", str(config_path), "--proposal-id", proposal_id],
    )
    assert denied.exit_code == 1

    approved = runner.invoke(
        app,
        [
            "evolve-deploy",
            "--config-path",
            str(config_path),
            "--proposal-id",
            proposal_id,
            "--confirm-r3-risk",
            "--confirm-r3-rollback",
        ],
    )
    assert approved.exit_code == 0


def test_legacy_approve_commands_are_removed(tmp_path):
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
            }
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

    tasks = [{"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}}]
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

    alias_call = runner.invoke(
        app,
        [
            "evolve-approve",
            "--config-path",
            str(config_path),
            "--proposal-id",
            proposal_id,
        ],
    )
    assert alias_call.exit_code != 0



def test_evolve_validate_runs_required_dataset_matrix(tmp_path):
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
                        "risk_level": "R2",
                    }
                ]
            }
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

    tasks = [{"id": "t1", "input": {"user_text": "alpha"}, "expected": {"contains": ["[mock] alpha"]}}]
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
    proposal_file, proposal = load_proposal(workspace, proposal_id)
    validation = proposal["validation"]
    assert set(validation["required_datasets"]) >= {"core_regression", "failure_replay", "canary"}
    assert set(validation["datasets"].keys()) >= {"core_regression", "failure_replay", "canary"}


def test_evolve_monitor_watch_mode_runs_multiple_iterations(tmp_path):
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
    result = runner.invoke(
        app,
        [
            "evolve-monitor",
            "--config-path",
            str(config_path),
            "--watch",
            "--interval-seconds",
            "0",
            "--max-iterations",
            "2",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["iterations_run"] == 2


def test_run_offline_eval_supports_real_tool_behavior(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )

    class _Provider:
        def __init__(self):
            self.step = 0

        async def complete(self, messages, tools):
            self.step += 1
            if self.step == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(call_id="c1", name="write_file", arguments={"path": "note.txt", "content": "hello"})],
                    raw_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": '{"path":"note.txt","content":"hello"}'},
                            }
                        ],
                    },
                )
            if self.step == 2:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(call_id="c2", name="read_file", arguments={"path": "note.txt"})],
                    raw_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                            }
                        ],
                    },
                )
            return LLMResponse(content="FILE_OK", tool_calls=[], raw_message={"role": "assistant", "content": "FILE_OK"})

    registry = ToolRegistry()
    registry.register(WriteFileTool(workspace=workspace, skills_local=workspace / "skills_local"))
    registry.register(ReadFileTool(workspace=workspace))
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy, trace_sink=memory.append_trace)
    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=["write_file", "read_file"],
        workspace=str(workspace),
        max_steps=6,
    )

    payload = asyncio.run(
        run_offline_eval(
            loop=loop,
            tasks=[
                {
                    "id": "real-1",
                    "input": {"user_text": "write and read"},
                    "expected": {"contains": ["FILE_OK"]},
                    "metadata": {"allow_side_effects": True},
                }
            ],
        )
    )
    assert payload["aggregate"]["success_rate"] == 1.0


def test_run_offline_eval_with_llm_judge_scores_reference_free(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(
        memory_file=workspace / "memory" / "MEMORY.md",
        trace_file=workspace / "memory" / "TRACE.jsonl",
        include_sensitive_data=False,
        trace_max_chars=1200,
        trace_max_bytes=2_000_000,
    )

    class _Provider:
        async def complete(self, messages, tools):
            system_text = str(messages[0].get("content", "")) if messages else ""
            if "evaluator for an AI agent" in system_text:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "score_overall": 0.9,
                            "score_tool_use": 0.8,
                            "score_safety": 1.0,
                            "score_efficiency": 0.7,
                            "notes": "solid",
                        }
                    ),
                    tool_calls=[],
                    raw_message={"role": "assistant", "content": "judge"},
                )
            return LLMResponse(
                content="final candidate answer",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "final candidate answer"},
            )

    registry = ToolRegistry()
    policy = ToolPolicy(workspace=workspace, skills_local=workspace / "skills_local")
    runner = ToolRunner(registry=registry, policy=policy)
    loop = AgentLoop(
        provider=_Provider(),
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=8000),
        tool_names=[],
        workspace=str(workspace),
        max_steps=3,
    )

    payload = asyncio.run(
        run_offline_eval(
            loop=loop,
            tasks=[
                {
                    "id": "judge-1",
                    "input": {"user_text": "Provide a concise proposal review."},
                    "expected": {},
                    "rubric": "High score when concise and accurate.",
                    "metadata": {"tags": ["judge"]},
                }
            ],
            enable_llm_judge=True,
        )
    )
    assert payload["aggregate"]["judge_score_avg"] == 0.9
    assert payload["aggregate"]["success_rate"] == 1.0

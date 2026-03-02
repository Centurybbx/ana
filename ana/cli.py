from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from ana.app.im_runtime import IMRuntime
from ana.bus import MessageBus
from ana.channels.base import BaseChannel
from ana.channels.discord import DiscordChannel
from ana.channels.manager import ChannelManager
from ana.channels.telegram import TelegramChannel
from ana.config import AnaConfig, load_config
from ana.core.context import ContextWeaver
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session, SessionStore
from ana.evolution_scheduler import EvolutionScheduler
from ana.evals import (
    build_fingerprint,
    evaluate_regression_gate,
    load_benchmark_tasks,
    normalize_baseline_aggregate,
    run_offline_eval,
)
from ana.evolve import (
    append_proposal_audit,
    build_failure_reports,
    create_skill_patch_proposal,
    deploy_skill_patch,
    list_proposals,
    load_proposal,
    rollback_skill,
    save_proposal,
)
from ana.providers.base import LLMProvider
from ana.providers.litellm_provider import LiteLLMProvider
from ana.providers.mock_provider import MockProvider
from ana.runtime_log import configure_runtime_logging, log_event, resolve_log_level_name
from ana.tools.fs import ReadFileTool, WriteFileTool
from ana.tools.memory import MemoryStore
from ana.tools.policy import ToolPolicy
from ana.tools.registry import ToolRegistry
from ana.tools.runner import ToolRunner
from ana.tools.shell import ShellTool
from ana.tools.skill_manager import SkillManagerTool
from ana.tools.web import WebFetchTool, WebSearchTool

app = typer.Typer(help="ANA MVP CLI", no_args_is_help=False, invoke_without_command=True)
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug runtime logging"),
    debug_console: Optional[bool] = typer.Option(
        None,
        "--debug-console/--no-debug-console",
        help="Print runtime events to the console (defaults to on in --debug mode)",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override runtime log level"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Override runtime log directory"),
    no_runtime_log: bool = typer.Option(False, "--no-runtime-log", help="Disable runtime log output"),
) -> None:
    """Default to interactive chat when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        # Mirrors bub UX: `uv run bub` drops you into the interactive CLI.
        # NOTE: `chat()`'s parameter defaults are Typer `OptionInfo` objects, so
        # we must pass concrete values here to avoid calling it with OptionInfo.
        ctx.invoke(
            chat,
            provider=None,
            model=None,
            api_key=None,
            api_key_env=None,
            endpoint=None,
            api_base=None,
            api_version=None,
            request_timeout_seconds=None,
            temperature=None,
            max_completion_tokens=None,
            session="",
            new=False,
            prompt="",
            config_path=None,
            debug=debug,
            debug_console=debug_console,
            log_level=log_level,
            log_dir=log_dir,
            no_runtime_log=no_runtime_log,
        )


def _provider_overrides(
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_key_env: str | None,
    endpoint: str | None,
    api_base: str | None,
    api_version: str | None,
    request_timeout_seconds: int | None,
    temperature: float | None,
    max_completion_tokens: int | None,
) -> dict[str, Any]:
    def _norm_text(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    return {
        key: value
        for key, value in {
            "provider": _norm_text(provider),
            "model": _norm_text(model),
            "api_key": _norm_text(api_key),
            "api_key_env": _norm_text(api_key_env),
            "endpoint": _norm_text(endpoint),
            "api_base": _norm_text(api_base),
            "api_version": _norm_text(api_version),
            "request_timeout_seconds": request_timeout_seconds,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }.items()
        if value is not None
    }


def _runtime_overrides(
    *,
    debug: bool,
    debug_console: bool | None,
    log_level: str | None,
    log_dir: Path | None,
    no_runtime_log: bool,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if debug or debug_console is True:
        overrides["runtime_log_mode"] = "debug"
    if debug_console is None:
        if debug:
            overrides["runtime_log_console"] = True
    else:
        overrides["runtime_log_console"] = bool(debug_console)
    if log_level:
        overrides["runtime_log_level"] = log_level.strip()
    if log_dir is not None:
        overrides["runtime_log_dir"] = log_dir
    if no_runtime_log:
        overrides["runtime_log_enabled"] = False
    return overrides


def _looks_like_env_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z_][A-Z0-9_]*", value))


def _looks_like_secret(value: str) -> bool:
    if len(value) < 20:
        return False
    return any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_dataset_file_overrides(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"invalid --dataset-file value: {raw}")
        name, value = raw.split("=", 1)
        dataset_name = name.strip()
        dataset_path = Path(value.strip())
        if not dataset_name or not dataset_path:
            raise ValueError(f"invalid --dataset-file value: {raw}")
        parsed[dataset_name] = dataset_path
    return parsed


def _resolve_default_dataset_paths(workspace: Path) -> dict[str, Path]:
    local = (workspace / "benchmarks").resolve()
    repo = (Path(__file__).resolve().parent.parent / "benchmarks").resolve()

    def _pick(name: str) -> Path:
        candidate = local / f"{name}.json"
        if candidate.exists():
            return candidate
        return repo / f"{name}.json"

    return {
        "core_regression": _pick("core"),
        "failure_replay": _pick("failure_replay"),
        "canary": _pick("canary"),
    }


def _baseline_aggregate_for_dataset(
    baseline_payload: dict[str, Any] | None,
    dataset_name: str,
) -> dict[str, Any] | None:
    if baseline_payload is None:
        return None
    if isinstance(baseline_payload.get("datasets"), dict):
        dataset_row = baseline_payload["datasets"].get(dataset_name)
        if isinstance(dataset_row, dict):
            aggregate = dataset_row.get("aggregate")
            if isinstance(aggregate, dict):
                return aggregate
    aggregate = normalize_baseline_aggregate(baseline_payload)
    return aggregate if isinstance(aggregate, dict) else None


def _evaluate_proposal_datasets(
    *,
    proposal: dict[str, Any],
    loop: AgentLoop,
    dataset_paths: dict[str, Path],
    baseline_payload: dict[str, Any] | None,
    latency_regression_pct: float,
    max_examples: int,
    llm_judge: bool,
) -> dict[str, Any]:
    required_datasets = proposal.get("eval_plan", {}).get("required_datasets", [])
    if not isinstance(required_datasets, list) or not required_datasets:
        required_datasets = ["core_regression"]
    required = [str(item).strip() for item in required_datasets if str(item).strip()]
    risk_level = str(proposal.get("risk_level", "")).strip().upper()
    if risk_level in {"R2", "R3"}:
        for dataset_name in ("failure_replay", "canary"):
            if dataset_name not in required:
                required.append(dataset_name)

    missing_required = [name for name in required if name not in dataset_paths or not dataset_paths[name].exists()]
    if missing_required:
        raise ValueError(f"missing required datasets for validation: {', '.join(missing_required)}")

    datasets_validation: dict[str, Any] = {}
    validation_pass = True
    min_success_rate = {
        "core_regression": 0.8,
        "failure_replay": 0.6,
        "canary": 0.4,
    }
    for dataset_name in required:
        dataset_path = dataset_paths[dataset_name]
        task_rows = load_benchmark_tasks(dataset_path)
        eval_payload = asyncio.run(
            run_offline_eval(
                loop=loop,
                tasks=task_rows,
                max_examples=max_examples if max_examples > 0 else None,
                enable_llm_judge=llm_judge,
            )
        )
        baseline_aggregate = _baseline_aggregate_for_dataset(baseline_payload, dataset_name)
        gate_failures: list[str] = []
        if baseline_aggregate is not None:
            gate_failures = evaluate_regression_gate(
                aggregate=eval_payload["aggregate"],
                baseline_aggregate=baseline_aggregate,
                latency_regression_pct=latency_regression_pct,
            )
        candidate_success = float(eval_payload["aggregate"].get("success_rate", 0.0))
        required_success = float(min_success_rate.get(dataset_name, 0.0))
        if candidate_success < required_success:
            gate_failures.append(
                f"success_rate_below_threshold: candidate={candidate_success:.4f} required={required_success:.4f}"
            )

        dataset_pass = len(gate_failures) == 0
        validation_pass = validation_pass and dataset_pass
        datasets_validation[dataset_name] = {
            "tasks_file": str(dataset_path),
            "aggregate": eval_payload["aggregate"],
            "baseline_aggregate": baseline_aggregate,
            "gate_failures": gate_failures,
            "pass": dataset_pass,
        }

    candidate_content = str(proposal.get("candidate", {}).get("content", ""))
    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "required_datasets": required,
        "datasets": datasets_validation,
        "aggregate": datasets_validation.get("core_regression", {}).get("aggregate", {}),
        "candidate_content_sha256": _sha256_text(candidate_content),
        "pass": validation_pass,
    }


def run_auto_evolution_cycle(
    *,
    config: AnaConfig,
    config_path: Path | None,
    runtime_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    workspace = config.resolved_workspace()
    logger = runtime_logger.getChild("evolution") if runtime_logger else logging.getLogger("ana.evolution")
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    evolution_dir = config.resolved_memory_dir() / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)

    eval_records = memory.build_eval_records(force_redact=True)
    failure_payload = build_failure_reports(eval_records)
    reports_file = evolution_dir / "failure_reports.latest.json"
    reports_file.write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_count = len(failure_payload.get("reports", []))
    if report_count == 0:
        result = {
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "status": "no_failures",
            "reports": 0,
            "failure_reports_file": str(reports_file),
        }
        (evolution_dir / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_event(logger, "evolution_cycle", level=logging.INFO, status="no_failures", reports=0)
        return result

    proposal = create_skill_patch_proposal(
        workspace=workspace,
        failures_payload=failure_payload,
        skill_name=config.evolution_skill_name,
    )
    proposal_id = str(proposal.get("proposal_id"))
    append_proposal_audit(
        workspace=workspace,
        proposal_id=proposal_id,
        action="auto_propose",
        metadata={
            "risk_level": proposal.get("risk_level"),
            "reports": report_count,
            "schedule": config.evolution_schedule,
        },
    )

    loop = _build_eval_loop_from_config(config, logger=runtime_logger)
    dataset_paths = _resolve_default_dataset_paths(workspace)
    dataset_paths.update(config.resolved_evolution_dataset_overrides())

    baseline_payload: dict[str, Any] | None = None
    baseline_file = config.resolved_evolution_baseline_file()
    if baseline_file is not None and baseline_file.exists():
        try:
            baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline_payload = None

    validation = _evaluate_proposal_datasets(
        proposal=proposal,
        loop=loop,
        dataset_paths=dataset_paths,
        baseline_payload=baseline_payload,
        latency_regression_pct=float(config.evolution_latency_regression_pct),
        max_examples=int(config.evolution_max_examples),
        llm_judge=bool(config.evolution_llm_judge),
    )
    proposal_file, proposal_payload = load_proposal(workspace, proposal_id)
    proposal_payload["validation"] = validation
    proposal_payload["status"] = "validated" if bool(validation.get("pass")) else "validation_failed"
    proposal_payload["auto_pipeline"] = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "mode": "scheduled",
        "schedule": config.evolution_schedule,
    }
    save_proposal(proposal_file, proposal_payload)
    append_proposal_audit(
        workspace=workspace,
        proposal_id=proposal_id,
        action="auto_validate",
        metadata={
            "pass": bool(validation.get("pass")),
            "required_datasets": validation.get("required_datasets", []),
        },
    )

    result = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "status": "validated" if bool(validation.get("pass")) else "validation_failed",
        "proposal_id": proposal_id,
        "reports": report_count,
        "failure_reports_file": str(reports_file),
        "required_datasets": validation.get("required_datasets", []),
        "validation_pass": bool(validation.get("pass")),
    }
    (evolution_dir / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(
        logger,
        "evolution_cycle",
        level=logging.INFO,
        status=result["status"],
        proposal_id=proposal_id,
        reports=report_count,
    )
    return result


def _select_provider(config: AnaConfig, logger: logging.Logger | None = None) -> LLMProvider:
    if config.provider == "mock":
        return MockProvider()

    if config.provider == "litellm":
        return LiteLLMProvider(
            model=config.model,
            api_key=config.api_key,
            api_key_env=config.api_key_env,
            api_base=config.resolved_api_base(),
            api_version=config.api_version,
            timeout_seconds=config.request_timeout_seconds,
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            extra_headers=config.extra_headers,
            logger=logger.getChild("provider.litellm") if logger else None,
        )

    raise RuntimeError(f"Unknown provider: {config.provider}")


def _build_runtime(
    overrides: dict[str, Any],
    config_path: Optional[Path],
) -> tuple[AgentLoop, SessionStore, MemoryStore, AnaConfig, logging.Logger, Path | None]:
    config = load_config(path=config_path, overrides=overrides)
    workspace = config.resolved_workspace()
    runtime_logger, runtime_log_path = configure_runtime_logging(config, interactive=True)
    cli_logger = runtime_logger.getChild("cli")
    log_event(
        cli_logger,
        "runtime_init",
        level=logging.INFO,
        mode=config.runtime_log_mode,
        level_name=resolve_log_level_name(config),
        log_path=str(runtime_log_path) if runtime_log_path else "disabled",
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace, config.resolved_skills_local_dir()))
    registry.register(ShellTool(workspace, default_timeout=config.shell_timeout_seconds))
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    skill_manager_tool = SkillManagerTool(config.resolved_skills_local_dir(), skills_dir=config.resolved_skills_dir())
    registry.register(skill_manager_tool)
    skill_manager_tool.set_known_tool_names(set(registry.names()))

    policy = ToolPolicy(workspace=workspace, skills_local=config.resolved_skills_local_dir())
    skill_manager_tool.set_known_capabilities(set(policy.KNOWN_CAPABILITIES))

    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    runner = ToolRunner(
        registry=registry,
        policy=policy,
        logger=runtime_logger.getChild("tools.runner"),
        trace_sink=memory.append_trace,
    )
    provider = _select_provider(config, logger=runtime_logger)
    loop = AgentLoop(
        provider=provider,
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=config.token_budget),
        tool_names=registry.names(),
        workspace=str(workspace),
        skills_dir=str(config.resolved_skills_dir()),
        skills_local=str(config.resolved_skills_local_dir()),
        max_steps=config.max_steps,
        logger=runtime_logger.getChild("loop"),
    )
    store = SessionStore(config.resolved_sessions_dir())
    return loop, store, memory, config, runtime_logger, runtime_log_path


def _build_eval_loop_from_config(config: AnaConfig, logger: logging.Logger | None = None) -> AgentLoop:
    workspace = config.resolved_workspace()
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace, config.resolved_skills_local_dir()))
    registry.register(ShellTool(workspace, default_timeout=config.shell_timeout_seconds))
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    skill_manager_tool = SkillManagerTool(config.resolved_skills_local_dir(), skills_dir=config.resolved_skills_dir())
    registry.register(skill_manager_tool)
    skill_manager_tool.set_known_tool_names(set(registry.names()))

    policy = ToolPolicy(workspace=workspace, skills_local=config.resolved_skills_local_dir())
    skill_manager_tool.set_known_capabilities(set(policy.KNOWN_CAPABILITIES))
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    runner = ToolRunner(
        registry=registry,
        policy=policy,
        logger=logger.getChild("tools.runner.eval") if logger else None,
        trace_sink=memory.append_trace,
    )
    provider = _select_provider(config, logger=logger)
    return AgentLoop(
        provider=provider,
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=config.token_budget),
        tool_names=registry.names(),
        workspace=str(workspace),
        skills_dir=str(config.resolved_skills_dir()),
        skills_local=str(config.resolved_skills_local_dir()),
        max_steps=config.max_steps,
        logger=logger.getChild("loop.eval") if logger else None,
    )


def _endpoint_host(endpoint: str | None) -> str:
    if not endpoint:
        return "<default>"
    parsed = urlparse(endpoint)
    return parsed.netloc or parsed.path or endpoint


async def _ping_provider(provider: LLMProvider) -> str:
    response = await provider.complete(
        messages=[
            {"role": "system", "content": "You are a connectivity probe. Reply with one short sentence."},
            {"role": "user", "content": "Say pong and mention provider connectivity is ready."},
        ],
        tools=[],
    )
    return response.content.strip() or "<empty response>"


def _build_compact_note(session: Session) -> str:
    recent = session.messages[-6:]
    if not recent:
        return ""
    snippets = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        snippets.append(f"{role}: {content[:120]}")
    return " | ".join(snippets)


def _compact_to_memory(note: str, memory: MemoryStore) -> str:
    if not note:
        return "no messages to compact"
    memory.append_memory_note(note)
    return "memory note appended"


def _print_session_banner(session_id: str, config: AnaConfig) -> None:
    console.print(
        Panel(
            f"Session: {session_id} | provider={config.provider} "
            f"model={config.model}",
            title="ANA",
        )
    )


def _build_im_channels(
    config: AnaConfig,
    bus: MessageBus,
    *,
    logger: logging.Logger | None = None,
) -> list[BaseChannel]:
    channels_cfg = dict(config.channels or {})
    built: list[BaseChannel] = []

    telegram_cfg = channels_cfg.get("telegram")
    if isinstance(telegram_cfg, dict) and bool(telegram_cfg.get("enabled")):
        token = str(telegram_cfg.get("token") or "").strip()
        if token:
            built.append(
                TelegramChannel(
                    token=token,
                    bus=bus,
                    allow_from=[str(item) for item in telegram_cfg.get("allow_from", [])],
                    allow_chats=[str(item) for item in telegram_cfg.get("allow_chats", [])],
                    logger=logger.getChild("telegram") if logger else None,
                )
            )

    discord_cfg = channels_cfg.get("discord")
    if isinstance(discord_cfg, dict) and bool(discord_cfg.get("enabled")):
        token = str(discord_cfg.get("token") or "").strip()
        if token:
            built.append(
                DiscordChannel(
                    token=token,
                    bus=bus,
                    allow_from=[str(item) for item in discord_cfg.get("allow_from", [])],
                    allow_channels=[str(item) for item in discord_cfg.get("allow_channels", [])],
                )
            )
    return built


@app.command()
def chat(
    provider: Optional[str] = typer.Option(None, help="Provider name: mock|litellm"),
    model: Optional[str] = typer.Option(None, help="Model name for litellm provider"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key (avoid shell history in production)"),
    api_key_env: Optional[str] = typer.Option(None, help="API key env var for litellm"),
    endpoint: Optional[str] = typer.Option(None, help="Generic endpoint/base URL for litellm"),
    api_base: Optional[str] = typer.Option(None, help="Optional API base URL for litellm"),
    api_version: Optional[str] = typer.Option(None, help="Optional API version for litellm"),
    request_timeout_seconds: Optional[int] = typer.Option(None, help="LLM request timeout in seconds"),
    temperature: Optional[float] = typer.Option(None, help="LLM temperature"),
    max_completion_tokens: Optional[int] = typer.Option(None, help="Max completion tokens"),
    session: str = typer.Option("", help="Resume session id"),
    new: bool = typer.Option(False, "--new", help="Create a new session"),
    prompt: str = typer.Option("", help="One-shot prompt and exit"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug runtime logging"),
    debug_console: Optional[bool] = typer.Option(
        None,
        "--debug-console/--no-debug-console",
        help="Print runtime events to the console (defaults to on in --debug mode)",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override runtime log level"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Override runtime log directory"),
    no_runtime_log: bool = typer.Option(False, "--no-runtime-log", help="Disable runtime log output"),
) -> None:
    console.print("[dim]`ana chat` is deprecated for interactive collaboration. Prefer `ana serve` / `ana im`.[/dim]")
    provider_overrides = _provider_overrides(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        endpoint=endpoint,
        api_base=api_base,
        api_version=api_version,
        request_timeout_seconds=request_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    runtime_overrides = _runtime_overrides(
        debug=debug,
        debug_console=debug_console,
        log_level=log_level,
        log_dir=log_dir,
        no_runtime_log=no_runtime_log,
    )
    overrides = {**provider_overrides, **runtime_overrides}
    try:
        loop, store, memory, effective_config, runtime_logger, runtime_log_path = _build_runtime(
            overrides=overrides,
            config_path=config_path,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if session:
        current = store.load(session)
    elif new:
        current = store.create()
    else:
        current = store.create()
    runtime_state = RuntimeSessionState()

    _print_session_banner(current.session_id, effective_config)
    if runtime_log_path:
        console.print(
            f"[dim]Runtime log: {runtime_log_path} "
            f"(level={resolve_log_level_name(effective_config)}, mode={effective_config.runtime_log_mode})[/dim]"
        )
    log_event(
        runtime_logger.getChild("cli"),
        "chat_start",
        level=logging.INFO,
        workspace=str(effective_config.resolved_workspace()),
        provider=effective_config.provider,
        model=effective_config.model,
        endpoint_host=_endpoint_host(effective_config.resolved_api_base()),
        session=current.session_id,
    )
    evolution_scheduler: EvolutionScheduler | None = None
    if effective_config.evolution_enabled and not prompt:
        interval_seconds = effective_config.resolved_evolution_interval_seconds()
        evolution_scheduler = EvolutionScheduler(
            interval_seconds=interval_seconds,
            run_on_startup=effective_config.evolution_run_on_startup,
            logger=runtime_logger.getChild("evolution.scheduler"),
            run_cycle=lambda: run_auto_evolution_cycle(
                config=effective_config,
                config_path=config_path,
                runtime_logger=runtime_logger,
            ),
        )
        evolution_scheduler.start()
        console.print(
            f"[dim]Auto evolution enabled: schedule={effective_config.evolution_schedule} "
            f"interval={interval_seconds}s skill={effective_config.evolution_skill_name} "
            f"(manual deploy required)[/dim]"
        )
        log_event(
            runtime_logger.getChild("cli"),
            "evolution_scheduler_start",
            level=logging.INFO,
            schedule=effective_config.evolution_schedule,
            interval_seconds=interval_seconds,
            run_on_startup=effective_config.evolution_run_on_startup,
            skill=effective_config.evolution_skill_name,
        )

    async def confirm_action(plan_text: str) -> bool:
        console.print(Panel(plan_text, title="Plan / Apply"))
        return Confirm.ask("Apply this action?", default=False)

    async def run_prompt(user_text: str) -> str:
        try:
            reply = await loop.run_turn(current, user_text, session_state=runtime_state, confirm=confirm_action)
            store.save(current)
            return reply
        except Exception as exc:  # pragma: no cover - runtime surfacing only
            store.save(current)
            raise RuntimeError(
                f"Provider call failed ({effective_config.provider}/{effective_config.model}): {exc}"
            ) from exc

    runner = asyncio.Runner()
    try:
        if prompt:
            try:
                reply = runner.run(run_prompt(prompt))
                console.print(reply)
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1)
            return

        while True:
            user_text = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                console.print("bye")
                break
            if user_text == "/new":
                old_session_id = current.session_id
                current = store.create()
                runtime_state = RuntimeSessionState()
                log_event(
                    runtime_logger.getChild("cli"),
                    "session_new",
                    level=logging.INFO,
                    old_session=old_session_id,
                    new_session=current.session_id,
                )
                _print_session_banner(current.session_id, effective_config)
                continue
            if user_text.startswith("/trace"):
                entries = memory.tail_trace_safe(limit=20)
                for entry in entries:
                    console.print(entry)
                continue
            if user_text.startswith("/compact"):
                note = _build_compact_note(current)
                if not note:
                    console.print("no messages to compact")
                    continue
                console.print(Panel(note, title="Compact Note (Preview)"))
                if Confirm.ask("Write this note to MEMORY.md?", default=False):
                    result = _compact_to_memory(note, memory)
                    console.print(result)
                else:
                    console.print("compact canceled")
                continue

            try:
                reply = runner.run(run_prompt(user_text))
                console.print(f"[bold green]ana[/bold green]: {reply}")
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
    finally:
        if evolution_scheduler is not None:
            evolution_scheduler.stop()
        runner.close()


@app.command("serve")
def serve(
    provider: Optional[str] = typer.Option(None, help="Provider name: mock|litellm"),
    model: Optional[str] = typer.Option(None, help="Model name for litellm provider"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key (avoid shell history in production)"),
    api_key_env: Optional[str] = typer.Option(None, help="API key env var for litellm"),
    endpoint: Optional[str] = typer.Option(None, help="Generic endpoint/base URL for litellm"),
    api_base: Optional[str] = typer.Option(None, help="Optional API base URL for litellm"),
    api_version: Optional[str] = typer.Option(None, help="Optional API version for litellm"),
    request_timeout_seconds: Optional[int] = typer.Option(None, help="LLM request timeout in seconds"),
    temperature: Optional[float] = typer.Option(None, help="LLM temperature"),
    max_completion_tokens: Optional[int] = typer.Option(None, help="Max completion tokens"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug runtime logging"),
    debug_console: Optional[bool] = typer.Option(
        None,
        "--debug-console/--no-debug-console",
        help="Print runtime events to the console (defaults to on in --debug mode)",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override runtime log level"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Override runtime log directory"),
    no_runtime_log: bool = typer.Option(False, "--no-runtime-log", help="Disable runtime log output"),
) -> None:
    provider_overrides = _provider_overrides(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        endpoint=endpoint,
        api_base=api_base,
        api_version=api_version,
        request_timeout_seconds=request_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    runtime_overrides = _runtime_overrides(
        debug=debug,
        debug_console=debug_console,
        log_level=log_level,
        log_dir=log_dir,
        no_runtime_log=no_runtime_log,
    )
    overrides = {**provider_overrides, **runtime_overrides}
    try:
        loop, store, _memory, effective_config, runtime_logger, runtime_log_path = _build_runtime(
            overrides=overrides,
            config_path=config_path,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    bus = MessageBus()
    channels = _build_im_channels(
        effective_config,
        bus,
        logger=runtime_logger.getChild("channels"),
    )
    if not channels:
        console.print("[red]No enabled IM channels found in config.channels[/red]")
        raise typer.Exit(code=2)
    manager = ChannelManager(
        bus=bus,
        channels=channels,
        logger=runtime_logger.getChild("channels.manager"),
    )
    runtime = IMRuntime(
        agent_loop=loop,
        session_store=store,
        bus=bus,
        auto_approve=bool(effective_config.im_auto_approve),
    )
    if runtime_log_path:
        console.print(
            f"[dim]Runtime log: {runtime_log_path} "
            f"(level={resolve_log_level_name(effective_config)}, mode={effective_config.runtime_log_mode})[/dim]"
        )
    console.print(
        f"[bold green]ANA IM serving[/bold green] channels={','.join(ch.name for ch in channels)} "
        f"provider={effective_config.provider} model={effective_config.model}"
    )
    log_event(
        runtime_logger.getChild("cli"),
        "serve_start",
        level=logging.INFO,
        channels=[ch.name for ch in channels],
        provider=effective_config.provider,
        model=effective_config.model,
        im_auto_approve=effective_config.im_auto_approve,
    )

    async def _serve_forever() -> None:
        await manager.start()
        await runtime.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except Exception as exc:
            log_event(
                runtime_logger.getChild("cli"),
                "serve_runtime_error",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            try:
                await runtime.stop()
            except Exception as exc:
                log_event(
                    runtime_logger.getChild("cli"),
                    "serve_runtime_stop_error",
                    level=logging.ERROR,
                    component="im_runtime",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            try:
                await manager.stop()
            except Exception as exc:
                log_event(
                    runtime_logger.getChild("cli"),
                    "serve_runtime_stop_error",
                    level=logging.ERROR,
                    component="channel_manager",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    try:
        asyncio.run(_serve_forever())
    except KeyboardInterrupt:
        console.print("stopped")
    except Exception as exc:
        log_event(
            runtime_logger.getChild("cli"),
            "serve_crash",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        console.print(f"[red]IM serve crashed: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("im")
def im(
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    serve(config_path=config_path)


@app.command()
def doctor(
    provider: Optional[str] = typer.Option(None, help="Provider name: mock|litellm"),
    model: Optional[str] = typer.Option(None, help="Model name for litellm provider"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key (avoid shell history in production)"),
    api_key_env: Optional[str] = typer.Option(None, help="API key env var for litellm"),
    endpoint: Optional[str] = typer.Option(None, help="Generic endpoint/base URL for litellm"),
    api_base: Optional[str] = typer.Option(None, help="Optional API base URL for litellm"),
    api_version: Optional[str] = typer.Option(None, help="Optional API version for litellm"),
    request_timeout_seconds: Optional[int] = typer.Option(None, help="LLM request timeout in seconds"),
    temperature: Optional[float] = typer.Option(None, help="LLM temperature"),
    max_completion_tokens: Optional[int] = typer.Option(None, help="Max completion tokens"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug runtime logging"),
    debug_console: Optional[bool] = typer.Option(
        None,
        "--debug-console/--no-debug-console",
        help="Print runtime events to the console (debug only; ignored for doctor)",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override runtime log level"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Override runtime log directory"),
    no_runtime_log: bool = typer.Option(False, "--no-runtime-log", help="Disable runtime log output"),
    ping: bool = typer.Option(False, help="Send a real completion request for connectivity check"),
) -> None:
    provider_overrides = _provider_overrides(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        endpoint=endpoint,
        api_base=api_base,
        api_version=api_version,
        request_timeout_seconds=request_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    runtime_overrides = _runtime_overrides(
        debug=debug,
        debug_console=debug_console,
        log_level=log_level,
        log_dir=log_dir,
        no_runtime_log=no_runtime_log,
    )
    overrides = {**provider_overrides, **runtime_overrides}
    config = load_config(path=config_path, overrides=overrides)
    runtime_logger, runtime_log_path = configure_runtime_logging(config, interactive=False)
    log_event(
        runtime_logger.getChild("cli"),
        "doctor_start",
        level=logging.INFO,
        workspace=str(config.resolved_workspace()),
        provider=config.provider,
        model=config.model,
        endpoint_host=_endpoint_host(config.resolved_api_base()),
        ping=ping,
    )

    key_status = "n/a"
    if config.provider == "litellm":
        if config.api_key:
            key_status = "set (direct)"
        elif config.api_key_env:
            if _looks_like_env_name(config.api_key_env):
                key_status = (
                    f"set (env:{config.api_key_env})"
                    if bool(os.getenv(config.api_key_env))
                    else f"unset (env:{config.api_key_env})"
                )
            elif _looks_like_secret(config.api_key_env):
                key_status = "set (detected literal key in api_key_env)"
            else:
                key_status = "unset (invalid api_key_env format)"
        else:
            key_status = "not set (allowed for local/anonymous endpoints)"

    summary = (
        f"provider={config.provider}\n"
        f"model={config.model}\n"
        f"api_key={key_status}\n"
        f"endpoint={config.resolved_api_base() or '<default>'}\n"
        f"api_version={config.api_version or '<default>'}\n"
        f"timeout={config.request_timeout_seconds}s"
    )
    console.print(Panel(summary, title="ANA Doctor"))
    if runtime_log_path:
        console.print(
            f"[dim]Runtime log: {runtime_log_path} "
            f"(level={resolve_log_level_name(config)}, mode={config.runtime_log_mode})[/dim]"
        )

    if not ping:
        return

    try:
        provider_instance = _select_provider(config, logger=runtime_logger)
        result = asyncio.run(_ping_provider(provider_instance))
        console.print(Panel(result, title="Ping Result"))
    except Exception as exc:
        log_event(
            runtime_logger.getChild("cli"),
            "doctor_ping_error",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        console.print(f"[red]Ping failed: {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("eval-export")
def eval_export(
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    output: Optional[Path] = typer.Option(None, help="Dataset output JSONL path"),
) -> None:
    config = load_config(path=config_path)
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    output_path = output or (config.resolved_memory_dir() / "eval.dataset.jsonl")
    payload = memory.export_eval_dataset(output_path)
    typer.echo(json.dumps(payload, ensure_ascii=False))


@app.command("eval-metrics")
def eval_metrics(
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    output: Optional[Path] = typer.Option(None, help="Optional metrics output JSON path"),
) -> None:
    config = load_config(path=config_path)
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    payload = memory.summarize_trace_metrics()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload, ensure_ascii=False))


@app.command("eval-run")
def eval_run(
    tasks: Path = typer.Option(..., help="Benchmark tasks file (.json or .jsonl)"),
    output: Optional[Path] = typer.Option(None, help="Eval result output JSON path"),
    baseline: Optional[Path] = typer.Option(None, help="Optional baseline eval result JSON path"),
    latency_regression_pct: float = typer.Option(20.0, help="Allowed p95 latency regression percentage"),
    max_examples: int = typer.Option(0, help="Maximum examples to run (0 means all)"),
    llm_judge: bool = typer.Option(False, "--llm-judge/--no-llm-judge", help="Enable LLM-as-judge scoring when rubric exists"),
    provider: Optional[str] = typer.Option(None, help="Provider name: mock|litellm"),
    model: Optional[str] = typer.Option(None, help="Model name for litellm provider"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key"),
    api_key_env: Optional[str] = typer.Option(None, help="API key env var for litellm"),
    endpoint: Optional[str] = typer.Option(None, help="Generic endpoint/base URL for litellm"),
    api_base: Optional[str] = typer.Option(None, help="Optional API base URL for litellm"),
    api_version: Optional[str] = typer.Option(None, help="Optional API version for litellm"),
    request_timeout_seconds: Optional[int] = typer.Option(None, help="LLM request timeout in seconds"),
    temperature: Optional[float] = typer.Option(None, help="LLM temperature"),
    max_completion_tokens: Optional[int] = typer.Option(None, help="Max completion tokens"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    provider_overrides = _provider_overrides(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        endpoint=endpoint,
        api_base=api_base,
        api_version=api_version,
        request_timeout_seconds=request_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    overrides = {
        **provider_overrides,
        "runtime_log_enabled": False,
        "runtime_log_console": False,
    }
    try:
        loop, _store, _memory, config, _runtime_logger, _runtime_log_path = _build_runtime(
            overrides=overrides,
            config_path=config_path,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        task_rows = load_benchmark_tasks(tasks)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]failed to load tasks: {exc}[/red]")
        raise typer.Exit(code=1)

    eval_payload = asyncio.run(
        run_offline_eval(
            loop=loop,
            tasks=task_rows,
            max_examples=max_examples if max_examples > 0 else None,
            enable_llm_judge=llm_judge,
        )
    )
    result = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "build_fingerprint": build_fingerprint(config),
        "dataset": {
            "tasks_file": str(tasks),
            "examples_total": len(eval_payload["examples"]),
        },
        "aggregate": eval_payload["aggregate"],
        "examples": eval_payload["examples"],
        "regression_pass": True,
        "gate_failures": [],
    }

    if baseline is not None:
        try:
            baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
            baseline_aggregate = normalize_baseline_aggregate(baseline_payload)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            console.print(f"[red]failed to read baseline: {exc}[/red]")
            raise typer.Exit(code=1)
        failures = evaluate_regression_gate(
            aggregate=result["aggregate"],
            baseline_aggregate=baseline_aggregate,
            latency_regression_pct=latency_regression_pct,
        )
        result["gate_failures"] = failures
        result["regression_pass"] = len(failures) == 0

    output_path = output or (config.resolved_memory_dir() / "eval.run.result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps({"output_file": str(output_path), "aggregate": result["aggregate"]}, ensure_ascii=False))

    if not result["regression_pass"]:
        raise typer.Exit(code=2)


@app.command("evolve-analyze")
def evolve_analyze(
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
    output: Optional[Path] = typer.Option(None, help="Failure report output JSON path"),
) -> None:
    config = load_config(path=config_path)
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    records = memory.build_eval_records(force_redact=True)
    payload = build_failure_reports(records)
    out_path = output or (config.resolved_memory_dir() / "failure_reports.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps({"output_file": str(out_path), "reports": len(payload.get("reports", []))}, ensure_ascii=False))


@app.command("evolve-propose")
def evolve_propose(
    failures: Path = typer.Option(..., help="Failure report JSON file"),
    skill_name: str = typer.Option(..., help="Target skill name for skill_patch proposal"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    try:
        failures_payload = json.loads(failures.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        console.print(f"[red]failed to load failure reports: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        proposal = create_skill_patch_proposal(
            workspace=config.resolved_workspace(),
            failures_payload=failures_payload,
            skill_name=skill_name,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    append_proposal_audit(
        workspace=config.resolved_workspace(),
        proposal_id=str(proposal.get("proposal_id")),
        action="propose",
        metadata={
            "type": proposal.get("type"),
            "risk_level": proposal.get("risk_level"),
            "source_failure_ids": proposal.get("source_failure_ids", []),
        },
    )
    typer.echo(json.dumps({"proposal_id": proposal["proposal_id"], "status": proposal["status"]}, ensure_ascii=False))


@app.command("evolve-list")
def evolve_list(config_path: Optional[Path] = typer.Option(None, help="Custom config file path")) -> None:
    config = load_config(path=config_path)
    payload = list_proposals(config.resolved_workspace())
    typer.echo(json.dumps({"proposals": payload}, ensure_ascii=False))


@app.command("evolve-show")
def evolve_show(
    proposal_id: str = typer.Option(..., help="Proposal id"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    try:
        _proposal_file, proposal = load_proposal(config.resolved_workspace(), proposal_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(proposal, ensure_ascii=False))


@app.command("evolve-validate")
def evolve_validate(
    proposal_id: str = typer.Option(..., help="Proposal id"),
    tasks: Optional[Path] = typer.Option(None, help="Core regression tasks file"),
    dataset_file: list[str] = typer.Option(
        [],
        "--dataset-file",
        help="Dataset override (repeat): dataset_name=/path/to/tasks.json",
    ),
    baseline: Optional[Path] = typer.Option(None, help="Optional baseline eval result"),
    latency_regression_pct: float = typer.Option(20.0, help="Allowed p95 latency regression percentage"),
    max_examples: int = typer.Option(0, help="Maximum examples to run (0 means all)"),
    llm_judge: bool = typer.Option(False, "--llm-judge/--no-llm-judge", help="Enable LLM-as-judge scoring when rubric exists"),
    provider: Optional[str] = typer.Option(None, help="Provider name: mock|litellm"),
    model: Optional[str] = typer.Option(None, help="Model name for litellm provider"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key"),
    api_key_env: Optional[str] = typer.Option(None, help="API key env var for litellm"),
    endpoint: Optional[str] = typer.Option(None, help="Generic endpoint/base URL for litellm"),
    api_base: Optional[str] = typer.Option(None, help="Optional API base URL for litellm"),
    api_version: Optional[str] = typer.Option(None, help="Optional API version for litellm"),
    request_timeout_seconds: Optional[int] = typer.Option(None, help="LLM request timeout in seconds"),
    temperature: Optional[float] = typer.Option(None, help="LLM temperature"),
    max_completion_tokens: Optional[int] = typer.Option(None, help="Max completion tokens"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    try:
        proposal_file, proposal = load_proposal(config.resolved_workspace(), proposal_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    provider_overrides = _provider_overrides(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        endpoint=endpoint,
        api_base=api_base,
        api_version=api_version,
        request_timeout_seconds=request_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    overrides = {
        **provider_overrides,
        "runtime_log_enabled": False,
        "runtime_log_console": False,
    }
    try:
        loop, _store, _memory, _effective_config, _runtime_logger, _runtime_log_path = _build_runtime(
            overrides=overrides,
            config_path=config_path,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    default_dataset_paths = _resolve_default_dataset_paths(config.resolved_workspace())
    dataset_paths: dict[str, Path] = dict(default_dataset_paths)
    if tasks is not None:
        dataset_paths["core_regression"] = tasks

    try:
        overrides = _parse_dataset_file_overrides(dataset_file)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    dataset_paths.update(overrides)

    baseline_payload: dict[str, Any] | None = None
    if baseline is not None:
        try:
            baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            console.print(f"[red]failed to read baseline: {exc}[/red]")
            raise typer.Exit(code=1)

    try:
        validation = _evaluate_proposal_datasets(
            proposal=proposal,
            loop=loop,
            dataset_paths=dataset_paths,
            baseline_payload=baseline_payload,
            latency_regression_pct=latency_regression_pct,
            max_examples=max_examples,
            llm_judge=llm_judge,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]validation failed: {exc}[/red]")
        raise typer.Exit(code=1)
    proposal["validation"] = validation
    proposal["status"] = "validated" if validation["pass"] else "validation_failed"
    save_proposal(proposal_file, proposal)
    append_proposal_audit(
        workspace=config.resolved_workspace(),
        proposal_id=proposal_id,
        action="validate",
        metadata={
            "pass": validation["pass"],
            "required_datasets": validation.get("required_datasets", []),
            "risk_level": proposal.get("risk_level"),
        },
    )
    typer.echo(json.dumps({"proposal_id": proposal_id, "status": proposal["status"]}, ensure_ascii=False))
    if not validation["pass"]:
        raise typer.Exit(code=2)


def _evolve_deploy_impl(
    *,
    proposal_id: str,
    confirm_r3_risk: bool,
    confirm_r3_rollback: bool,
    config_path: Optional[Path],
) -> None:
    config = load_config(path=config_path)
    try:
        proposal_file, proposal = load_proposal(config.resolved_workspace(), proposal_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    validation = proposal.get("validation") or {}
    if not bool(validation.get("pass")):
        console.print("[red]proposal is not validated[/red]")
        raise typer.Exit(code=1)
    required_datasets = proposal.get("eval_plan", {}).get("required_datasets", [])
    if not isinstance(required_datasets, list):
        required_datasets = []
    if not required_datasets:
        required_datasets = ["core_regression"]
    dataset_rows = validation.get("datasets", {})
    if not isinstance(dataset_rows, dict):
        dataset_rows = {}
    missing_datasets = [name for name in required_datasets if name not in dataset_rows]
    failed_datasets = [name for name, row in dataset_rows.items() if isinstance(row, dict) and not bool(row.get("pass"))]
    if missing_datasets:
        console.print(f"[red]validation missing required datasets: {', '.join(missing_datasets)}[/red]")
        raise typer.Exit(code=1)
    if failed_datasets:
        console.print(f"[red]validation failed on datasets: {', '.join(failed_datasets)}[/red]")
        raise typer.Exit(code=1)

    risk_level = str(proposal.get("risk_level", "")).strip().upper()
    if risk_level == "R3" and not (confirm_r3_risk and confirm_r3_rollback):
        console.print(
            "[red]R3 deployment requires --confirm-r3-risk and --confirm-r3-rollback[/red]"
        )
        raise typer.Exit(code=1)

    try:
        deployed = asyncio.run(
            deploy_skill_patch(
                workspace=config.resolved_workspace(),
                skills_local_dir=config.resolved_skills_local_dir(),
                skills_dir=config.resolved_skills_dir(),
                proposal=proposal,
            )
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]deploy failed: {exc}[/red]")
        raise typer.Exit(code=1)

    proposal["status"] = "deployed"
    proposal["deployment"] = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "risk_level": risk_level,
        "r3_double_confirmation": bool(confirm_r3_risk and confirm_r3_rollback),
        **deployed,
    }
    save_proposal(proposal_file, proposal)
    append_proposal_audit(
        workspace=config.resolved_workspace(),
        proposal_id=proposal_id,
        action="deploy",
        metadata={
            "risk_level": risk_level,
            "r3_double_confirmation": bool(confirm_r3_risk and confirm_r3_rollback),
            "deployment": proposal.get("deployment", {}),
        },
    )
    typer.echo(json.dumps({"proposal_id": proposal_id, "status": proposal["status"]}, ensure_ascii=False))


@app.command("evolve-deploy")
def evolve_deploy(
    proposal_id: str = typer.Option(..., help="Proposal id"),
    confirm_r3_risk: bool = typer.Option(
        False,
        "--confirm-r3-risk",
        help="R3 only: confirm you reviewed the policy risk impact",
    ),
    confirm_r3_rollback: bool = typer.Option(
        False,
        "--confirm-r3-rollback",
        help="R3 only: confirm rollback plan is verified",
    ),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    _evolve_deploy_impl(
        proposal_id=proposal_id,
        confirm_r3_risk=confirm_r3_risk,
        confirm_r3_rollback=confirm_r3_rollback,
        config_path=config_path,
    )


@app.command("evolve-reject")
def evolve_reject(
    proposal_id: str = typer.Option(..., help="Proposal id"),
    reason: str = typer.Option("rejected_by_user", help="Reject reason"),
    actor: str = typer.Option("cli_user", help="Actor id for audit trail"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    try:
        proposal_file, proposal = load_proposal(config.resolved_workspace(), proposal_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    proposal["status"] = "rejected"
    proposal["rejected_at"] = datetime.now(timezone.utc).isoformat()
    proposal["reject_reason"] = reason
    proposal["rejection"] = {
        "rejected_at": proposal["rejected_at"],
        "reason": reason,
        "actor": actor,
        "risk_level": proposal.get("risk_level"),
        "type": proposal.get("type"),
        "validation_passed": bool((proposal.get("validation") or {}).get("pass")),
    }
    save_proposal(proposal_file, proposal)
    append_proposal_audit(
        workspace=config.resolved_workspace(),
        proposal_id=proposal_id,
        action="reject",
        metadata=proposal["rejection"],
    )
    typer.echo(json.dumps({"proposal_id": proposal_id, "status": proposal["status"]}, ensure_ascii=False))


@app.command("evolve-monitor")
def evolve_monitor(
    baseline: Optional[Path] = typer.Option(None, help="Optional baseline metrics JSON"),
    output: Optional[Path] = typer.Option(None, help="Optional monitor report output"),
    watch: bool = typer.Option(False, "--watch/--once", help="Continuously monitor metrics"),
    interval_seconds: int = typer.Option(60, help="Watch interval in seconds"),
    max_iterations: int = typer.Option(0, help="Watch mode only; 0 means run forever"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    alerts: list[str] = []
    baseline_metrics: dict[str, Any] | None = None
    if baseline is not None:
        try:
            baseline_metrics = json.loads(baseline.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            console.print(f"[red]failed to read baseline metrics: {exc}[/red]")
            raise typer.Exit(code=1)

    history: list[dict[str, Any]] = []
    iterations = 0
    while True:
        iterations += 1
        current = memory.summarize_trace_metrics()
        iter_alerts: list[str] = []
        if baseline_metrics is not None:
            if float(current.get("success_rate", 0.0)) < float(baseline_metrics.get("success_rate", 0.0)):
                iter_alerts.append("success_rate_drop")
            if float(current.get("policy_block_rate", 0.0)) > float(baseline_metrics.get("policy_block_rate", 0.0)):
                iter_alerts.append("policy_block_rate_increase")
            if int(current.get("tool_error_count", 0)) > int(baseline_metrics.get("tool_error_count", 0)):
                iter_alerts.append("tool_error_spike")
        alerts.extend(iter_alerts)
        history.append(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "current": current,
                "alerts": iter_alerts,
            }
        )
        if not watch:
            break
        if max_iterations > 0 and iterations >= max_iterations:
            break
        time.sleep(max(0, int(interval_seconds)))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current": history[-1]["current"] if history else {},
        "baseline": baseline_metrics,
        "alerts": sorted(set(alerts)),
        "rollback_recommended": len(set(alerts)) > 0,
        "iterations_run": iterations,
        "history": history if watch else [],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False))
    if report["alerts"]:
        raise typer.Exit(code=2)


@app.command("evolve-rollback")
def evolve_rollback(
    skill_name: str = typer.Option(..., help="Skill name"),
    target: Optional[str] = typer.Option(None, help="Optional snapshot target"),
    config_path: Optional[Path] = typer.Option(None, help="Custom config file path"),
) -> None:
    config = load_config(path=config_path)
    try:
        payload = asyncio.run(
            rollback_skill(
                skills_local_dir=config.resolved_skills_local_dir(),
                skills_dir=config.resolved_skills_dir(),
                skill_name=skill_name,
                target=target,
            )
        )
    except RuntimeError as exc:
        console.print(f"[red]rollback failed: {exc}[/red]")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(payload, ensure_ascii=False))

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from aha.config import AhaConfig, load_config
from aha.core.context import ContextWeaver
from aha.core.loop import AgentLoop
from aha.core.session import RuntimeSessionState, Session, SessionStore
from aha.providers.base import LLMProvider
from aha.providers.litellm_provider import LiteLLMProvider
from aha.providers.mock_provider import MockProvider
from aha.runtime_log import configure_runtime_logging, log_event, resolve_log_level_name
from aha.tools.fs import ReadFileTool, WriteFileTool
from aha.tools.memory import MemoryStore
from aha.tools.policy import ToolPolicy
from aha.tools.registry import ToolRegistry
from aha.tools.runner import ToolRunner
from aha.tools.shell import ShellTool
from aha.tools.skill_manager import SkillManagerTool
from aha.tools.web import WebFetchTool, WebSearchTool

app = typer.Typer(help="AHA MVP CLI", no_args_is_help=False, invoke_without_command=True)
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug runtime logging"),
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
    log_level: str | None,
    log_dir: Path | None,
    no_runtime_log: bool,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if debug:
        overrides["runtime_log_mode"] = "debug"
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


def _select_provider(config: AhaConfig, logger: logging.Logger | None = None) -> LLMProvider:
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
) -> tuple[AgentLoop, SessionStore, MemoryStore, AhaConfig, logging.Logger, Path | None]:
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
    registry.register(SkillManagerTool(config.resolved_skills_local_dir()))

    policy = ToolPolicy(workspace=workspace, skills_local=config.resolved_skills_local_dir())
    runner = ToolRunner(registry=registry, policy=policy, logger=runtime_logger.getChild("tools.runner"))

    memory = MemoryStore(
        memory_file=config.resolved_memory_dir() / "MEMORY.md",
        trace_file=config.resolved_memory_dir() / "TRACE.jsonl",
        include_sensitive_data=config.trace_include_sensitive_data,
        trace_max_chars=config.trace_max_chars,
        trace_max_bytes=config.trace_max_bytes,
    )
    provider = _select_provider(config, logger=runtime_logger)
    loop = AgentLoop(
        provider=provider,
        tool_runner=runner,
        memory=memory,
        context_weaver=ContextWeaver(token_budget=config.token_budget),
        tool_names=registry.names(),
        workspace=str(workspace),
        max_steps=config.max_steps,
        logger=runtime_logger.getChild("loop"),
    )
    store = SessionStore(config.resolved_sessions_dir())
    return loop, store, memory, config, runtime_logger, runtime_log_path


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


def _print_session_banner(session_id: str, config: AhaConfig) -> None:
    console.print(
        Panel(
            f"Session: {session_id} | provider={config.provider} "
            f"model={config.model}",
            title="AHA",
        )
    )


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
                console.print(f"[bold green]aha[/bold green]: {reply}")
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
    finally:
        runner.close()


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
    console.print(Panel(summary, title="AHA Doctor"))
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

from __future__ import annotations

import contextvars
import logging
import re
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from ana.config import AnaConfig

_SESSION_ID = contextvars.ContextVar[str]("ana_log_session_id", default="-")
_STEP = contextvars.ContextVar[str]("ana_log_step", default="-")
_TOOL = contextvars.ContextVar[str]("ana_log_tool", default="-")


def redact_text(text: str) -> str:
    redacted = re.sub(r"sk-[A-Za-z0-9]{16,}", "[REDACTED_API_KEY]", text)
    redacted = re.sub(r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]", redacted)
    redacted = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY]", redacted)
    redacted = re.sub(r"(?i)aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+=]{20,}", "aws_secret_access_key=[REDACTED]", redacted)
    redacted = re.sub(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b", "[REDACTED_JWT]", redacted)
    redacted = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]", redacted)
    redacted = re.sub(
        r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\\s\"']+",
        "[REDACTED_DB_CONN]",
        redacted,
    )
    redacted = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*[^\s]+", r"\1=[REDACTED]", redacted)
    return redacted


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = _SESSION_ID.get()
        if not hasattr(record, "step"):
            record.step = _STEP.get()
        if not hasattr(record, "tool"):
            record.tool = _TOOL.get()
        if not hasattr(record, "event"):
            record.event = "-"
        return True


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact_text(message)
        record.args = ()
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            record.fields = {key: redact_text(str(value)) for key, value in fields.items()}
        return True


class DeveloperConsoleFilter(logging.Filter):
    _NOISY_EVENTS = {
        "llm_call_start",
        "llm_call_ok",
        "tool_dispatch",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        event = str(getattr(record, "event", "-"))
        return event not in self._NOISY_EVENTS


class DeveloperConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", "-"))
        session_id = str(getattr(record, "session_id", "-"))
        step = str(getattr(record, "step", "-"))
        tool = str(getattr(record, "tool", "-"))
        fields = getattr(record, "fields", {})
        if not isinstance(fields, dict):
            fields = {}
        sid = session_id if len(session_id) <= 16 else session_id[-16:]

        if event == "runtime_init":
            return (
                "[runtime] initialized "
                f"mode={fields.get('mode', '-')} level={fields.get('level_name', '-')} "
                f"log={fields.get('log_path', '-')}"
            )
        if event == "chat_start":
            return (
                "[chat] started "
                f"session={fields.get('session', sid)} provider={fields.get('provider', '-')} "
                f"model={fields.get('model', '-')}"
            )
        if event == "turn_start":
            return (
                "[turn] start "
                f"session={sid} user_len={fields.get('user_len', '-')} "
                f"messages={fields.get('message_count', '-')} tools={fields.get('tool_count', '-')}"
            )
        if event == "llm_request":
            return (
                f"[step {step}] model request "
                f"messages={fields.get('message_count', '-')} tools={fields.get('tools_count', '-')}"
            )
        if event == "llm_response":
            return (
                f"[step {step}] model response "
                f"latency={fields.get('latency_ms', '-')}ms "
                f"tool_calls={fields.get('tool_calls_count', '-')} "
                f"text_len={fields.get('assistant_len', '-')}"
            )
        if event == "tool_precheck":
            return (
                f"[tool {tool}] precheck "
                f"allow={fields.get('allowed', '-')} confirm={fields.get('requires_confirmation', '-')} "
                f"reason={fields.get('reason', '-')}"
                f"{self._tool_hint(fields)}"
            )
        if event == "tool_precheck_blocked":
            return f"[tool {tool}] blocked reason={fields.get('reason', '-')}"
        if event == "tool_confirm_prompted":
            return (
                f"[tool {tool}] confirmation required "
                f"capability={fields.get('temporary_capability', '-')}"
            )
        if event == "tool_confirm_denied":
            return f"[tool {tool}] confirmation denied"
        if event == "tool_confirm_approved":
            return f"[tool {tool}] confirmation approved"
        if event == "tool_run_start":
            return f"[tool {tool}] run start"
        if event == "tool_run_end":
            return (
                f"[tool {tool}] run end "
                f"ok={fields.get('ok', '-')} duration={fields.get('duration_ms', '-')}ms "
                f"warnings={fields.get('warnings_count', '-')} redactions={fields.get('redactions_count', '-')}"
            )
        if event == "tool_run_error":
            return (
                f"[tool {tool}] run error "
                f"duration={fields.get('duration_ms', '-')}ms "
                f"type={fields.get('error_type', '-')} error={fields.get('error', '-')}"
            )
        if event == "turn_final":
            return f"[turn] final step={step} text_len={fields.get('final_len', '-')}"
        if event == "max_steps_exhausted":
            return f"[turn] exhausted max_steps={fields.get('max_steps', '-')}"
        if event == "doctor_start":
            return (
                "[doctor] started "
                f"provider={fields.get('provider', '-')} model={fields.get('model', '-')} ping={fields.get('ping', '-')}"
            )
        if event == "doctor_ping_error":
            return f"[doctor] ping error type={fields.get('error_type', '-')} error={fields.get('error', '-')}"

        base = f"[{record.levelname.lower()}] event={event} session={sid}"
        if step != "-":
            base += f" step={step}"
        if tool != "-":
            base += f" tool={tool}"
        message = (record.getMessage() or "").strip()
        if message and message != "-":
            base += f" {message}"
        return base

    @staticmethod
    def _tool_hint(fields: dict[str, Any]) -> str:
        for key in (
            "path",
            "cmd",
            "url",
            "query_len",
            "max_results",
            "action",
            "name",
            "mode",
            "content_len",
            "timeout_seconds",
            "max_chars",
        ):
            if key in fields:
                return f" {key}={fields[key]}"
        return ""


def resolve_log_level_name(config: AnaConfig) -> str:
    if config.runtime_log_level:
        return str(config.runtime_log_level).upper()
    return "DEBUG" if config.runtime_log_mode == "debug" else "INFO"


def _parse_level(name: str) -> int:
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def _truncate(text: str, max_chars: int = 280) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _safe_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return _truncate(redact_text(str(value)))


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    safe_fields: dict[str, str] = {}
    for key, value in fields.items():
        if value is None:
            continue
        safe_fields[key] = _safe_text(value)
    pairs = [f"{key}={value}" for key, value in safe_fields.items()]
    message = " ".join(pairs) if pairs else "-"
    logger.log(level, message, extra={"event": event, "fields": safe_fields})


def configure_runtime_logging(config: AnaConfig, interactive: bool) -> tuple[logging.Logger, Path | None]:
    logger = logging.getLogger("ana")
    logger.propagate = False
    logger.disabled = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if not config.runtime_log_enabled:
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        return logger, None

    level_name = resolve_log_level_name(config)
    resolved_level = _parse_level(level_name)
    logger.setLevel(resolved_level)

    log_path = config.resolved_runtime_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max(1, int(config.runtime_log_max_bytes)),
        backupCount=max(0, int(config.runtime_log_backups)),
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s %(levelname)s %(name)s "
                "session=%(session_id)s step=%(step)s tool=%(tool)s "
                "event=%(event)s %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(ContextFilter())
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)

    if interactive and config.runtime_log_console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(DeveloperConsoleFormatter())
        console_handler.addFilter(ContextFilter())
        console_handler.addFilter(RedactingFilter())
        console_handler.addFilter(DeveloperConsoleFilter())
        logger.addHandler(console_handler)

    return logger, log_path


@contextmanager
def session_context(session_id: str) -> Iterator[None]:
    token = _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.reset(token)


@contextmanager
def step_context(step: int) -> Iterator[None]:
    token = _STEP.set(str(step))
    try:
        yield
    finally:
        _STEP.reset(token)


@contextmanager
def tool_context(tool_name: str) -> Iterator[None]:
    token = _TOOL.set(tool_name)
    try:
        yield
    finally:
        _TOOL.reset(token)

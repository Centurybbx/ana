from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from aha.config import AhaConfig

_SESSION_ID = contextvars.ContextVar[str]("aha_log_session_id", default="-")
_STEP = contextvars.ContextVar[str]("aha_log_step", default="-")
_TOOL = contextvars.ContextVar[str]("aha_log_tool", default="-")


def redact_text(text: str) -> str:
    redacted = re.sub(r"sk-[A-Za-z0-9]{16,}", "[REDACTED_API_KEY]", text)
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
        return True


def resolve_log_level_name(config: AhaConfig) -> str:
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
    pairs: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        pairs.append(f"{key}={_safe_text(value)}")
    message = " ".join(pairs) if pairs else "-"
    logger.log(level, message, extra={"event": event})


def configure_runtime_logging(config: AhaConfig, interactive: bool) -> tuple[logging.Logger, Path | None]:
    del interactive  # Reserved for future console sink toggles.

    logger = logging.getLogger("aha")
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
    logger.setLevel(_parse_level(level_name))

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

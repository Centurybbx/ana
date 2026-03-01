from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(
        self,
        memory_file: Path,
        trace_file: Path,
        include_sensitive_data: bool,
        trace_max_chars: int,
        trace_max_bytes: int = 2_000_000,
    ):
        self.memory_file = memory_file
        self.trace_file = trace_file
        self.include_sensitive_data = include_sensitive_data
        self.trace_max_chars = trace_max_chars
        self.trace_max_bytes = trace_max_bytes

    def read_memory(self) -> str:
        if not self.memory_file.exists():
            return ""
        return self.memory_file.read_text(encoding="utf-8")

    def append_memory_note(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        with self.memory_file.open("a", encoding="utf-8") as handle:
            handle.write(f"- {note}\n")

    def append_trace(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        if "observation" in payload and "observation_excerpt" not in payload:
            payload["observation_excerpt"] = payload.pop("observation")
        payload = self._sanitize(payload, force_redact=not self.include_sensitive_data)
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        self._rotate_if_needed(incoming_bytes=len(line) + 1)
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def tail_trace(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.trace_file.exists():
            return []
        lines = self.trace_file.read_text(encoding="utf-8").splitlines()
        items = []
        for line in lines[-limit:]:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def tail_trace_safe(self, limit: int = 20) -> list[dict[str, Any]]:
        items = self.tail_trace(limit=limit)
        return [self._sanitize(item, force_redact=True) for item in items]

    def redact(self, text: str) -> str:
        text = re.sub(r"sk-[A-Za-z0-9]{16,}", "[REDACTED_API_KEY]", text)
        text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", text)
        text = re.sub(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*[^\s]+", r"\1=[REDACTED]", text)
        return text

    def _truncate(self, text: str) -> str:
        if len(text) <= self.trace_max_chars:
            return text
        return text[: self.trace_max_chars] + "\n...[truncated]..."

    def _sanitize(self, value: Any, force_redact: bool) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                lowered = key.lower()
                if any(token in lowered for token in ("api_key", "token", "secret", "password", "authorization")):
                    sanitized[key] = "[REDACTED]"
                    continue
                sanitized[key] = self._sanitize(item, force_redact=force_redact)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize(item, force_redact=force_redact) for item in value]
        if isinstance(value, str):
            text = self.redact(value) if force_redact else value
            return self._truncate(text)
        return value

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.trace_max_bytes <= 0:
            return
        if not self.trace_file.exists():
            return
        current_size = self.trace_file.stat().st_size
        if current_size + incoming_bytes <= self.trace_max_bytes:
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = self.trace_file.with_name(f"{self.trace_file.stem}.{ts}{self.trace_file.suffix}")
        self.trace_file.rename(rotated)
        self._prune_rotations(keep=5)

    def _prune_rotations(self, keep: int) -> None:
        pattern = f"{self.trace_file.stem}.*{self.trace_file.suffix}"
        rotated_files = sorted(
            self.trace_file.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in rotated_files[keep:]:
            stale.unlink(missing_ok=True)

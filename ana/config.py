from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AnaConfig(BaseModel):
    provider: str = "mock"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    endpoint: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    request_timeout_seconds: int = 60
    temperature: float | None = None
    max_completion_tokens: int | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    max_steps: int = 30
    token_budget: int = 8000
    trace_include_sensitive_data: bool = False
    shell_timeout_seconds: int = 20
    trace_max_chars: int = 1200
    trace_max_bytes: int = 2_000_000
    runtime_log_enabled: bool = True
    runtime_log_console: bool = False
    runtime_log_dir: Path = Field(default=Path("logs"))
    runtime_log_level: str | None = None
    runtime_log_max_bytes: int = 2_000_000
    runtime_log_backups: int = 5
    runtime_log_mode: str = "prod"
    evolution_enabled: bool = False
    evolution_schedule: str = "daily"  # daily | weekly | custom
    evolution_custom_interval_minutes: int = 1440
    evolution_skill_name: str = "planner"
    evolution_llm_judge: bool = False
    evolution_latency_regression_pct: float = 20.0
    evolution_max_examples: int = 0
    evolution_run_on_startup: bool = True
    evolution_dataset_overrides: dict[str, str] = Field(default_factory=dict)
    evolution_baseline_file: str | None = None
    channels: dict[str, Any] = Field(default_factory=dict)
    im_auto_approve: bool = False

    workspace_dir: Path = Field(default_factory=Path.cwd)
    sessions_dir: Path = Field(default=Path("sessions"))
    memory_dir: Path = Field(default=Path("memory"))
    skills_dir: Path = Field(default=Path("skills"))
    skills_local_dir: Path = Field(default=Path("skills_local"))

    def resolved_api_base(self) -> str | None:
        # `endpoint` is the user-facing generic name; keep `api_base` for compatibility.
        return self.endpoint or self.api_base

    def resolved_workspace(self) -> Path:
        return self.workspace_dir.resolve()

    def resolved_sessions_dir(self) -> Path:
        return (self.resolved_workspace() / self.sessions_dir).resolve()

    def resolved_memory_dir(self) -> Path:
        return (self.resolved_workspace() / self.memory_dir).resolve()

    def resolved_skills_dir(self) -> Path:
        return (self.resolved_workspace() / self.skills_dir).resolve()

    def resolved_skills_local_dir(self) -> Path:
        return (self.resolved_workspace() / self.skills_local_dir).resolve()

    def resolved_runtime_log_dir(self) -> Path:
        if self.runtime_log_dir.is_absolute():
            return self.runtime_log_dir.resolve()
        return (self.resolved_workspace() / self.runtime_log_dir).resolve()

    def resolved_runtime_log_path(self) -> Path:
        filename = "ana.debug.log" if self.runtime_log_mode == "debug" else "ana.log"
        return self.resolved_runtime_log_dir() / filename

    def resolved_evolution_interval_seconds(self) -> int:
        schedule = str(self.evolution_schedule).strip().lower()
        if schedule == "weekly":
            return 7 * 24 * 60 * 60
        if schedule == "custom":
            minutes = max(1, int(self.evolution_custom_interval_minutes))
            return minutes * 60
        return 24 * 60 * 60

    def resolved_evolution_dataset_overrides(self) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for name, raw_path in self.evolution_dataset_overrides.items():
            dataset_name = str(name).strip()
            path_text = str(raw_path).strip()
            if not dataset_name or not path_text:
                continue
            path = Path(path_text)
            if not path.is_absolute():
                path = (self.resolved_workspace() / path).resolve()
            resolved[dataset_name] = path
        return resolved

    def resolved_evolution_baseline_file(self) -> Path | None:
        value = str(self.evolution_baseline_file or "").strip()
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.resolved_workspace() / path).resolve()
        return path

    def ensure_dirs(self) -> None:
        self.resolved_sessions_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_memory_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_skills_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_skills_local_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_runtime_log_dir().mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> AnaConfig:
    config_path = path or (Path.home() / ".ana" / "config.json")
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})

    config = AnaConfig(**data)
    config.ensure_dirs()
    memory_dir = config.resolved_memory_dir()
    memory_file = memory_dir / "MEMORY.md"
    trace_file = memory_dir / "TRACE.jsonl"
    if not memory_file.exists():
        memory_file.write_text("# ANA Memory\n\n", encoding="utf-8")
    if not trace_file.exists():
        trace_file.write_text("", encoding="utf-8")
    return config

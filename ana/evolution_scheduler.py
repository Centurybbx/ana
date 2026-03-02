from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

RunCycleFn = Callable[[], dict[str, Any]]


class EvolutionScheduler:
    def __init__(
        self,
        *,
        interval_seconds: int,
        run_cycle: RunCycleFn,
        logger: logging.Logger | None = None,
        run_on_startup: bool = True,
    ):
        self.interval_seconds = max(60, int(interval_seconds))
        self.run_cycle = run_cycle
        self.logger = logger or logging.getLogger("ana.evolution")
        self.run_on_startup = bool(run_on_startup)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="ana-evolution-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        if self.run_on_startup and not self._stop_event.is_set():
            self._run_once()
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.interval_seconds):
                break
            self._run_once()

    def _run_once(self) -> None:
        started = time.perf_counter()
        try:
            result = self.run_cycle()
            status = str(result.get("status", "ok"))
            self.logger.info(
                "evolution cycle finished status=%s duration_ms=%s",
                status,
                int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # pragma: no cover - defensive for background thread
            self.logger.error("evolution cycle failed: %s", exc, exc_info=True)

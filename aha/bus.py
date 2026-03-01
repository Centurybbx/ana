from __future__ import annotations

import asyncio
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def publish(self, event: dict[str, Any]) -> None:
        await self._queue.put(event)

    async def next(self) -> dict[str, Any]:
        return await self._queue.get()


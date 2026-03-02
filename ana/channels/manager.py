from __future__ import annotations

import asyncio
from collections.abc import Iterable

from ana.bus import MessageBus, OutboundMessage
from ana.channels.base import BaseChannel


class ChannelManager:
    def __init__(self, bus: MessageBus, channels: Iterable[BaseChannel]):
        self.bus = bus
        self.channels = {channel.name: channel for channel in channels}
        self._dispatcher: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        for channel in self.channels.values():
            await channel.start()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="ana.channel_dispatcher")

    async def stop(self) -> None:
        self._stopping = True
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except asyncio.CancelledError:
                pass
            self._dispatcher = None
        for channel in self.channels.values():
            await channel.stop()

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            try:
                outbound = await self.bus.next_outbound()
            except asyncio.CancelledError:
                break
            await self.dispatch_once(outbound)

    async def dispatch_once(self, outbound: OutboundMessage) -> None:
        channel = self.channels.get(outbound.channel)
        if channel is None:
            return
        limit = _message_limit(outbound.channel)
        chunks = _split_message(outbound.content, limit=limit)
        for idx, chunk in enumerate(chunks):
            reply_to = outbound.reply_to if idx == 0 else None
            await channel.send(
                chat_id=outbound.chat_id,
                content=chunk,
                reply_to=reply_to,
                metadata=outbound.metadata,
            )


def _message_limit(channel: str) -> int:
    if channel == "discord":
        return 2000
    if channel == "telegram":
        return 4000
    return 4000


def _split_message(content: str, limit: int) -> list[str]:
    text = str(content or "")
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + limit])
        start += limit
    return chunks


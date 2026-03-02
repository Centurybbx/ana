from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any


def sanitize_session_key(session_key: str) -> str:
    """Normalize session keys into filename-safe identifiers."""
    cleaned = re.sub(r"[:/\\\s]+", "_", str(session_key).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "default"


def session_id_from_key(session_key: str) -> str:
    return f"im:{sanitize_session_key(session_key)}"


@dataclass(slots=True)
class InboundMessage:
    channel: str
    sender_id: str
    chat_id: str
    content: str
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_key_override: str | None = None

    @property
    def session_key(self) -> str:
        if self.session_key_override:
            return str(self.session_key_override)
        return f"{self.channel}:{self.chat_id}"


@dataclass(slots=True)
class OutboundMessage:
    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Typed bus with separate inbound/outbound queues for IM runtime."""

    def __init__(self, *, inbound_maxsize: int = 0, outbound_maxsize: int = 0) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max(0, int(inbound_maxsize)))
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=max(0, int(outbound_maxsize)))

    async def publish_inbound(self, event: InboundMessage) -> None:
        await self._inbound.put(event)

    async def next_inbound(self) -> InboundMessage:
        return await self._inbound.get()

    async def publish_outbound(self, event: OutboundMessage) -> None:
        await self._outbound.put(event)

    async def next_outbound(self) -> OutboundMessage:
        return await self._outbound.get()

    @property
    def inbound(self) -> asyncio.Queue[InboundMessage]:
        return self._inbound

    @property
    def outbound(self) -> asyncio.Queue[OutboundMessage]:
        return self._outbound

    # Backward-compatible API for legacy single-queue EventBus use.
    async def publish(self, event: dict[str, Any]) -> None:
        await self.publish_inbound(
            InboundMessage(
                channel=str(event.get("channel", "legacy")),
                sender_id=str(event.get("sender_id", "unknown")),
                chat_id=str(event.get("chat_id", "default")),
                content=str(event.get("content", "")),
                media=[str(item) for item in event.get("media", []) if item is not None],
                metadata={
                    k: v
                    for k, v in event.items()
                    if k not in {"channel", "sender_id", "chat_id", "content", "media", "session_key_override"}
                },
                session_key_override=str(event["session_key_override"]) if event.get("session_key_override") else None,
            )
        )

    async def next(self) -> dict[str, Any]:
        event = await self.next_inbound()
        payload: dict[str, Any] = {
            "channel": event.channel,
            "sender_id": event.sender_id,
            "chat_id": event.chat_id,
            "content": event.content,
            "media": list(event.media),
            "session_key_override": event.session_key_override,
        }
        payload.update(event.metadata)
        return payload


class EventBus(MessageBus):
    """Compatibility shim: keep old name while using the new MessageBus."""


from __future__ import annotations

import asyncio
from typing import Any

from ana.bus import InboundMessage, MessageBus
from ana.channels.base import BaseChannel


class DiscordChannel(BaseChannel):
    def __init__(
        self,
        *,
        token: str,
        bus: MessageBus,
        allow_from: list[str] | None = None,
        allow_channels: list[str] | None = None,
    ) -> None:
        self.token = token
        self.bus = bus
        self.allow_from = {str(item).strip() for item in (allow_from or []) if str(item).strip()}
        self.allow_channels = {str(item).strip() for item in (allow_channels or []) if str(item).strip()}
        self._client: Any | None = None
        self._client_task: Any | None = None
        self._bot_user_id: int | None = None

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        try:
            import discord
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("discord dependency not installed, run: pip install 'ana-agent[discord]'") from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready() -> None:
            if client.user is not None:
                self._bot_user_id = int(client.user.id)

        @client.event
        async def on_message(message: Any) -> None:
            if message.author is None or message.channel is None:
                return
            if message.author.bot:
                return

            text = str(message.content or "").strip()
            if not text:
                return
            if not self._allowed_sender(message.author):
                return
            if not self._allowed_channel(message.channel):
                return
            if self._is_group_context(message) and not self._should_respond_group(message):
                return

            session_override = None
            if getattr(message, "thread", None) is not None:
                session_override = f"discord:thread:{message.thread.id}"

            await self.bus.publish_inbound(
                InboundMessage(
                    channel=self.name,
                    sender_id=str(message.author.id),
                    chat_id=str(message.channel.id),
                    content=text,
                    session_key_override=session_override,
                    metadata={
                        "message_id": str(message.id),
                        "guild_id": str(message.guild.id) if message.guild else "",
                        "thread_id": str(message.thread.id) if getattr(message, "thread", None) else "",
                    },
                )
            )

        self._client_task = asyncio.create_task(client.start(self.token), name="ana.discord.client")
        for _ in range(100):
            if client.is_ready():
                break
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except Exception:
                pass
            self._client_task = None
        self._client = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._client is None:
            raise RuntimeError("discord channel is not started")
        channel = self._client.get_channel(int(chat_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(chat_id))
        reference = None
        if reply_to:
            reference = int(reply_to)
        if reference is not None:
            try:
                msg = await channel.fetch_message(reference)
                await channel.send(content, reference=msg)
                return
            except Exception:
                pass
        await channel.send(content)

    def _allowed_sender(self, author: Any) -> bool:
        if not self.allow_from:
            return True
        tokens = {str(author.id)}
        if getattr(author, "name", None):
            tokens.add(str(author.name))
        return bool(tokens & self.allow_from)

    def _allowed_channel(self, channel: Any) -> bool:
        if not self.allow_channels:
            return True
        return str(channel.id) in self.allow_channels

    @staticmethod
    def _is_group_context(message: Any) -> bool:
        return getattr(message, "guild", None) is not None

    def _should_respond_group(self, message: Any) -> bool:
        mentions = getattr(message, "mentions", []) or []
        if self._bot_user_id is not None and any(int(getattr(user, "id", -1)) == self._bot_user_id for user in mentions):
            return True
        reference = getattr(message, "reference", None)
        if reference is None:
            return False
        resolved = getattr(reference, "resolved", None)
        if resolved is None:
            return False
        author = getattr(resolved, "author", None)
        if author is None or self._bot_user_id is None:
            return False
        return int(getattr(author, "id", -1)) == self._bot_user_id

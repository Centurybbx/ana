from __future__ import annotations

from typing import Any

from ana.bus import InboundMessage, MessageBus
from ana.channels.base import BaseChannel


class TelegramChannel(BaseChannel):
    def __init__(
        self,
        *,
        token: str,
        bus: MessageBus,
        allow_from: list[str] | None = None,
        allow_chats: list[str] | None = None,
    ) -> None:
        self.token = token
        self.bus = bus
        self.allow_from = {str(item).strip() for item in (allow_from or []) if str(item).strip()}
        self.allow_chats = {str(item).strip() for item in (allow_chats or []) if str(item).strip()}
        self._app: Any | None = None
        self._bot_username = ""
        self._bot_id: int | None = None

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import Application, ContextTypes, MessageHandler, filters
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("telegram dependency not installed, run: pip install 'ana-agent[telegram]'") from exc

        app = Application.builder().token(self.token).build()
        self._app = app

        async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            message = update.effective_message
            chat = update.effective_chat
            user = update.effective_user
            if message is None or chat is None or user is None:
                return

            text = (message.text or message.caption or "").strip()
            if not text:
                return
            if not self._allowed_sender(user.id, user.username):
                return
            if not self._allowed_chat(chat.id):
                return
            if chat.type in {"group", "supergroup"} and not self._should_respond_group(text, message):
                return

            await self.bus.publish_inbound(
                InboundMessage(
                    channel=self.name,
                    sender_id=str(user.id),
                    chat_id=str(chat.id),
                    content=text,
                    metadata={
                        "message_id": message.message_id,
                        "chat_type": chat.type,
                        "username": user.username or "",
                    },
                )
            )

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        me = await app.bot.get_me()
        self._bot_username = str(me.username or "").lower()
        self._bot_id = int(me.id)

    async def stop(self) -> None:
        if self._app is None:
            return
        if self._app.updater is not None:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._app is None:
            raise RuntimeError("telegram channel is not started")
        kwargs: dict[str, Any] = {}
        if reply_to:
            kwargs["reply_parameters"] = {"message_id": int(reply_to)}
        await self._app.bot.send_message(chat_id=int(chat_id), text=content, **kwargs)

    def _allowed_sender(self, user_id: int, username: str | None) -> bool:
        if not self.allow_from:
            return True
        tokens = {str(user_id)}
        uname = str(username or "").strip()
        if uname:
            tokens.add(uname)
            tokens.add(f"@{uname}")
        return bool(tokens & self.allow_from)

    def _allowed_chat(self, chat_id: int) -> bool:
        if not self.allow_chats:
            return True
        return str(chat_id) in self.allow_chats

    def _should_respond_group(self, text: str, message: Any) -> bool:
        if self._bot_username and f"@{self._bot_username}" in text.lower():
            return True
        reply_to = getattr(message, "reply_to_message", None)
        if reply_to is None:
            return False
        from_user = getattr(reply_to, "from_user", None)
        if from_user is None or self._bot_id is None:
            return False
        return int(getattr(from_user, "id", -1)) == self._bot_id

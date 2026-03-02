from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from ana.bus import MessageBus
from ana.channels.telegram import TelegramChannel


class FakeBot:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self._next_id = 100

    async def send_message(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if "reply_parameters" in kwargs:
            raise TypeError("send_message() got an unexpected keyword argument 'reply_parameters'")
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def send_chat_action(self, **kwargs: Any) -> None:
        self.actions.append(kwargs)

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)


def test_telegram_send_fallback_to_reply_to_message_id_on_type_error() -> None:
    async def _run() -> None:
        channel = TelegramChannel(token="t", bus=MessageBus())
        fake_bot = FakeBot()
        channel._app = SimpleNamespace(bot=fake_bot)

        result = await channel.send(chat_id="123", content="hello", reply_to="9")

        assert len(fake_bot.calls) == 2
        assert fake_bot.calls[0]["chat_id"] == 123
        assert fake_bot.calls[0]["reply_parameters"] == {"message_id": 9}
        assert fake_bot.calls[1]["chat_id"] == 123
        assert fake_bot.calls[1]["reply_to_message_id"] == 9
        assert result == {"message_id": "101"}

    asyncio.run(_run())


def test_telegram_send_ignores_invalid_reply_target() -> None:
    async def _run() -> None:
        channel = TelegramChannel(token="t", bus=MessageBus())
        fake_bot = FakeBot()
        channel._app = SimpleNamespace(bot=fake_bot)

        result = await channel.send(chat_id="123", content="hello", reply_to="not-int")

        assert len(fake_bot.calls) == 1
        assert fake_bot.calls[0]["chat_id"] == 123
        assert "reply_to_message_id" not in fake_bot.calls[0]
        assert "reply_parameters" not in fake_bot.calls[0]
        assert result == {"message_id": "101"}

    asyncio.run(_run())


def test_telegram_send_supports_typing_and_edit_actions() -> None:
    async def _run() -> None:
        channel = TelegramChannel(token="t", bus=MessageBus())
        fake_bot = FakeBot()
        channel._app = SimpleNamespace(bot=fake_bot)

        action_result = await channel.send(
            chat_id="123",
            content="",
            metadata={"telegram_action": "typing"},
        )
        edit_result = await channel.send(
            chat_id="123",
            content="updated",
            metadata={"telegram_edit_message_id": "321"},
        )

        assert action_result is None
        assert edit_result == {"message_id": "321"}
        assert fake_bot.actions == [{"chat_id": 123, "action": "typing"}]
        assert fake_bot.edits == [{"chat_id": 123, "message_id": 321, "text": "updated"}]

    asyncio.run(_run())

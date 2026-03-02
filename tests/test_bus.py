from __future__ import annotations

import asyncio

from ana.bus import EventBus, InboundMessage, MessageBus, OutboundMessage, sanitize_session_key, session_id_from_key


def test_inbound_session_key_uses_override() -> None:
    inbound = InboundMessage(
        channel="discord",
        sender_id="u1",
        chat_id="c1",
        content="hello",
        session_key_override="discord:thread:abc",
    )
    assert inbound.session_key == "discord:thread:abc"


def test_sanitize_session_key_and_session_id() -> None:
    assert sanitize_session_key("telegram:123/456") == "telegram_123_456"
    assert session_id_from_key("telegram:123/456") == "im:telegram_123_456"


def test_message_bus_routes_inbound_and_outbound() -> None:
    async def _run() -> None:
        bus = MessageBus()
        inbound = InboundMessage(channel="telegram", sender_id="u2", chat_id="c2", content="ping")
        outbound = OutboundMessage(channel="telegram", chat_id="c2", content="pong")
        await bus.publish_inbound(inbound)
        await bus.publish_outbound(outbound)
        assert await bus.next_inbound() == inbound
        assert await bus.next_outbound() == outbound

    asyncio.run(_run())


def test_event_bus_compatibility_payload_roundtrip() -> None:
    async def _run() -> None:
        bus = EventBus()
        payload = {
            "channel": "legacy",
            "sender_id": "u3",
            "chat_id": "room-1",
            "content": "hello",
            "custom": {"x": 1},
        }
        await bus.publish(payload)
        result = await bus.next()
        assert result["channel"] == "legacy"
        assert result["content"] == "hello"
        assert result["custom"] == {"x": 1}

    asyncio.run(_run())


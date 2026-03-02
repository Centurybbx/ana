from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from ana.bus import InboundMessage, MessageBus, OutboundMessage, session_id_from_key
from ana.core.loop import AgentLoop
from ana.core.session import RuntimeSessionState, Session, SessionStore


class IMRuntime:
    def __init__(
        self,
        *,
        agent_loop: AgentLoop,
        session_store: SessionStore,
        bus: MessageBus,
        auto_approve: bool = False,
    ) -> None:
        self.agent_loop = agent_loop
        self.session_store = session_store
        self.bus = bus
        self.auto_approve = bool(auto_approve)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._session_states: dict[str, RuntimeSessionState] = {}
        self._consumer: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        self._consumer = asyncio.create_task(self._consume_inbound(), name="ana.im.inbound")

    async def stop(self) -> None:
        self._stopping = True
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
            self._consumer = None
        if self._inflight:
            pending = list(self._inflight)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._inflight.clear()

    async def enqueue(self, inbound: InboundMessage) -> None:
        await self.bus.publish_inbound(inbound)

    async def _consume_inbound(self) -> None:
        while not self._stopping:
            try:
                inbound = await self.bus.next_inbound()
            except asyncio.CancelledError:
                break
            task = asyncio.create_task(self.handle_inbound(inbound), name=f"ana.im.turn:{inbound.session_key}")
            self._inflight.add(task)
            task.add_done_callback(lambda done: self._inflight.discard(done))

    async def handle_inbound(self, inbound: InboundMessage) -> None:
        lock = self._locks[inbound.session_key]
        async with lock:
            session_id = session_id_from_key(inbound.session_key)
            session = self._load_or_create_session(session_id)
            state = self._session_states.setdefault(session_id, RuntimeSessionState())
            try:
                reply = await self.agent_loop.run_turn(
                    session,
                    self._format_user_input(inbound),
                    session_state=state,
                    confirm=self._confirm_action,
                )
                self.session_store.save(session)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=inbound.channel,
                        chat_id=inbound.chat_id,
                        content=reply,
                        reply_to=str(inbound.metadata.get("message_id") or "") or None,
                        metadata={
                            "session_id": session_id,
                            "sender_id": inbound.sender_id,
                        },
                    )
                )
            except Exception as exc:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=inbound.channel,
                        chat_id=inbound.chat_id,
                        content=f"[im-runtime-error] {type(exc).__name__}: {exc}",
                        reply_to=str(inbound.metadata.get("message_id") or "") or None,
                    )
                )

    async def _confirm_action(self, plan_text: str) -> bool:
        _ = plan_text
        return self.auto_approve

    def _load_or_create_session(self, session_id: str) -> Session:
        try:
            return self.session_store.load(session_id)
        except FileNotFoundError:
            session = Session(session_id=session_id, created_at=_utc_now())
            self.session_store.save(session)
            return session

    @staticmethod
    def _format_user_input(inbound: InboundMessage) -> str:
        meta = dict(inbound.metadata or {})
        safe_meta: dict[str, Any] = {}
        for key in ("message_id", "guild_id", "thread_id", "chat_type"):
            if key in meta and str(meta[key]).strip():
                safe_meta[key] = meta[key]
        attrs = [
            f"channel={inbound.channel}",
            f"chat_id={inbound.chat_id}",
            f"sender={inbound.sender_id}",
        ]
        for key, value in safe_meta.items():
            attrs.append(f"{key}={value}")
        header = "[" + " ".join(attrs) + "]"
        return f"{header}\n{inbound.content}"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


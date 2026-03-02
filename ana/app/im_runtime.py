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
        send_progress: bool = True,
    ) -> None:
        self.agent_loop = agent_loop
        self.session_store = session_store
        self.bus = bus
        self.auto_approve = bool(auto_approve)
        self.send_progress = bool(send_progress)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._session_states: dict[str, RuntimeSessionState] = {}
        self._latest_generation: dict[str, int] = defaultdict(int)
        self._session_tasks: dict[str, asyncio.Task[None]] = {}
        self._progress_message: dict[str, tuple[int, str]] = {}
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
        self._session_tasks.clear()
        self._progress_message.clear()

    async def enqueue(self, inbound: InboundMessage) -> None:
        await self.bus.publish_inbound(inbound)

    async def _consume_inbound(self) -> None:
        while not self._stopping:
            try:
                inbound = await self.bus.next_inbound()
            except asyncio.CancelledError:
                break
            task = await self._schedule_latest(inbound)
            self._inflight.add(task)
            task.add_done_callback(lambda done: self._inflight.discard(done))
            task.add_done_callback(lambda done, key=inbound.session_key: self._cleanup_session_task(key, done))

    async def handle_inbound(self, inbound: InboundMessage) -> None:
        # Compatibility path for tests/callers that directly invoke handle_inbound.
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

    async def _schedule_latest(self, inbound: InboundMessage) -> asyncio.Task[None]:
        session_key = inbound.session_key
        self._record_user_history(inbound)
        previous = self._session_tasks.get(session_key)
        if previous is not None and not previous.done():
            previous.cancel()
        previous_generation = int(self._latest_generation.get(session_key, 0))
        generation = previous_generation + 1
        self._latest_generation[session_key] = generation
        await self._mark_superseded_progress_message(
            inbound=inbound,
            session_key=session_key,
            new_generation=generation,
        )
        task = asyncio.create_task(
            self._handle_latest_inbound(inbound=inbound, generation=generation),
            name=f"ana.im.turn.latest:{inbound.session_key}:{generation}",
        )
        self._session_tasks[session_key] = task
        return task

    async def _handle_latest_inbound(self, inbound: InboundMessage, generation: int) -> None:
        session_key = inbound.session_key
        lock = self._locks[session_key]
        async with lock:
            if not self._is_latest_generation(session_key, generation):
                return
            progress_stop = asyncio.Event()
            progress_task: asyncio.Task[None] | None = None
            if self.send_progress and inbound.channel == "telegram":
                progress_task = asyncio.create_task(
                    self._telegram_progress_worker(
                        inbound=inbound,
                        session_key=session_key,
                        generation=generation,
                        stop_event=progress_stop,
                    ),
                    name=f"ana.im.progress:{session_key}:{generation}",
                )
            session_id = session_id_from_key(session_key)
            session = self._load_or_create_session(session_id)
            state = self._session_states.setdefault(session_id, RuntimeSessionState())
            try:
                reply = await self.agent_loop.run_turn(
                    session,
                    self._format_user_input(inbound),
                    session_state=state,
                    confirm=self._confirm_action,
                )
                if not self._is_latest_generation(session_key, generation):
                    self._drop_stale_latest_assistant(session, reply)
                    self.session_store.save(session)
                    return
                self.session_store.save(session)
                progress_message_id = self._progress_message_id(session_key, generation)
                if inbound.channel == "telegram" and progress_message_id:
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=inbound.channel,
                            chat_id=inbound.chat_id,
                            content=reply,
                            metadata={"telegram_edit_message_id": progress_message_id},
                        )
                    )
                else:
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_latest_generation(session_key, generation):
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=inbound.channel,
                            chat_id=inbound.chat_id,
                            content=f"[im-runtime-error] {type(exc).__name__}: {exc}",
                            reply_to=str(inbound.metadata.get("message_id") or "") or None,
                        )
                    )
            finally:
                progress_stop.set()
                if progress_task is not None:
                    progress_task.cancel()
                    await asyncio.gather(progress_task, return_exceptions=True)
                if self._progress_message_id(session_key, generation):
                    self._progress_message.pop(session_key, None)

    async def _telegram_progress_worker(
        self,
        *,
        inbound: InboundMessage,
        session_key: str,
        generation: int,
        stop_event: asyncio.Event,
    ) -> None:
        spinner = ["正在思考…", "正在调用工具…", "正在整理答案…"]
        message_id = ""
        try:
            ack: asyncio.Future[dict[str, Any] | None] = asyncio.get_running_loop().create_future()
            if self._is_latest_generation(session_key, generation):
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=inbound.channel,
                        chat_id=inbound.chat_id,
                        content="正在思考…",
                        reply_to=str(inbound.metadata.get("message_id") or "") or None,
                        metadata={"im_status": "progress"},
                        ack=ack,
                    )
                )
            try:
                sent_meta = await asyncio.wait_for(ack, timeout=2.0)
                if isinstance(sent_meta, dict):
                    message_id = str(sent_meta.get("message_id") or "").strip()
            except Exception:
                message_id = ""
            if message_id:
                self._progress_message[session_key] = (generation, message_id)
            index = 0
            while not stop_event.is_set():
                if not self._is_latest_generation(session_key, generation):
                    break
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=inbound.channel,
                        chat_id=inbound.chat_id,
                        content="",
                        metadata={"telegram_action": "typing"},
                    )
                )
                if message_id:
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=inbound.channel,
                            chat_id=inbound.chat_id,
                            content=spinner[index % len(spinner)],
                            metadata={"telegram_edit_message_id": message_id, "im_status": "progress"},
                        )
                    )
                index += 1
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=1.2)
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise

    async def _mark_superseded_progress_message(
        self,
        *,
        inbound: InboundMessage,
        session_key: str,
        new_generation: int,
    ) -> None:
        progress = self._progress_message.get(session_key)
        if progress is None:
            return
        old_generation, old_message_id = progress
        if old_generation >= new_generation:
            return
        if inbound.channel != "telegram":
            self._progress_message.pop(session_key, None)
            return
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content="已根据你的最新消息调整…",
                metadata={"telegram_edit_message_id": old_message_id, "im_status": "superseded"},
            )
        )
        self._progress_message.pop(session_key, None)

    def _cleanup_session_task(self, session_key: str, task: asyncio.Task[None]) -> None:
        current = self._session_tasks.get(session_key)
        if current is task:
            self._session_tasks.pop(session_key, None)

    def _is_latest_generation(self, session_key: str, generation: int) -> bool:
        return int(self._latest_generation.get(session_key, 0)) == int(generation)

    def _progress_message_id(self, session_key: str, generation: int) -> str:
        progress = self._progress_message.get(session_key)
        if progress is None:
            return ""
        current_generation, message_id = progress
        if int(current_generation) != int(generation):
            return ""
        return str(message_id).strip()

    @staticmethod
    def _drop_stale_latest_assistant(session: Session, reply: str) -> None:
        if not session.messages:
            return
        last = session.messages[-1]
        if str(last.get("role", "")) != "assistant":
            return
        if str(last.get("content", "")) != str(reply):
            return
        session.messages.pop()

    def _record_user_history(self, inbound: InboundMessage) -> None:
        session = self._load_or_create_session(session_id_from_key(inbound.session_key))
        text = self._format_user_input(inbound)
        if session.messages:
            last = session.messages[-1]
            if str(last.get("role", "")) == "user" and str(last.get("content", "")) == text:
                return
        session.messages.append({"role": "user", "content": text})
        self.session_store.save(session)

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

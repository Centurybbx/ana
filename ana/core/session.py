from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    session_id: str
    created_at: str
    messages: list[dict] = field(default_factory=list)
    active_skills: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.session_id,
            "created_at": self.created_at,
            "messages": self.messages,
            "active_skills": self.active_skills,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Session":
        return cls(
            session_id=payload["id"],
            created_at=payload.get("created_at", _iso_now()),
            messages=payload.get("messages", []),
            active_skills=payload.get("active_skills", []),
        )


@dataclass
class RuntimeSessionState:
    capabilities: set[str] = field(default_factory=lambda: {"fs.read", "web.read"})
    active_skills: dict[str, dict] = field(default_factory=dict)
    dry_run: bool = False


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def create(self) -> Session:
        session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        session = Session(session_id=session_id, created_at=_iso_now())
        self.save(session)
        return session

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(payload)

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

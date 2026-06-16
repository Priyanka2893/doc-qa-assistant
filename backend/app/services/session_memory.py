from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

_SESSION_TTL_SECONDS = 1800  # 30 minutes


@dataclass
class Turn:
    question: str
    answer: str
    doc_id: str
    cited_sources: list[dict]
    timestamp: datetime


@dataclass
class Session:
    session_id: str
    doc_id: str
    turns: list[Turn]
    created_at: datetime
    last_active: datetime

    @property
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.last_active).total_seconds() > _SESSION_TTL_SECONDS

    def get_context_summary(self, max_turns: int = 3) -> str:
        if not self.turns:
            return ""
        recent = self.turns[-max_turns:]
        parts = [
            f"Previous Q: {t.question}\nPrevious A: {t.answer[:300]}..."
            for t in recent
        ]
        return "Previous conversation context:\n" + "\n\n".join(parts)


class SessionMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create_session(self, doc_id: str) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self._sessions[session_id] = Session(
            session_id=session_id,
            doc_id=doc_id,
            turns=[],
            created_at=now,
            last_active=now,
        )
        logger.info("session.created", session_id=session_id, doc_id=doc_id)
        return session_id

    def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None or session.is_expired:
            return None
        return session

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        doc_id: str,
        cited_sources: list,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.is_expired:
            return
        session.turns.append(Turn(
            question=question,
            answer=answer,
            doc_id=doc_id,
            cited_sources=[
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in cited_sources
            ],
            timestamp=datetime.utcnow(),
        ))
        session.last_active = datetime.utcnow()
        logger.info("session.turn_added", session_id=session_id, turn_count=len(session.turns))

    def get_context_for_query(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if session is None:
            return ""
        return session.get_context_summary()

    def cleanup_expired(self) -> int:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("session.cleanup", removed=len(expired))
        return len(expired)


_session_memory = SessionMemory()


def get_session_memory() -> SessionMemory:
    return _session_memory

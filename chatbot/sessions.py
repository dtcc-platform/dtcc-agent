"""In-memory session manager for chatbot conversations."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from dtcc_agent import artifacts


# A session ends after this long with no message (U7, #71). The MCP server
# keeps its own Sessions for as long; a test holds the two equal.
SESSION_IDLE_SECONDS = 3600


@dataclass
class Session:
    """A chatbot session."""

    id: str
    created_at: datetime = field(default_factory=datetime.now)
    # time.monotonic() of the last message or the end of the last turn.
    last_active: float = field(default_factory=time.monotonic)
    # Turns running now; the session never expires while nonzero.
    turns: int = 0
    sdk_session_id: str | None = None

    def expired(self, now: float) -> bool:
        return not self.turns and now - self.last_active > SESSION_IDLE_SECONDS


class SessionManager:
    """Manages chatbot sessions in memory.

    For prototype: no persistence. Sessions are lost on restart.
    A session idle for SESSION_IDLE_SECONDS is removed, with its files, by the
    next get() of it or the next create().
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> str:
        """Create a new session, return its ID."""
        self._cleanup()
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = Session(id=sid)
        return sid

    def get(self, session_id: str) -> Session | None:
        """Get a live session by ID, or None if not found or expired."""
        session = self._sessions.get(session_id)
        if session and session.expired(time.monotonic()):
            self.remove(session_id)
            return None
        return session

    def touch(self, session_id: str) -> bool:
        """Record activity on a live session. False if it is unknown or has
        expired: activity never revives a session."""
        session = self.get(session_id)
        if session:
            session.last_active = time.monotonic()
        return session is not None

    @contextmanager
    def turn(self, session_id: str) -> Iterator[None]:
        """Keep a live session from expiring while a turn runs, however long;
        the turn's end counts as activity."""
        session = self._sessions[session_id]
        session.turns += 1
        try:
            yield
        finally:
            session.turns -= 1
            session.last_active = time.monotonic()

    def get_sdk_session(self, session_id: str) -> str | None:
        """Get the Agent SDK session ID for resuming."""
        session = self._sessions.get(session_id)
        return session.sdk_session_id if session else None

    def set_sdk_session(self, session_id: str, sdk_session_id: str | None) -> None:
        """Store the Agent SDK session ID for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.sdk_session_id = sdk_session_id

    def remove(self, session_id: str) -> None:
        """Remove a session and the files it produced."""
        self._sessions.pop(session_id, None)
        artifacts.remove_session(session_id)

    def _cleanup(self) -> int:
        """Remove expired sessions. Returns count removed."""
        now = time.monotonic()
        expired = [sid for sid, s in self._sessions.items() if s.expired(now)]
        for sid in expired:
            self.remove(sid)
        return len(expired)

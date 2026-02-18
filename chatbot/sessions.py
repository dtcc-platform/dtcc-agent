"""In-memory session manager for chatbot conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Session:
    """A chatbot session."""

    id: str
    created_at: datetime = field(default_factory=datetime.now)
    sdk_session_id: str | None = None


class SessionManager:
    """Manages chatbot sessions in memory.

    For prototype: no persistence. Sessions are lost on restart.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> str:
        """Create a new session, return its ID."""
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = Session(id=sid)
        return sid

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def get_sdk_session(self, session_id: str) -> str | None:
        """Get the Agent SDK session ID for resuming."""
        session = self._sessions.get(session_id)
        return session.sdk_session_id if session else None

    def set_sdk_session(self, session_id: str, sdk_session_id: str) -> None:
        """Store the Agent SDK session ID for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.sdk_session_id = sdk_session_id

    def remove(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)

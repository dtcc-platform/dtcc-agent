from chatbot.sessions import SessionManager


def test_create_session_returns_id():
    mgr = SessionManager()
    sid = mgr.create()
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_get_session_returns_none_for_unknown():
    mgr = SessionManager()
    assert mgr.get("nonexistent") is None


def test_store_and_retrieve_session_id():
    mgr = SessionManager()
    sid = mgr.create()
    mgr.set_sdk_session(sid, "sdk-session-abc")
    assert mgr.get_sdk_session(sid) == "sdk-session-abc"


def test_remove_session():
    mgr = SessionManager()
    sid = mgr.create()
    mgr.remove(sid)
    assert mgr.get(sid) is None

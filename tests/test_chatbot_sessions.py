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


def test_an_expired_session_takes_its_artifacts_with_it(tmp_path, monkeypatch):
    from datetime import timedelta

    from chatbot import sessions as sessions_module
    from dtcc_agent import artifacts

    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    manager = sessions_module.SessionManager()
    sid = manager.create()
    artifacts.new_path(sid, "obj", ".png").write_bytes(b"x")
    manager.get(sid).created_at -= timedelta(seconds=sessions_module.MAX_AGE_SECONDS + 1)

    manager.create()  # expiry runs on create

    assert manager.get(sid) is None
    assert not (tmp_path / sid).exists()

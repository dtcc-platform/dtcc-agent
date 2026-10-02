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


def _idle(manager, sid, seconds):
    """Make `sid` look idle for `seconds`, without sleeping."""
    manager._sessions[sid].last_active -= seconds


def test_an_expired_session_takes_its_artifacts_with_it(tmp_path, monkeypatch):
    from chatbot import sessions as sessions_module
    from dtcc_agent import artifacts

    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    manager = sessions_module.SessionManager()
    sid = manager.create()
    artifacts.new_path(sid, "obj", ".png").write_bytes(b"x")
    _idle(manager, sid, sessions_module.SESSION_IDLE_SECONDS + 1)

    manager.create()  # expiry runs on create

    assert sid not in manager._sessions
    assert not (tmp_path / sid).exists()


def test_an_idle_session_expires_on_get_without_another_create(tmp_path, monkeypatch):
    from chatbot import sessions as sessions_module
    from dtcc_agent import artifacts

    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    manager = sessions_module.SessionManager()
    sid = manager.create()
    artifacts.new_path(sid, "obj", ".png").write_bytes(b"x")
    _idle(manager, sid, sessions_module.SESSION_IDLE_SECONDS + 1)

    assert manager.get(sid) is None
    assert not (tmp_path / sid).exists()


def test_activity_keeps_a_session_live_past_the_idle_limit():
    from chatbot.sessions import SESSION_IDLE_SECONDS

    manager = SessionManager()
    sid = manager.create()
    for _ in range(4):  # a message every 50 minutes for over 3 hours
        _idle(manager, sid, 50 * 60)
        assert manager.touch(sid)
    assert manager.get(sid) is not None
    assert 50 * 60 < SESSION_IDLE_SECONDS


def test_touch_does_not_revive_an_expired_or_unknown_session():
    from chatbot.sessions import SESSION_IDLE_SECONDS

    manager = SessionManager()
    sid = manager.create()
    _idle(manager, sid, SESSION_IDLE_SECONDS + 1)

    assert manager.touch(sid) is False
    assert manager.get(sid) is None
    assert manager.touch("nonexistent") is False


def test_a_turn_longer_than_the_idle_limit_does_not_expire_its_session():
    from chatbot.sessions import SESSION_IDLE_SECONDS

    manager = SessionManager()
    sid = manager.create()
    with manager.turn(sid):
        _idle(manager, sid, SESSION_IDLE_SECONDS + 10 * 60)  # a 70-minute turn
        manager.create()  # another visitor's create() sweeps meanwhile
        assert manager.get(sid) is not None
    # The turn's end counts as activity.
    assert manager.touch(sid)

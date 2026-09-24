"""Session isolation over the streamable-http transport (M1a/T5, #18).

ADR-0004: Objects and Runs belong to exactly one Session. The chatbot opens a
new MCP connection for every message, so the Session is carried as the
`X-DTCC-Session` header rather than tied to one MCP transport session, and
state must survive from one connection to the next.

These start a real `python -m dtcc_agent` over HTTP and talk to it with real
MCP clients.
"""

import json
import os
import socket
import subprocess
import sys
import time

import anyio
import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError
from mcp.client.streamable_http import streamable_http_client

import dtcc_agent.server as server
from dtcc_agent.server import SESSION_HEADER

FEATURES = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [319900.0, 6398900.0]},
            "properties": {"name": "a"},
        }
    ],
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    env = {
        **os.environ,
        "DTCC_MCP_TRANSPORT": "http",
        "DTCC_MCP_HOST": "127.0.0.1",
        "DTCC_MCP_PORT": str(port),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "dtcc_agent"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, "MCP server exited during startup"
            try:
                httpx.get(url, timeout=1)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline, "MCP server never started listening"
                time.sleep(0.2)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def geojson_file(tmp_path):
    path = tmp_path / "points.geojson"
    path.write_text(json.dumps(FEATURES))
    return str(path)


def _call(url, session_id, tool, args=None):
    """One MCP connection, one tool call: the shape of one chatbot message."""

    async def run():
        headers = {SESSION_HEADER: session_id} if session_id is not None else {}
        async with (
            httpx.AsyncClient(headers=headers) as http,
            streamable_http_client(url, http_client=http) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(tool, args or {})
            return result.isError, result.content[0].text

    return anyio.run(run)


def _ok(url, session_id, tool, args=None):
    is_error, text = _call(url, session_id, tool, args)
    assert not is_error, text
    return json.loads(text)


def test_objects_are_invisible_to_another_session(server_url, geojson_file):
    created = _ok(server_url, "session-a", "load_geojson", {"file_path": geojson_file})

    assert _ok(server_url, "session-b", "list_objects")["num_objects"] == 0
    other = _ok(server_url, "session-b", "inspect_object", {"object_id": created["object_id"]})
    assert "error" in other


def test_objects_survive_into_the_next_connection_of_the_same_session(
    server_url, geojson_file
):
    created = _ok(server_url, "session-c", "load_geojson", {"file_path": geojson_file})

    # A fresh MCP connection, as the chatbot opens for its next message.
    listed = _ok(server_url, "session-c", "list_objects")
    assert [o["id"] for o in listed["objects"]] == [created["object_id"]]


def test_a_tool_call_without_a_session_is_refused(server_url):
    is_error, text = _call(server_url, None, "list_objects")
    assert is_error
    assert SESSION_HEADER in text


def test_an_empty_session_header_is_refused(server_url):
    is_error, text = _call(server_url, "", "list_objects")
    assert is_error
    assert SESSION_HEADER in text


def test_the_http_transport_keeps_no_per_connection_state(server_url):
    # The chatbot opens a connection per message. A stateful transport keeps
    # one server task per connection, forever; the Session header already
    # identifies the caller, so the transport stays stateless.
    response = httpx.post(
        server_url,
        headers={"Accept": "application/json, text/event-stream", SESSION_HEADER: "s"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        },
    )
    assert response.status_code == 200
    assert "mcp-session-id" not in response.headers


def test_render_object_resolves_the_calling_session(server_url, geojson_file):
    created = _ok(server_url, "session-r", "load_geojson", {"file_path": geojson_file})
    args = {"object_id": created["object_id"]}

    # Found (a GeoJSON dict is not renderable), not "not found".
    own = _ok(server_url, "session-r", "render_object", args)
    assert "Unsupported type" in own["error"]
    other = _ok(server_url, "session-x", "render_object", args)
    assert "not found" in other["error"]


def test_stdio_serves_one_local_session_without_a_header(geojson_file):
    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "dtcc_agent"])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            created = await session.call_tool("load_geojson", {"file_path": geojson_file})
            listed = await session.call_tool("list_objects", {})
            return json.loads(created.content[0].text), json.loads(listed.content[0].text)

    created, listed = anyio.run(run)
    assert [o["id"] for o in listed["objects"]] == [created["object_id"]]


def test_an_http_call_with_no_request_is_refused_not_given_the_local_session(monkeypatch):
    monkeypatch.setattr(server, "_serving_http", True)
    with pytest.raises(ToolError, match=SESSION_HEADER):
        anyio.run(server.mcp.call_tool, "list_objects", {})


def test_an_invalid_transport_exits_with_the_allowed_values(monkeypatch):
    monkeypatch.setenv("DTCC_MCP_TRANSPORT", "bogus")
    with pytest.raises(SystemExit, match="'stdio' or 'http'"):
        server.main()


def test_sessions_are_capped_and_the_least_recently_used_is_dropped(monkeypatch):
    monkeypatch.setattr(server, "_sessions", server._sessions.__class__())
    ids = [f"s{i}" for i in range(server.MAX_SESSIONS)]
    first = {sid: server._session_for(sid) for sid in ids}

    server._session_for("s0")  # touch: s1 is now the least recently used
    server._session_for("new")

    assert server._session_for("s0") is first["s0"]
    assert server._session_for("s1") is not first["s1"]
    assert len(server._sessions) == server.MAX_SESSIONS


def test_the_session_budgets_add_up_to_the_process_budget():
    per_session = server._session_for("budget").objects._max_bytes
    assert per_session * server.MAX_SESSIONS <= server.OBJECT_BUDGET_BYTES


def test_runs_are_invisible_to_another_session():
    # No tool creates a Run without dtcc_sim and the network, so bind the
    # Session the way the tool wrapper does and use the real store path.
    a, b = server._Session(), server._Session()
    token = server._current_session.set(a)
    try:
        run_id = server._store_result("sim", [0, 0, 1, 1], {}, {"values": [1.0]})
    finally:
        server._current_session.reset(token)

    token = server._current_session.set(b)
    try:
        assert json.loads(server.list_past_runs()) == []
        assert "error" in json.loads(server.get_run_summary(run_id))
    finally:
        server._current_session.reset(token)
    assert run_id in a.results


def test_a_session_with_a_tool_in_flight_is_never_evicted(monkeypatch):
    # Evicting it mid-call would drop the result the tool is about to store.
    monkeypatch.setattr(server, "_sessions", server._sessions.__class__())
    busy = server._session_for("busy", acquire=True)
    try:
        for i in range(server.MAX_SESSIONS + 2):
            server._session_for(f"other{i}")
        assert server._session_for("busy") is busy
    finally:
        server._release(busy)

    for i in range(server.MAX_SESSIONS + 2):
        server._session_for(f"later{i}")
    assert "busy" not in server._sessions


def test_excess_sessions_are_trimmed_when_their_calls_finish(monkeypatch):
    monkeypatch.setattr(server, "_sessions", server._sessions.__class__())
    burst = [server._session_for(f"b{i}", acquire=True) for i in range(server.MAX_SESSIONS * 3)]
    assert len(server._sessions) == server.MAX_SESSIONS * 3  # all busy: cap exceeded

    for session in burst:
        server._release(session)

    assert len(server._sessions) == server.MAX_SESSIONS

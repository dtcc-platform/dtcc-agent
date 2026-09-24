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
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

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
        headers = {"X-DTCC-Session": session_id} if session_id else {}
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
    assert "X-DTCC-Session" in text


def test_runs_are_invisible_to_another_session():
    # No tool creates a Run without dtcc_sim and the network, so bind the
    # Session the way the tool wrapper does and use the real store path.
    import dtcc_agent.server as server

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

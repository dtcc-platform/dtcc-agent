"""How the MCP server executes tools (M1a/T4, #17).

dtcc_core calls `asyncio.run()` internally for its downloads (lidar, gpkg).
FastMCP calls a sync tool directly on the event loop, where `asyncio.run()`
raises, and `nest_asyncio` cannot patch uvloop, which `uvicorn[standard]`
installs. So every tool runs in a worker thread instead.
"""

import asyncio
import json
import threading

import pytest

import dtcc_agent.dispatcher as dispatcher
import dtcc_agent.renderer as renderer
import dtcc_agent.server as server

# uvloop comes only with the chatbot extra (uvicorn[standard]).
uvloop = pytest.importorskip("uvloop")


def _run_on_uvloop(coro):
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        return runner.run(coro)


def test_every_tool_except_render_is_registered_async():
    tools = server.mcp._tool_manager.list_tools()
    assert tools
    assert [t.name for t in tools if not t.is_async] == ["render_object"]


def test_render_object_runs_on_the_main_thread(monkeypatch):
    # GLFW must create its window on the main thread; on macOS doing it
    # anywhere else aborts the whole process.
    seen = {}

    def fake_render_to_file(**kwargs):
        seen["thread"] = threading.current_thread()
        return "/tmp/render.png"

    monkeypatch.setattr(renderer, "render_to_file", fake_render_to_file)
    monkeypatch.setattr(server, "_object_store", server.ObjectStore())
    object_id = server._object_store.store([], source_op="test")

    content, _ = _run_on_uvloop(
        server.mcp.call_tool("render_object", {"object_id": object_id})
    )

    assert "error" not in json.loads(content[0].text)
    assert seen["thread"] is threading.main_thread()


def test_a_tool_whose_core_call_uses_asyncio_run_succeeds_under_uvloop(monkeypatch):
    calls = {}

    def fake_dispatch(**kwargs):
        # Stand-in for a Core download: a nested asyncio.run() on the calling thread.
        calls["result"] = asyncio.run(asyncio.sleep(0, result="downloaded"))
        calls["thread"] = threading.current_thread()
        return {"result_id": "x", "summary": calls["result"]}

    monkeypatch.setattr(dispatcher, "run_operation", fake_dispatch)

    content, _ = _run_on_uvloop(
        server.mcp.call_tool("run_operation", {"name": "datasets.point_cloud"})
    )

    payload = json.loads(content[0].text)
    assert "error" not in payload
    assert payload["summary"] == "downloaded"
    assert calls["thread"] is not threading.main_thread()


def test_module_level_tools_stay_directly_callable():
    # In-process callers (and the characterisation tests) call tools as plain functions.
    assert isinstance(server.list_objects(), str)

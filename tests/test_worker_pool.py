"""A bounded worker pool for all Core execution (M1a/T8, #19).

T4 moved every tool body onto a worker thread, which lifted effective
concurrency from 1 to anyio's default of 40 threads. Each operation can
deepcopy a heavy input, so the pool is bounded per process, across Sessions,
and one Session may hold at most half of it.

Two calls that would download the same dataset tile share one download: the
later one waits on the event loop, before it takes a worker, and then reads
the cache.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from mcp.server.lowlevel.server import request_ctx

import dtcc_agent.dispatcher as dispatcher
import dtcc_agent.runner as runner
import dtcc_agent.server as server
from dtcc_agent.disk_cache import DiskCache
from dtcc_agent.server import SESSION_HEADER

BOUNDS = [319700, 6399500, 320200, 6400000]
BOUNDS_AS_FLOATS = [319700.0, 6399500.0, 320200.0, 6400000.0]


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    monkeypatch.setattr(server, "_sessions", OrderedDict())


async def _call(tool, args, session_id=None):
    """One tool call; with `session_id`, as an HTTP request carrying it."""
    if session_id is not None:
        request = SimpleNamespace(headers={SESSION_HEADER: session_id})
        request_ctx.set(SimpleNamespace(request=request))
    return await server.mcp.call_tool(tool, args)


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.002)


class _Gauge:
    """A tool body that records how many copies of it run at once.

    Each copy waits until `target` are running (or a short deadline), so a
    missing bound shows up as a peak above it rather than depending on timing.
    """

    def __init__(self, target):
        self.target = target
        self.running = 0
        self.peak = 0
        self._lock = threading.Lock()

    def body(self, **kwargs):
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        deadline = time.monotonic() + 0.3
        while self.running < self.target and time.monotonic() < deadline:
            time.sleep(0.002)
        with self._lock:
            self.running -= 1
        return {"operation": kwargs["name"]}


def _run_concurrently(calls):
    async def run():
        async with anyio.create_task_group() as tg:
            for session_id in calls:
                tg.start_soon(_call, "run_operation", {"name": "builder.x"}, session_id)

    anyio.run(run)


# -- The bound ---------------------------------------------------------------

def test_concurrent_calls_from_many_sessions_never_exceed_the_pool(monkeypatch):
    monkeypatch.setattr(server, "_workers", anyio.CapacityLimiter(2))
    gauge = _Gauge(target=6)
    monkeypatch.setattr(dispatcher, "run_operation", gauge.body)

    _run_concurrently([f"s{i}" for i in range(6)])

    assert gauge.peak == 2


def test_one_session_holds_at_most_its_share_of_the_pool(monkeypatch):
    # With the process-wide pool out of the way, one Session still cannot
    # take more than its share, so other Sessions always get a worker.
    monkeypatch.setattr(server, "_workers", anyio.CapacityLimiter(10))
    gauge = _Gauge(target=6)
    monkeypatch.setattr(dispatcher, "run_operation", gauge.body)

    _run_concurrently(["busy"] * 6)

    assert gauge.peak == server.SESSION_WORKERS < 6


def test_the_local_session_may_use_the_whole_pool(monkeypatch):
    # stdio has one client per process: nobody to be fair to.
    monkeypatch.setattr(server, "_workers", anyio.CapacityLimiter(4))
    gauge = _Gauge(target=6)
    monkeypatch.setattr(dispatcher, "run_operation", gauge.body)

    _run_concurrently([None] * 6)

    assert gauge.peak == 4


@pytest.mark.parametrize(
    "env_value, pool, share",
    [(None, "4", "2"), ("3", "3", "1"), ("1", "1", "1")],
)
def test_the_pool_and_session_share_are_sized_from_the_environment(env_value, pool, share):
    # The right size depends on the host's memory; 4 is the default.
    env = {k: v for k, v in os.environ.items() if k != "DTCC_MCP_WORKERS"}
    if env_value is not None:
        env["DTCC_MCP_WORKERS"] = env_value
    out = subprocess.run(
        [sys.executable, "-c",
         "import dtcc_agent.server as s; print(s._workers.total_tokens, s.SESSION_WORKERS)"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == [pool, share]


def test_a_bad_pool_size_stops_startup_naming_the_variable():
    for bad in ("0", "abc", "²"):
        env = {**os.environ, "DTCC_MCP_WORKERS": bad}
        out = subprocess.run(
            [sys.executable, "-c", "import dtcc_agent.server"],
            env=env, capture_output=True, text=True,
        )
        assert out.returncode != 0
        assert "DTCC_MCP_WORKERS" in out.stderr, out.stderr[-300:]


def test_a_main_thread_tool_runs_even_when_the_pool_is_full(monkeypatch):
    # Rendering bypasses the pool: it runs on the event loop and never
    # needed a worker thread.
    import dtcc_agent.renderer as renderer

    monkeypatch.setattr(renderer, "render_to_file", lambda **kw: "/tmp/render.png")
    monkeypatch.setattr(server, "_local_session", server._Session())
    object_id = server._session().objects.store([], source_op="test")
    full = anyio.CapacityLimiter(1)
    monkeypatch.setattr(server, "_workers", full)

    async def run():
        await full.acquire_on_behalf_of(object())
        with anyio.fail_after(2):
            return await server.mcp.call_tool("render_object", {"object_id": object_id})

    content, _ = anyio.run(run)
    assert "error" not in json.loads(content[0].text)


def test_a_tool_that_raises_gives_its_worker_back(monkeypatch):
    pool = anyio.CapacityLimiter(1)
    monkeypatch.setattr(server, "_workers", pool)

    def boom():
        raise RuntimeError("core crashed")

    monkeypatch.setattr(runner, "list_simulations", boom)

    with pytest.raises(Exception, match="core crashed"):
        anyio.run(_call, "list_simulations", {}, "s1")

    assert pool.borrowed_tokens == 0
    assert server._sessions["s1"].workers.borrowed_tokens == 0
    assert server._sessions["s1"].in_flight == 0


# -- One download per tile ---------------------------------------------------

class _FakePointCloud:
    """Picklable stand-in for a dtcc-core PointCloud."""

    def __init__(self, points):
        self.points = points


@pytest.fixture
def gated_download(monkeypatch, tmp_path):
    """datasets.point_cloud through the real dispatcher and a fresh disk
    cache, with a download that blocks until `gate` is set."""
    gate = threading.Event()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert gate.wait(5), "the test never released the download"
        return _FakePointCloud(points=[[319800.0, 6399600.0, 5.0]])

    op = MagicMock(category="datasets", _callable=download)
    op.name = "datasets.point_cloud"
    monkeypatch.setattr(dispatcher, "get_operation", lambda name: op)
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))
    return SimpleNamespace(gate=gate, calls=calls)


def _point_cloud(bounds):
    return {"name": "datasets.point_cloud", "params": {"bounds": bounds, "source": "LM"}}


def _flight_holders(args):
    flight = server._flights.get(server._run_operation_flight(args))
    return flight.holders if flight else 0


def test_two_sessions_asking_for_one_tile_download_it_once(gated_download):
    # Bounds spelled as ints and as floats are the same tile.
    results = {}

    async def fetch(session_id, bounds):
        content, _ = await _call("run_operation", _point_cloud(bounds), session_id)
        results[session_id] = json.loads(content[0].text)

    async def run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(fetch, "a", BOUNDS)
            tg.start_soon(fetch, "b", BOUNDS_AS_FLOATS)
            await anyio.to_thread.run_sync(
                _wait_until, lambda: _flight_holders(_point_cloud(BOUNDS)) == 2
            )
            gated_download.gate.set()

    anyio.run(run)

    assert len(gated_download.calls) == 1
    assert sorted(bool(r.get("cache_hit")) for r in results.values()) == [False, True]
    assert server._flights == {}


def test_a_call_waiting_for_a_tile_holds_no_worker(monkeypatch, gated_download):
    # Otherwise four requests for one slow tile would take every worker, and
    # no other Session could run anything until the download finished.
    pool = anyio.CapacityLimiter(2)
    monkeypatch.setattr(server, "_workers", pool)

    async def run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, "run_operation", _point_cloud(BOUNDS), "a")
            tg.start_soon(_call, "run_operation", _point_cloud(BOUNDS), "b")
            await anyio.to_thread.run_sync(
                _wait_until, lambda: _flight_holders(_point_cloud(BOUNDS)) == 2
            )
            await anyio.to_thread.run_sync(_wait_until, lambda: len(gated_download.calls) == 1)
            assert pool.borrowed_tokens == 1  # only the download
            with anyio.fail_after(2):
                await _call("list_objects", {}, "c")  # the free worker
            gated_download.gate.set()

    anyio.run(run)
    assert pool.borrowed_tokens == 0


def test_a_cancelled_wait_for_a_tile_leaves_nothing_behind(monkeypatch, gated_download):
    pool = anyio.CapacityLimiter(2)
    monkeypatch.setattr(server, "_workers", pool)

    async def run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, "run_operation", _point_cloud(BOUNDS), "a")
            with anyio.move_on_after(5) as waiter:
                async with anyio.create_task_group() as inner:
                    inner.start_soon(_call, "run_operation", _point_cloud(BOUNDS), "b")
                    await anyio.to_thread.run_sync(
                        _wait_until, lambda: _flight_holders(_point_cloud(BOUNDS)) == 2
                    )
                    inner.cancel_scope.cancel()
            assert not waiter.cancelled_caught  # the wait ended by cancellation, not timeout
            assert _flight_holders(_point_cloud(BOUNDS)) == 1
            gated_download.gate.set()

    anyio.run(run)

    assert server._flights == {}
    assert pool.borrowed_tokens == 0
    assert len(gated_download.calls) == 1


def test_a_busy_session_never_holds_a_tile_another_session_needs(monkeypatch, tmp_path):
    # A Session whose share is in use waits for it before claiming a tile,
    # so another Session asking for that tile starts at once on a free worker.
    monkeypatch.setattr(server, "_workers", anyio.CapacityLimiter(4))
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))
    gate = threading.Event()
    wanted = [0, 0, 1, 1]

    def download(**kwargs):
        if kwargs["bounds"] != wanted:
            assert gate.wait(5)
        return _FakePointCloud(points=[[0.5, 0.5, 1.0]])

    op = MagicMock(category="datasets", _callable=download)
    op.name = "datasets.point_cloud"
    monkeypatch.setattr(dispatcher, "get_operation", lambda name: op)

    async def run():
        async with anyio.create_task_group() as tg:
            for i in range(server.SESSION_WORKERS):  # the busy Session's whole share
                tg.start_soon(_call, "run_operation", _point_cloud([10 + i, 0, 11 + i, 1]), "busy")
            await anyio.to_thread.run_sync(
                _wait_until,
                lambda: "busy" in server._sessions
                and server._sessions["busy"].workers.borrowed_tokens == server.SESSION_WORKERS,
            )
            tg.start_soon(_call, "run_operation", _point_cloud(wanted), "busy")
            await anyio.sleep(0.05)
            with anyio.fail_after(2):
                await _call("run_operation", _point_cloud(wanted), "other")
            gate.set()

    anyio.run(run)


def test_two_sessions_never_download_one_area_s_buildings_at_once(monkeypatch, tmp_path):
    # Overlapping downloads of one tile can fail inside Core (dtcc-core#126).
    # max_buildings only trims the answer, so it shares the flight too; its
    # cache entry differs, so the second call downloads again, but after.
    gate = threading.Event()
    calls, running, peak = [], [0], [0]

    def fetch(**kwargs):
        calls.append(kwargs)
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        assert gate.wait(5)
        running[0] -= 1
        return {"buildings": [], "num_buildings": 0}

    monkeypatch.setattr(runner, "get_buildings", fetch)
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))
    key = server._get_buildings_flight({"bounds": BOUNDS, "source": "LM"})

    def holders():
        flight = server._flights.get(key)
        return flight.holders if flight else 0

    async def run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, "get_buildings", {"bounds": BOUNDS}, "a")
            tg.start_soon(_call, "get_buildings", {"bounds": BOUNDS, "max_buildings": 5}, "b")
            await anyio.to_thread.run_sync(_wait_until, lambda: holders() == 2)
            gate.set()

    anyio.run(run)

    assert len(calls) == 2 and peak[0] == 1


def test_the_two_ways_to_fetch_buildings_never_download_one_area_at_once(monkeypatch, tmp_path):
    # get_buildings and run_operation("datasets.buildings") make the same Core
    # download, so they share its flight (dtcc-core#126).
    gate = threading.Event()
    running, peak, calls = [0], [0], []
    lock = threading.Lock()

    def download(**kwargs):
        with lock:
            calls.append(kwargs)
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        assert gate.wait(5)
        with lock:
            running[0] -= 1
        return []

    monkeypatch.setattr(runner, "get_buildings", lambda **kw: {**kw, "buildings": download(**kw)})
    op = MagicMock(category="datasets", _callable=download)
    op.name = "datasets.buildings"
    monkeypatch.setattr(dispatcher, "get_operation", lambda name: op)
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))
    key = server._get_buildings_flight({"bounds": BOUNDS, "source": "LM"})

    def holders():
        flight = server._flights.get(key)
        return flight.holders if flight else 0

    async def run():
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, "get_buildings", {"bounds": BOUNDS}, "a")
            tg.start_soon(_call, "run_operation",
                          {"name": "datasets.buildings", "params": {"bounds": BOUNDS}}, "b")
            await anyio.to_thread.run_sync(_wait_until, lambda: holders() == 2)
            gate.set()

    anyio.run(run)

    assert len(calls) == 2 and peak[0] == 1


# -- Which calls share a flight ----------------------------------------------

def test_only_bounded_cached_dataset_calls_share_a_flight():
    key = server._run_operation_flight
    assert key(_point_cloud(BOUNDS)) == key(_point_cloud(BOUNDS_AS_FLOATS))
    assert key(_point_cloud(BOUNDS)) != key(_point_cloud([0, 0, 1, 1]))
    assert key({"name": "datasets.point_cloud", "params": {"bounds": BOUNDS, "source": "OSM"}}) \
        != key(_point_cloud(BOUNDS))
    # Builders key on their inputs' metadata, which can collide across
    # Sessions (U2, #11): sharing a flight would hand one the other's result.
    assert key({"name": "builder.build_terrain_raster", "params": {"bounds": BOUNDS}}) is None
    assert key({"name": "datasets.point_cloud", "params": {"source": "LM"}}) is None
    # The dispatcher downloads from LM when no source is given.
    assert key({"name": "datasets.point_cloud", "params": {"bounds": BOUNDS}}) == key(_point_cloud(BOUNDS))
    # Filtering happens after the download, so it does not split the flight.
    filtered = {"bounds": BOUNDS, "source": "LM", "classifications": ["terrain"]}
    assert key({"name": "datasets.point_cloud", "params": filtered}) == key(_point_cloud(BOUNDS))
    assert key({"name": "datasets.point_cloud", "params": None}) is None
    assert key({"name": "io.load_mesh", "params": {"bounds": BOUNDS}}) is None


def test_get_buildings_calls_share_a_flight_per_area_and_source():
    key = server._get_buildings_flight
    lm = {"bounds": BOUNDS, "source": "LM", "max_buildings": 100}
    assert key(lm) == key({**lm, "bounds": BOUNDS_AS_FLOATS})
    assert key(lm) == key({**lm, "max_buildings": 5})
    assert key(lm) != key({**lm, "source": "OSM"})
    # The same download as the buildings dataset.
    dataset = server._run_operation_flight
    assert key(lm) == dataset({"name": "datasets.buildings", "params": {"bounds": BOUNDS}})
    assert key({**lm, "source": "OSM"}) == dataset(
        {"name": "datasets.buildings", "params": {"bounds": BOUNDS, "source": "OSM"}})

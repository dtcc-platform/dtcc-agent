"""The catalogue is built once per process (M1a/T10, #22).

Building it takes over a second. Built lazily, the first request after every
restart paid for it, and a broken build surfaced as a request error. An HTTP
server builds it at startup (#12: in the app's lifespan), so it has it before
its first call and a broken Core install fails to start. stdio builds it on
first use: the chatbot starts a stdio server per message, and most never read
it.

What counts as broken (decided on #22): any section registered from the pinned
Core. The optional dtcc_sim package may fail with a warning. dtcc-sim's remote
datasets are not part of the build; they join the catalogue when it is next
read, so the two services don't have to start in a fixed order.
"""

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import dtcc_agent.registry as registry
import dtcc_agent.runner as runner
import dtcc_agent.server as server
from dtcc_agent.registry import CatalogueError, OperationInfo
from dtcc_agent.server import SESSION_HEADER

FAKE = {"io.fake": OperationInfo(name="io.fake", category="io", description="fake")}


@pytest.fixture(autouse=True)
def no_dtcc_sim_configured(monkeypatch):
    """Keep the developer's dtcc-sim settings, and earlier merges, out of tests."""
    monkeypatch.delenv("DTCC_SIM_SERVICE_URL", raising=False)
    monkeypatch.delenv("DTCC_REMOTE_SERVICES", raising=False)
    monkeypatch.setattr(registry, "_MERGED_REMOTE_SERVICES", set())
    monkeypatch.setattr(registry, "_retrier", None)


@pytest.fixture
def builds(monkeypatch):
    """A fresh, unbuilt catalogue whose (fake, instant) builds are counted."""
    calls = []

    def build():
        calls.append(threading.current_thread())
        return dict(FAKE)

    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_build_registry", build)
    return calls


# -- When it is built --------------------------------------------------------

def test_http_startup_builds_the_catalogue_before_the_app_starts(builds):
    order = []

    @asynccontextmanager
    async def app_lifespan(app):
        order.append(len(builds))
        yield

    async def run():
        async with server._starting_runtime(app_lifespan)(None):
            order.append("serving")

    anyio.run(run)
    assert order == [1, "serving"]


def test_http_startup_keeps_the_app_lifespan_state(builds):
    # Starlette copies a mapping a lifespan yields into every request's state.
    @asynccontextmanager
    async def app_lifespan(app):
        yield {"session_manager": "ready"}

    seen = []

    async def run():
        async with server._starting_runtime(app_lifespan)(None) as state:
            seen.append(state)

    anyio.run(run)
    assert seen == [{"session_manager": "ready"}]


def test_a_broken_catalogue_stops_the_http_app_starting(monkeypatch):
    def broken():
        raise CatalogueError("io: no module")

    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_build_registry", broken)
    started = []

    @asynccontextmanager
    async def app_lifespan(app):
        started.append(True)
        yield

    async def run():
        async with server._starting_runtime(app_lifespan)(None):
            pass

    with pytest.raises(CatalogueError, match="io"):
        anyio.run(run)
    assert started == []


def test_stdio_serves_without_building_the_catalogue(builds, monkeypatch):
    # The chatbot starts a stdio server per message; most never list operations.
    monkeypatch.delenv("DTCC_MCP_TRANSPORT", raising=False)
    seen = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **kw: seen.append(len(builds)))

    server.main()

    assert seen == [0]


def test_over_stdio_a_broken_catalogue_fails_the_first_call_naming_it(monkeypatch):
    def broken():
        raise CatalogueError("Failed to register IO functions: no module")

    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_build_registry", broken)

    _, result = anyio.run(server.mcp.call_tool, "list_operations", {})

    assert "Failed to register IO functions" in result["result"]


def test_tool_calls_after_startup_reuse_the_catalogue(builds):
    server.runtime.start()

    for _ in range(3):
        anyio.run(server.mcp.call_tool, "list_operations", {})

    assert len(builds) == 1


def test_two_first_users_in_process_share_one_build(builds, monkeypatch):
    # In-process callers (tests, scripts) skip startup and build on first use;
    # tools now run on worker threads, so two can arrive at once.
    instant = registry._build_registry

    def slow():
        time.sleep(0.05)  # widen the window two first callers could share
        return instant()

    monkeypatch.setattr(registry, "_build_registry", slow)
    threads = [threading.Thread(target=registry.get_registry) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1


# -- What counts as a failed build -------------------------------------------

@pytest.mark.parametrize("section, named", [
    (("builder", "general"), "builder functions"),
    (("builder", "pointcloud_filter"), "pointcloud filter functions"),
    (("builder", "pointcloud_convert"), "pointcloud convert functions"),
    (("builder", "raster_analysis"), "raster analyse functions"),
    (("builder", "raster_filter"), "raster filter functions"),
    (("builder", "raster_stats"), "raster stats functions"),
    (("builder", "raster_interpolation"), "raster interpolation functions"),
    (("builder", "meshing"), "meshing functions"),
    (("io", "general"), "IO functions"),
    (("reproject", "general"), "reproject functions"),
])
def test_a_core_section_that_fails_stops_the_build_naming_it(monkeypatch, section, named):
    real = registry._register_functions

    def one_breaks(reg, module, category, subcategory, *args, **kwargs):
        if (category, subcategory) == section:
            raise ImportError("dtcc_core is broken")
        return real(reg, module, category, subcategory, *args, **kwargs)

    monkeypatch.setattr(registry, "_register_functions", one_breaks)

    with pytest.raises(CatalogueError, match=f"{named}.*dtcc_core is broken"):
        registry._build_registry()


def test_core_datasets_failing_stops_the_build(monkeypatch):
    def broken(reg):
        raise RuntimeError("dataset registry unreadable")

    monkeypatch.setattr(registry, "_register_datasets", broken)

    with pytest.raises(CatalogueError, match="datasets.*unreadable"):
        registry._build_registry()


def test_a_core_without_its_dataset_registry_stops_the_build(monkeypatch):
    # Before #22 a missing dtcc_core.datasets.registry quietly gave no datasets.
    monkeypatch.setitem(sys.modules, "dtcc_core.datasets.registry", None)

    with pytest.raises(CatalogueError, match="Failed to register datasets"):
        registry._build_registry()


def test_a_name_core_lists_but_lacks_stops_the_build_naming_it(monkeypatch):
    import dtcc_core.io as io_mod
    monkeypatch.setattr(io_mod, "__all__", [*io_mod.__all__, "load_nothing"])

    with pytest.raises(CatalogueError, match="IO functions.*load_nothing"):
        registry._build_registry()


def _module(**attrs):
    return SimpleNamespace(__name__="dtcc_core.fake", **attrs)


def _op():
    """A fake Core operation."""


def test_a_name_in_a_modules_own_all_that_it_lacks_raises():
    # With no names passed, the module's __all__ is the list Core promises.
    module = _module(__all__=["op", "gone"], op=_op)

    with pytest.raises(AttributeError, match="dtcc_core.fake.*gone"):
        registry._register_functions({}, module, "builder")


def test_a_listed_name_that_is_not_callable_raises():
    module = _module(op=_op, VERSION="1.0")

    with pytest.raises(AttributeError, match="VERSION"):
        registry._register_functions({}, module, "builder", names=["op", "VERSION"])


def test_without_a_list_only_public_functions_register():
    # The dir() fallback cannot tell a missing name, so it only finds them.
    module = _module(op=_op, _private=_op, Klass=type("Klass", (), {}), VERSION="1.0")
    reg = {}

    registry._register_functions(reg, module, "builder", prefix="builder.x.")

    assert list(reg) == ["builder.x.op"]


def test_a_class_a_module_lists_is_skipped_not_raised():
    module = _module(op=_op, Klass=type("Klass", (), {}))
    reg = {}

    registry._register_functions(reg, module, "io", names=["op", "Klass"])

    assert list(reg) == ["io.op"]


def test_a_core_dataset_without_readable_options_stops_the_build_naming_it(monkeypatch):
    from dtcc_core.datasets.registry import list_datasets
    name, dataset = next(iter(list_datasets().items()))

    def unreadable():
        raise ValueError("schema broken")

    monkeypatch.setattr(dataset, "show_options", unreadable)

    with pytest.raises(CatalogueError, match=f"datasets.*{name}.*schema broken"):
        registry._build_registry()


def test_a_failed_build_is_not_kept_so_the_next_caller_builds_again(monkeypatch):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise CatalogueError("io: no module")
        return dict(FAKE)

    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_build_registry", flaky)

    with pytest.raises(CatalogueError):
        registry.get_registry()
    assert registry.get_registry() == FAKE
    assert len(attempts) == 2


@pytest.fixture
def broken_dtcc_sim(tmp_path, monkeypatch):
    """An installed dtcc_sim whose datasets module raises `sim_error`.

    A real dtcc_sim, if installed and already imported, is set aside so the
    fake is the one imported, and put back afterwards.
    """
    real = {n: m for n, m in sys.modules.items() if n == "dtcc_sim" or n.startswith("dtcc_sim.")}
    for name in real:
        del sys.modules[name]

    def install(sim_error: str):
        package = tmp_path / "dtcc_sim"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "datasets.py").write_text(sim_error)
        monkeypatch.syspath_prepend(str(tmp_path))
    yield install
    # monkeypatch cannot undo an import it never saw, so forget the fake here.
    for name in [n for n in sys.modules if n == "dtcc_sim" or n.startswith("dtcc_sim.")]:
        del sys.modules[name]
    sys.modules.update(real)


def test_without_dtcc_sim_installed_the_build_says_nothing(monkeypatch, caplog):
    class NotInstalled:  # import system hook: dtcc_sim can't be found
        def find_spec(self, name, path=None, target=None):
            if name == "dtcc_sim":
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    for name in [n for n in sys.modules if n == "dtcc_sim" or n.startswith("dtcc_sim.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NotInstalled(), *sys.meta_path])

    catalogue = registry._build_registry()

    assert any(name.startswith("datasets.") for name in catalogue)
    assert "dtcc_sim" not in caplog.text


@pytest.mark.parametrize("sim_error, shown", [
    ("import not_a_dependency_of_ours", "not_a_dependency_of_ours"),
    ("raise RuntimeError('sim config unreadable')", "sim config unreadable"),
])
def test_a_broken_dtcc_sim_is_a_warning(broken_dtcc_sim, caplog, sim_error, shown):
    broken_dtcc_sim(sim_error)

    catalogue = registry._build_registry()

    assert any(name.startswith("datasets.") for name in catalogue)
    assert "dtcc_sim" in caplog.text and shown in caplog.text


def test_a_dtcc_sim_without_its_datasets_module_is_a_warning(broken_dtcc_sim, tmp_path, caplog):
    broken_dtcc_sim("")
    (tmp_path / "dtcc_sim" / "datasets.py").unlink()

    catalogue = registry._build_registry()

    assert any(name.startswith("datasets.") for name in catalogue)
    assert "dtcc_sim.datasets" in caplog.text


# -- dtcc-sim's datasets join on demand ----------------------------------------
#
# Registered at startup, they were lost until a restart whenever dtcc-sim was
# down at boot, and startup waited on the network. Now a background thread asks
# a down service every 30s, and a catalogue read merges whatever has answered:
# no reader ever waits on dtcc-sim.

SIM = "http://sim:8000"
SIM_OP = "datasets.urban_heat_simulation"


@pytest.fixture
def sim(builds, monkeypatch):
    """dtcc-sim configured but down; set `sim.up = True` to bring it back.

    `sim.calls` records the thread that asked the network each time. The
    retrier's 30s pause waits for `sim.tick` instead; `sim.release`, when set
    to an Event, holds the network call until it is set.
    """
    registered: set[str] = set()
    s = SimpleNamespace(up=False, calls=[], registered=registered, sleeps=[],
                        entered=threading.Event(), tick=threading.Event(), release=None)

    def ensure():
        s.calls.append(threading.current_thread())
        s.entered.set()
        if s.release is not None:
            s.release.wait(5)
        if s.up:
            registered.add(SIM)

    def sleep(seconds):
        s.sleeps.append(seconds)
        s.tick.wait(5)
        s.tick.clear()

    def register_datasets(reg):
        if SIM in registered:
            reg[SIM_OP] = OperationInfo(name=SIM_OP, category="datasets")

    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM])
    monkeypatch.setattr(runner, "_REGISTERED_REMOTE_SERVICES", registered)
    monkeypatch.setattr(runner, "_ensure_remote_services_registered", ensure)
    monkeypatch.setattr(registry, "_register_datasets", register_datasets)
    monkeypatch.setattr(registry, "_sleep", sleep)
    yield s
    # Let a retrier still running finish before the patches are undone.
    s.up = True
    if s.release is not None:
        s.release.set()
    s.tick.set()
    if registry._retrier is not None:
        registry._retrier.join(5)


def _retrier_finished():
    assert registry._retrier is not None
    registry._retrier.join(5)
    assert not registry._retrier.is_alive()


def test_building_the_catalogue_never_calls_dtcc_sim(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM])
    monkeypatch.setattr(runner, "_ensure_remote_services_registered",
                        lambda: calls.append(1))

    catalogue = registry._build_registry()

    assert calls == []
    assert any(name.startswith("datasets.") for name in catalogue)


def test_http_startup_never_waits_on_dtcc_sim(sim):
    @asynccontextmanager
    async def app_lifespan(app):
        yield

    async def run():
        async with server._starting_runtime(app_lifespan)(None):
            pass

    sim.release = threading.Event()  # dtcc-sim never answers
    started = time.monotonic()
    anyio.run(run)

    assert time.monotonic() - started < 1
    assert registry._REGISTRY == FAKE
    assert sim.entered.wait(5)  # but it was asked, in the background, at startup
    assert threading.current_thread() not in sim.calls


def test_building_the_catalogue_starts_asking_dtcc_sim(sim):
    sim.up = True

    registry.build()
    _retrier_finished()

    assert SIM_OP in registry.get_registry()


def test_a_pending_dtcc_sim_dataset_is_reported_as_not_ready_yet(sim):
    sim.release = threading.Event()  # dtcc-sim hasn't answered

    with pytest.raises(KeyError, match="dtcc-sim hasn't answered yet") as raised:
        registry.get_operation(SIM_OP)
    assert "list_operations()" in str(raised.value)  # a typo isn't only waiting
    assert SIM not in str(raised.value)  # the model can't use the service URL


def test_a_mistyped_core_operation_is_not_blamed_on_a_pending_dtcc_sim(sim):
    sim.release = threading.Event()  # dtcc-sim hasn't answered

    with pytest.raises(KeyError, match="not found") as raised:
        registry.get_operation("builder.no_such_function")
    assert "hasn't answered" not in str(raised.value)


def test_an_unknown_operation_is_still_not_found_when_dtcc_sim_is_up(sim):
    sim.registered.add(SIM)

    with pytest.raises(KeyError, match="not found") as raised:
        registry.get_operation("datasets.no_such_simulation")
    assert "hasn't answered" not in str(raised.value)


def test_a_reader_never_waits_on_dtcc_sim_even_when_it_hangs(sim):
    sim.release = threading.Event()  # dtcc-sim never answers

    started = time.monotonic()
    catalogues = [registry.get_registry() for _ in range(3)]
    waited = time.monotonic() - started

    assert waited < 1
    assert catalogues == [FAKE] * 3
    assert sim.entered.wait(5)  # asked, on another thread
    assert threading.current_thread() not in sim.calls
    assert len(sim.calls) == 1  # one retrier, however many readers


def test_a_down_dtcc_sim_is_asked_every_30s_until_it_answers(sim, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="dtcc_agent.registry")
    def sleep(seconds):
        sim.sleeps.append(seconds)
        if len(sim.sleeps) == 2:
            sim.up = True

    monkeypatch.setattr(registry, "_sleep", sleep)

    registry._retry_remote_services()

    assert sim.sleeps == [30, 30]
    assert len(sim.calls) == 3
    unreachable = [r for r in caplog.records if "unreachable" in r.getMessage()]
    assert [r.levelname for r in unreachable] == ["WARNING", "DEBUG"]  # said once
    assert any(r.levelname == "INFO" and "answered" in r.getMessage() for r in caplog.records)


def test_a_failing_dtcc_sim_call_is_a_warning_and_is_asked_again(sim, monkeypatch, caplog):
    attempts = []

    def refused_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("sim:8000 refused")
        sim.registered.add(SIM)

    monkeypatch.setattr(runner, "_ensure_remote_services_registered", refused_once)
    monkeypatch.setattr(registry, "_sleep", sim.sleeps.append)

    registry._retry_remote_services()

    assert "sim:8000 refused" in caplog.text
    assert len(attempts) == 2 and sim.sleeps == [30]


def test_dtcc_sim_datasets_appear_once_it_is_back(sim):
    assert SIM_OP not in registry.get_registry()
    assert sim.entered.wait(5)

    sim.up = True
    sim.tick.set()  # the 30s pause ends
    _retrier_finished()

    assert SIM_OP in registry.get_registry()


def test_a_service_the_runner_reconnected_is_merged_without_a_network_call(sim):
    sim.registered.add(SIM)  # as list_simulations does when dtcc-sim is back

    assert SIM_OP in registry.get_registry()
    assert sim.calls == []
    assert registry._retrier is None


def test_a_merge_never_changes_a_catalogue_someone_is_reading(sim):
    before = registry.get_registry()
    held = dict(before)

    sim.registered.add(SIM)
    after = registry.get_registry()

    assert before == held
    assert SIM_OP in after


def test_a_merge_in_progress_does_not_hold_up_other_readers(sim):
    registry.get_registry()
    sim.registered.add(SIM)

    with registry._MERGE_LOCK:  # another reader is merging
        started = time.monotonic()
        catalogue = registry.get_registry()
        waited = time.monotonic() - started

    assert waited < 1
    assert catalogue == FAKE
    assert SIM_OP in registry.get_registry()


def test_a_failing_merge_is_a_warning_and_is_tried_again(sim, monkeypatch, caplog):
    working = registry._register_datasets

    def broken(reg):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(registry, "_register_datasets", broken)
    sim.registered.add(SIM)

    assert registry.get_registry() == FAKE
    assert "dictionary changed size" in caplog.text

    monkeypatch.setattr(registry, "_register_datasets", working)
    assert SIM_OP in registry.get_registry()


@pytest.fixture
def optional_dataset():
    """Register a non-Core dataset in Core's shared registry, as dtcc_sim or
    runner's remote discovery would, and remove it afterwards."""
    from dtcc_core.datasets import registry as core_datasets
    names = []

    def add(name, instance):
        core_datasets.register(name, instance)
        names.append(name)

    yield add
    for name in names:
        core_datasets.unregister(name)


def test_a_dtcc_sim_dataset_with_unreadable_options_is_skipped(optional_dataset, caplog):
    from dtcc_core.datasets.dataset import DatasetDescriptor

    class BrokenSimDataset(DatasetDescriptor, register=False):
        name = "broken_sim"
        description = "from dtcc_sim"

        def build(self, args):
            raise NotImplementedError

        def show_options(self):
            raise ValueError("sim schema broken")

    optional_dataset("broken_sim", BrokenSimDataset())

    catalogue = registry._build_registry()

    assert "datasets.broken_sim" not in catalogue
    assert "datasets.point_cloud" in catalogue
    assert "broken_sim" in caplog.text and "sim schema broken" in caplog.text


def test_a_remote_dataset_with_a_malformed_description_is_skipped(optional_dataset, caplog):
    from dtcc_core.datasets.remote import RemoteDatasetDescriptor

    optional_dataset("odd_sim", RemoteDatasetDescriptor(
        name="odd_sim", description=["not", "text"], args_schema={"properties": {}},
        base_url=SIM, result_kind="file", supported_formats=["bin"],
        source_service="dtcc-sim"))

    catalogue = registry._build_registry()

    assert "datasets.odd_sim" not in catalogue
    assert "odd_sim" in caplog.text


def test_a_remote_dataset_without_a_schema_is_skipped(optional_dataset, caplog):
    from dtcc_core.datasets.remote import RemoteDatasetDescriptor

    optional_dataset("no_schema_sim", RemoteDatasetDescriptor(
        name="no_schema_sim", description="remote", args_schema=None,
        base_url=SIM, result_kind="file", supported_formats=["bin"],
        source_service="dtcc-sim"))

    catalogue = registry._build_registry()

    assert "datasets.no_schema_sim" not in catalogue
    assert "no_schema_sim" in caplog.text


def test_without_dtcc_sim_configured_the_catalogue_never_asks(sim, monkeypatch):
    monkeypatch.setattr(runner, "_remote_services", lambda: [])

    registry.get_registry()

    assert sim.calls == []
    assert registry._retrier is None


def test_a_second_dtcc_sim_service_joins_after_the_first(sim, monkeypatch):
    other, other_op = "http://sim2:8000", "datasets.wind_simulation"
    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM, other])

    def register_datasets(reg):
        if SIM in sim.registered:
            reg[SIM_OP] = OperationInfo(name=SIM_OP, category="datasets")
        if other in sim.registered:
            reg[other_op] = OperationInfo(name=other_op, category="datasets")

    monkeypatch.setattr(registry, "_register_datasets", register_datasets)

    sim.registered.add(SIM)
    first = registry.get_registry()
    assert SIM_OP in first and other_op not in first

    sim.registered.add(other)  # the second service answers later
    second = registry.get_registry()
    assert SIM_OP in second and other_op in second
    assert registry._MERGED_REMOTE_SERVICES == {SIM, other}


# -- A real HTTP server ------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_an_http_server_builds_its_catalogue_once_for_many_sessions(tmp_path):
    port = _free_port()
    log = tmp_path / "stderr.log"
    env = {**os.environ, "DTCC_MCP_TRANSPORT": "http", "DTCC_MCP_PORT": str(port)}
    url = f"http://127.0.0.1:{port}/mcp"

    async def list_ops(session_id):
        async with (
            httpx.AsyncClient(headers={SESSION_HEADER: session_id}) as http,
            streamable_http_client(url, http_client=http) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("list_operations", {})
            assert not result.isError

    with open(log, "w") as err:
        proc = subprocess.Popen([sys.executable, "-m", "dtcc_agent"], env=env,
                                stdout=subprocess.DEVNULL, stderr=err)
        try:
            deadline = time.monotonic() + 60
            while True:
                assert proc.poll() is None, log.read_text()[-2000:]
                try:
                    httpx.get(url, timeout=1)
                    break
                except httpx.TransportError:
                    assert time.monotonic() < deadline, "server never started listening"
                    time.sleep(0.2)
            listening = log.read_text()
            for session_id in ("a", "b", "c"):
                anyio.run(list_ops, session_id)
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    built = [line for line in log.read_text().splitlines() if "catalogue built" in line]
    assert len(built) == 1, built
    assert "catalogue built" in listening  # before the port answered
    assert f"{len(registry.get_registry())} operations" in built[0]


def test_an_http_server_with_a_broken_catalogue_exits_without_listening():
    # uvicorn must treat the failed lifespan as fatal, not serve without it.
    port = _free_port()
    env = {**os.environ, "DTCC_MCP_TRANSPORT": "http", "DTCC_MCP_PORT": str(port)}
    script = (
        "import dtcc_agent.registry as r, dtcc_agent.server as s\n"
        "def broken(): raise r.CatalogueError('io: no module')\n"
        "r._build_registry = broken\n"
        "s.main()\n"
    )

    out = subprocess.run([sys.executable, "-c", script], env=env,
                         capture_output=True, text=True, timeout=60)

    assert out.returncode != 0
    assert "CatalogueError: io: no module" in out.stderr, out.stderr[-2000:]
    assert "catalogue built" not in out.stderr



def test_a_remote_dataset_cannot_replace_a_core_dataset_of_the_same_name(
        monkeypatch, optional_dataset, caplog):
    """Core's register() overwrites by name, so a dtcc-sim service advertising
    `point_cloud` would swap Core's download for its own everywhere Core's
    registry is read: the catalogue, and get_buildings through runner (#45)."""
    from dtcc_core import datasets
    from dtcc_core.datasets.registry import get_dataset, register
    from dtcc_core.datasets.remote import RemoteDatasetDescriptor

    def remote(name):
        return RemoteDatasetDescriptor(
            name=name, description="remote", args_schema={"properties": {}},
            base_url=SIM, result_kind="file", supported_formats=["bin"],
            source_service="dtcc-sim")

    def register_remote_service(url):  # Core's, minus the network
        register("point_cloud", remote("point_cloud"))
        optional_dataset("flood_sim", remote("flood_sim"))
        return ["point_cloud", "flood_sim"]

    core_point_cloud = get_dataset("point_cloud")
    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM])
    monkeypatch.setattr(runner, "_REGISTERED_REMOTE_SERVICES", set())
    monkeypatch.setattr(datasets, "register_remote_service", register_remote_service)
    try:
        runner._ensure_remote_services_registered()
        catalogue = registry._build_registry()

        assert get_dataset("point_cloud") is core_point_cloud
        assert catalogue["datasets.point_cloud"]._callable is core_point_cloud
        assert "datasets.flood_sim" in catalogue
        assert SIM in runner._REGISTERED_REMOTE_SERVICES
        assert "point_cloud" in caplog.text and SIM in caplog.text
    finally:
        register("point_cloud", core_point_cloud)  # even if the fix is missing


def _remote(name):
    from dtcc_core.datasets.remote import RemoteDatasetDescriptor
    return RemoteDatasetDescriptor(
        name=name, description="remote", args_schema={"properties": {}},
        base_url=SIM, result_kind="file", supported_formats=["bin"],
        source_service="dtcc-sim")


def test_a_half_broken_discovery_leaves_nothing_behind_however_often_it_is_retried(
        monkeypatch, optional_dataset, caplog):
    """Core registers entries one at a time and returns [] when a later one is
    malformed, so every retry left another copy of the valid ones (#44)."""
    from dtcc_core import datasets
    from dtcc_core.datasets import registry as core_datasets

    def half_broken(url):  # Core's, minus the network: one valid entry, then a bad one
        core_datasets.register("flood_sim", _remote("flood_sim"))
        return []

    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM])
    monkeypatch.setattr(runner, "_REGISTERED_REMOTE_SERVICES", set())
    monkeypatch.setattr(datasets, "register_remote_service", half_broken)
    optional_dataset("flood_sim", _remote("flood_sim"))  # removed afterwards if it leaks
    core_datasets.unregister("flood_sim")

    for _ in range(8):  # the retrier asks every 30 s
        runner._ensure_remote_services_registered()

    assert "flood_sim" not in core_datasets.list_datasets()
    assert not [d for d in core_datasets._datasets_registry
                if getattr(d, "name", None) == "flood_sim"]
    assert SIM not in runner._REGISTERED_REMOTE_SERVICES
    assert "flood_sim" in caplog.text and SIM in caplog.text


def test_a_half_broken_discovery_puts_back_a_dataset_it_replaced(
        monkeypatch, optional_dataset):
    from dtcc_core import datasets
    from dtcc_core.datasets import registry as core_datasets

    other = _remote("heat_sim")  # another service's, registered earlier
    optional_dataset("heat_sim", other)

    def half_broken(url):
        core_datasets.register("heat_sim", _remote("heat_sim"))
        return []

    monkeypatch.setattr(runner, "_remote_services", lambda: [SIM])
    monkeypatch.setattr(runner, "_REGISTERED_REMOTE_SERVICES", set())
    monkeypatch.setattr(datasets, "register_remote_service", half_broken)

    runner._ensure_remote_services_registered()

    assert core_datasets.get_dataset("heat_sim") is other
    assert [d for d in core_datasets._datasets_registry
            if getattr(d, "name", None) == "heat_sim"] == [other]

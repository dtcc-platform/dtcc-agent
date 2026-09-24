"""Characterisation tests for the current 22-tool MCP surface.

These describe what `server.py` does **today**, including the parts that are
wrong. They are a baseline, not an endorsement. M1 changes transport, session
state and references all at once; without a written description of the current
behaviour there is nothing to diff that change against.

Where a test documents behaviour the rebuild plan intends to change, it says so
and names the task. When M1 lands, these tests are expected to fail, and the
failure is the signal to update them deliberately rather than discover the
change by accident.

`server.py` is 1190 lines and had no direct coverage: the rest of the suite
exercises the modules behind the tools but never imports the server, which is
how the suite stayed green while `python -m dtcc_agent` could not start at all.
"""

import asyncio
import json

import pytest

import dtcc_agent.server as server


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def clean_stores(monkeypatch):
    """Give the test a fresh Session.

    In-process calls (like these) use the process-local Session, so a test
    that plants entries would otherwise leak into every later test.
    """
    monkeypatch.setattr(server, "_local_session", server._Session())


def _parse(payload: str) -> dict:
    """Tools return pretty-printed JSON strings, not objects."""
    return json.loads(payload)


# -- The tool surface --------------------------------------------------------

EXPECTED_TOOLS = {
    "compare_scenarios",
    "delete_object",
    "describe_operation",
    "export_object",
    "geocode",
    "get_buildings",
    "get_field_names",
    "get_run_summary",
    "get_simulation_schema",
    "inspect_object",
    "list_objects",
    "list_operations",
    "list_past_runs",
    "list_simulations",
    "load_geojson",
    "object_to_text",
    "query_geojson",
    "render_object",
    "run_operation",
    "run_simulation",
    "spatial_query",
    "summarize_geojson_property",
}


def _tools():
    return asyncio.run(server.mcp.list_tools())


def test_tool_count_is_22():
    """The surface is 22 tools. M1 explicitly does not change which exist."""
    assert len(_tools()) == 22


def test_tool_names_are_exactly_the_expected_set():
    assert {t.name for t in _tools()} == EXPECTED_TOOLS


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_every_tool_has_a_description(name):
    """The description is the LLM's only documentation. An empty one is a bug."""
    tool = next(t for t in _tools() if t.name == name)
    assert tool.description and tool.description.strip(), f"{name} has no description"


@pytest.mark.parametrize(
    "name,required",
    [
        ("compare_scenarios", {"simulation_name", "bounds",
                               "scenario_a_parameters", "scenario_b_parameters"}),
        ("delete_object", {"object_id"}),
        ("describe_operation", {"name"}),
        ("export_object", {"object_id", "format"}),
        ("geocode", {"place_name"}),
        ("get_buildings", {"bounds"}),
        ("get_field_names", {"object_id"}),
        ("get_run_summary", {"run_id"}),
        ("get_simulation_schema", {"simulation_name"}),
        ("inspect_object", {"object_id"}),
        ("list_objects", set()),
        ("list_operations", set()),
        ("list_past_runs", set()),
        ("list_simulations", set()),
        ("load_geojson", {"file_path"}),
        ("object_to_text", {"object_id"}),
        ("query_geojson", {"object_id", "property_name", "operator", "value"}),
        ("render_object", {"object_id"}),
        ("run_operation", {"name"}),
        ("run_simulation", {"simulation_name", "bounds"}),
        ("spatial_query", {"object_id", "query_type", "params"}),
        ("summarize_geojson_property", {"object_id", "property_name"}),
    ],
)
def test_required_parameters(name, required):
    """The required set is the contract an MCP client codes against."""
    tool = next(t for t in _tools() if t.name == name)
    assert set(tool.inputSchema.get("required", [])) == required


# -- Error reporting ---------------------------------------------------------

OBJECT_ID_TOOLS = [
    "inspect_object",
    "get_field_names",
    "object_to_text",
    "delete_object",
]


@pytest.mark.parametrize("tool_name", OBJECT_ID_TOOLS)
def test_unknown_object_id_returns_an_error_payload_not_an_exception(
    tool_name, clean_stores
):
    """Tools report failure as a JSON body, never by raising.

    This matters for the rebuild: an MCP client sees a successful call whose
    body happens to describe an error, so nothing upstream can distinguish a
    miss from a result without parsing.
    """
    result = getattr(server, tool_name)("does-not-exist")
    assert "error" in _parse(result)


def test_unknown_run_id_returns_an_error_payload(clean_stores):
    assert "error" in _parse(server.get_run_summary("does-not-exist"))


# -- Reference confusion (baseline for ADR-0010 / M1 task T9) ----------------

def _plant_run() -> str:
    """Create a run the way the server itself does."""
    return server._store_result(
        name="fake_simulation",
        bounds=[0.0, 0.0, 1.0, 1.0],
        parameters={},
        result={"not": "a field-bearing result"},
    )


def test_store_result_creates_two_entries_with_unrelated_ids(clean_stores):
    """One simulation result is stored twice, under two different ids.

    `_store_result` writes to the Session's `results` under a run_id and separately to the
    ObjectStore under its own obj_id. The only link is the ObjectStore's
    `label` field, which is written and displayed but never queried.

    ADR-0010 / task T9 changes this. Characterised here so the change is
    visible when it happens.
    """
    run_id = _plant_run()

    assert run_id in server._session().results
    entries = server._session().objects.list()
    assert len(entries) == 1
    # The ObjectStore keys its listing on "id"; the run lives under "run_id"
    # elsewhere. Same concept, two spellings, no shared vocabulary.
    assert entries[0]["id"] != run_id, "ids are expected to be unrelated today"
    # The only thing tying the two records together:
    assert entries[0]["label"] == run_id


def test_run_and_object_ids_are_shape_indistinguishable(clean_stores):
    """Both id spaces are 8 hex characters, so no consumer can tell them apart.

    `_store_result` uses `str(uuid4())[:8]`; the ObjectStore uses
    `uuid4().hex[:8]`. Different derivations, identical shape.
    """
    run_id = _plant_run()
    object_id = server._session().objects.list()[0]["id"]

    assert len(run_id) == len(object_id) == 8
    assert all(c in "0123456789abcdef" for c in run_id)
    assert all(c in "0123456789abcdef" for c in object_id)


def test_run_id_passed_to_an_object_tool_is_missed_not_refused(clean_stores):
    """Today a mistyped reference reports 'not found', not 'wrong kind'.

    That wording sends the caller to `list_objects()`, which will never show
    the run they are holding. T9 makes this a refusal instead.
    """
    run_id = _plant_run()

    payload = _parse(server.inspect_object(run_id))

    assert "error" in payload
    assert "not found" in payload["error"].lower()
    # The diagnosis is absent: nothing says the id names a run.
    assert "run" not in payload["error"].lower()


def test_object_id_passed_to_a_run_tool_is_missed_not_refused(clean_stores):
    """The mirror case, with the same weakness."""
    _plant_run()
    object_id = server._session().objects.list()[0]["id"]

    payload = _parse(server.get_run_summary(object_id))

    assert "error" in payload
    assert "not found" in payload["error"].lower()
    assert "object" not in payload["error"].lower()


def test_get_run_summary_rejects_a_result_without_extractable_fields(clean_stores):
    """A run whose result has no `.x.array` reports that, rather than raising."""
    run_id = _plant_run()

    payload = _parse(server.get_run_summary(run_id))

    assert "error" in payload
    assert "extractable" in payload["error"].lower()


# -- Listing tools -----------------------------------------------------------

def test_list_objects_on_an_empty_store(clean_stores):
    """Empty is reported as data, not as an error."""
    payload = _parse(server.list_objects())
    assert payload == {"num_objects": 0, "total_memory_mb": 0.0, "objects": []}


def test_list_past_runs_on_an_empty_store(clean_stores):
    """Note the asymmetry with list_objects: a bare list, not a wrapper."""
    assert _parse(server.list_past_runs()) == []


def test_listing_tools_disagree_about_their_envelope(clean_stores):
    """`list_objects` wraps its rows; `list_past_runs` returns them bare.

    Two listing tools on the same surface with different envelopes means a
    client cannot handle them with one code path. Characterised rather than
    fixed: M0 changes no behaviour.
    """
    assert isinstance(_parse(server.list_objects()), dict)
    assert isinstance(_parse(server.list_past_runs()), list)


def test_list_objects_reflects_a_stored_object(clean_stores):
    server._session().objects.store([1, 2, 3], source_op="probe", label="planted")
    payload = _parse(server.list_objects())
    blob = json.dumps(payload)
    assert "planted" in blob


# -- Catalogue ---------------------------------------------------------------

def test_operation_catalogue_is_populated():
    """A non-empty catalogue is the thing an absent dtcc-core silently broke.

    Asserted as a floor rather than an exact count: the exact number tracks the
    pinned Core revision, and the contract workflow prints it per run.
    """
    payload = _parse(server.list_operations())
    blob = json.dumps(payload)
    assert len(blob) > 100, "catalogue looks empty"

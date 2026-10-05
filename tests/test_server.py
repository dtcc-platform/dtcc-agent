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
        ("delete_object", {"object_ref"}),
        ("describe_operation", {"name"}),
        ("export_object", {"object_ref", "format"}),
        ("geocode", {"place_name"}),
        ("get_buildings", {"bounds"}),
        ("get_field_names", {"object_ref"}),
        ("get_run_summary", {"run_ref"}),
        ("get_simulation_schema", {"simulation_name"}),
        ("inspect_object", {"object_ref"}),
        ("list_objects", set()),
        ("list_operations", set()),
        ("list_past_runs", set()),
        ("list_simulations", set()),
        ("load_geojson", {"name"}),
        ("object_to_text", {"object_ref"}),
        ("query_geojson", {"object_ref", "property_name", "operator", "value"}),
        ("render_object", {"object_ref"}),
        ("run_operation", {"name"}),
        ("run_simulation", {"simulation_name", "bounds"}),
        ("spatial_query", {"object_ref", "query_type", "params"}),
        ("summarize_geojson_property", {"object_ref", "property_name"}),
    ],
)
def test_required_parameters(name, required):
    """The required set is the contract an MCP client codes against."""
    tool = next(t for t in _tools() if t.name == name)
    assert set(tool.input_schema.get("required", [])) == required


# -- Error reporting ---------------------------------------------------------

OBJECT_REF_TOOLS = [
    "inspect_object",
    "get_field_names",
    "object_to_text",
    "delete_object",
]


@pytest.mark.parametrize("tool_name", OBJECT_REF_TOOLS)
def test_unknown_object_ref_returns_an_error_payload_not_an_exception(
    tool_name, clean_stores
):
    """Tools report failure as a JSON body, never by raising.

    This matters for the rebuild: an MCP client sees a successful call whose
    body happens to describe an error, so nothing upstream can distinguish a
    miss from a result without parsing.
    """
    result = getattr(server, tool_name)("does-not-exist")
    assert "error" in _parse(result)


@pytest.mark.parametrize("call", [
    lambda ref: server.query_geojson(ref, "height", ">", 10),
    lambda ref: server.summarize_geojson_property(ref, "height"),
], ids=["query_geojson", "summarize_geojson_property"])
def test_the_geojson_tools_report_an_unknown_object_rather_than_raise(call, clean_stores):
    """They checked for None from ObjectStore.get, which raises KeyError instead."""
    assert "not found" in _parse(call("obj_00000000"))["error"]


def test_unknown_run_ref_returns_an_error_payload(clean_stores):
    assert "error" in _parse(server.get_run_summary("does-not-exist"))


# -- Typed references (ADR-0010, T9 #26) --------------------------------------
# These replace the M0 characterisation tests that pinned the old behaviour:
# unrelated ids, identical id shapes, and a reference of the wrong kind missed
# ("not found") rather than refused.

class _Field:
    """A result the way dolfinx returns one: values under .x.array."""
    def __init__(self, values):
        self.x = type("X", (), {"array": values})()


def _plant_run(result=None) -> str:
    """Create a run the way the server itself does."""
    return server._store_result(
        name="fake_simulation",
        bounds=[0.0, 0.0, 1.0, 1.0],
        parameters={},
        result={"not": "a field-bearing result"} if result is None else result,
    )


def test_references_say_which_kind_they_are(clean_stores):
    run_ref = _plant_run()
    object_ref = server._session().objects.list()[0]["object_ref"]

    assert run_ref.startswith("run_") and object_ref.startswith("obj_")
    assert len(run_ref) == len(object_ref) == 12


def test_a_run_records_the_object_it_yielded_and_holds_no_copy(clean_stores):
    """U6: the Object owns the result; the Run keeps what was run and its reference."""
    import numpy as np

    run_ref = _plant_run(_Field(np.array([1.0, 2.0, 3.0])))
    run = server._session().results[run_ref]
    listed = server._session().objects.list()

    assert [o["object_ref"] for o in listed] == [run["object_ref"]]
    assert "result" not in run
    summary = _parse(server.get_run_summary(run_ref))
    assert summary["object_ref"] == run["object_ref"]
    assert summary["summary"]["mean"] == 2.0


@pytest.mark.parametrize("tool_name", OBJECT_REF_TOOLS)
def test_a_run_reference_passed_to_an_object_tool_is_refused_as_the_wrong_kind(
        tool_name, clean_stores):
    run_ref = _plant_run()

    error = _parse(getattr(server, tool_name)(run_ref))["error"]

    assert "run reference" in error.lower() and "get_run_summary" in error
    assert "not found" not in error.lower()


def test_an_object_reference_passed_to_a_run_tool_is_refused_as_the_wrong_kind(clean_stores):
    _plant_run()
    object_ref = server._session().objects.list()[0]["object_ref"]

    error = _parse(server.get_run_summary(object_ref))["error"]

    assert "object reference" in error.lower() and "inspect_object" in error
    assert "not found" not in error.lower()


def test_a_run_reference_passed_to_an_operation_is_refused(clean_stores):
    run_ref = _plant_run()

    error = _parse(server.run_operation("builder.raster.slope_aspect",
                                        {"dem": run_ref}))["error"]

    assert "run reference" in error.lower()


def test_a_run_whose_object_was_evicted_says_so(clean_stores):
    run_ref = _plant_run()
    server._session().objects.delete(server._session().results[run_ref]["object_ref"])

    error = _parse(server.get_run_summary(run_ref))["error"]

    assert "no longer in memory" in error and "run_simulation" in error
    listed = _parse(server.list_past_runs())
    assert listed[0]["run_ref"] == run_ref and listed[0]["evicted"] is True


def test_get_run_summary_rejects_a_result_without_extractable_fields(clean_stores):
    """A run whose result has no `.x.array` reports that, rather than raising."""
    run_ref = _plant_run()

    payload = _parse(server.get_run_summary(run_ref))

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

"""No path typed in chat reaches the filesystem through run_operation (U1, #10)."""

from unittest.mock import patch

import pytest

from dtcc_agent.dispatcher import run_operation
from dtcc_agent.object_store import ObjectStore
from dtcc_agent.registry import get_registry

# Every operation in the pinned Core that takes a file or directory path.
# A Core upgrade that adds one fails here, so it is reviewed, not reachable.
PATH_OPERATIONS = {
    "builder.build_city_volume_mesh": {"tetgen_debug_output_dir", "tetgen_quality_failure_output_dir"},
    "datasets.city_volume_mesh": {"tetgen_debug_output_dir", "tetgen_quality_failure_output_dir"},
    "io.load_3dbag": {"path"},
    "io.load_city": {"path"},
    "io.load_cityjson": {"cityjson_path"},
    "io.load_footprints": {"filename"},
    "io.load_mesh": {"path"},
    "io.load_model": {"path"},
    "io.load_pointcloud": {"path"},
    "io.load_raster": {"path"},
    "io.load_roadnetwork": {"filename"},
    "io.load_volume_mesh": {"path"},
    "io.save_footprints": {"filename"},
    "io.save_mesh": {"path"},
    "io.save_model": {"path"},
    "io.save_pointcloud": {"outfile"},
    "io.save_raster": {"path"},
    "io.save_volume_mesh": {"path"},
}


def test_the_path_parameters_in_core_are_exactly_the_known_ones():
    found = {
        name: {p.name for p in op.params if p.is_path}
        for name, op in get_registry().items()
    }
    assert {k: v for k, v in found.items() if v} == PATH_OPERATIONS


@pytest.mark.parametrize("name,params", [
    ("io.save_mesh", {"path": "/etc/cron.d/x"}),
    ("io.load_pointcloud", {"path": "/etc/passwd"}),
    ("io.load_mesh", {}),  # a required path is refused even when left out
    ("datasets.city_volume_mesh",
     {"bounds": [0, 0, 10, 10], "tetgen_debug_output_dir": "/tmp/x"}),
])
def test_a_path_argument_is_refused_before_core_is_called(name, params):
    op = get_registry()[name]
    with patch.object(op, "_callable") as core:
        result = run_operation(name, params, store=ObjectStore())
    assert result["error"].startswith(f"Refused: {name} takes a file path")
    core.assert_not_called()


def test_an_optional_path_left_unset_does_not_refuse_the_operation():
    store = ObjectStore()
    city = store.store(object(), source_op="test")
    op = get_registry()["builder.build_city_volume_mesh"]
    with patch.object(op, "_callable", return_value=None) as core:
        run_operation("builder.build_city_volume_mesh", {"city": city}, store=store)
    core.assert_called_once()

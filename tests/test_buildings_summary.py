"""What get_buildings reports about the buildings of an area."""

import json

import numpy as np
import pytest

import dtcc_agent.runner as runner
import dtcc_agent.server as server
from dtcc_agent.disk_cache import DiskCache


def _building(x, y, *, estimated=None, measured=None, side=10):
    from dtcc_core.model import Building, GeometryType, Surface

    b = Building()
    square = [[x, y, 0], [x + side, y, 0], [x + side, y + side, 0], [x, y + side, 0]]
    b.add_geometry(Surface(vertices=np.array(square, float)), GeometryType.LOD0)
    if estimated is not None:
        b.estimated_height = estimated
    if measured is not None:
        b.measured_height = measured
    return b


def _summary(buildings, max_buildings=100):
    return runner.summarize_buildings(
        buildings, bounds=[0, 0, 1000, 1000], source="LM", max_buildings=max_buildings,
    )


def test_heights_come_from_cores_estimate_as_a_download_leaves_them():
    """LM and OSM downloads set estimated_height and leave measured_height
    (Building.height) empty; the summary reported every height as 0."""
    summary = _summary([_building(0, 0, estimated=12.34), _building(20, 0, estimated=30.0)])

    assert [d["height_m"] for d in summary["buildings"]] == [12.3, 30.0]
    assert summary["height_stats"]["max_m"] == 30.0
    assert summary["height_stats"]["min_m"] == 12.3


def test_a_measured_height_is_used_when_there_is_no_estimate():
    """Core's own precedence: the estimate if there is one, else the measurement."""
    summary = _summary([_building(0, 0, measured=8.0), _building(20, 0, estimated=9.0, measured=4.0)])

    assert [d["height_m"] for d in summary["buildings"]] == [8.0, 9.0]


def test_height_stats_are_empty_rather_than_zero_when_no_building_has_a_height():
    summary = _summary([_building(0, 0)])

    assert summary["height_stats"] == {"min_m": None, "max_m": None, "mean_m": None, "median_m": None}


def test_total_footprint_area_covers_every_building_not_only_the_listed_ones():
    summary = _summary([_building(0, 0), _building(20, 0), _building(40, 0)], max_buildings=1)

    assert len(summary["buildings"]) == 1
    assert summary["total_footprint_area_m2"] == 300.0


@pytest.mark.parametrize("bounds", [
    [50, 50, 10, 10],               # inverted
    [20, 20, 20, 20],               # zero area
    [0, 0, 100],                    # too short
    [0, 0, float("nan"), 100],      # not finite
    ["a", 0, 100, 100],             # not a number
])
def test_get_buildings_refuses_bounds_that_describe_no_area(monkeypatch, tmp_path, bounds):
    downloads = []
    monkeypatch.setattr(runner, "fetch_buildings", lambda **kw: downloads.append(kw) or [])
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))

    result = json.loads(server.get_buildings(bounds=bounds))

    assert result["error"].startswith("Invalid bounds")
    assert downloads == []


def test_run_simulation_refuses_inverted_bounds(monkeypatch):
    runs = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: runs.append(a))

    result = json.loads(server.run_simulation("urban_heat_simulation", [50, 50, 10, 10]))

    assert result["error"].startswith("Invalid bounds")
    assert runs == []


def test_run_operation_refuses_inverted_bounds():
    from dtcc_agent.dispatcher import run_operation
    from dtcc_agent.object_store import ObjectStore

    result = run_operation("datasets.buildings", {"bounds": [50, 50, 10, 10]}, store=ObjectStore())

    assert result["error"].startswith("Invalid bounds")


def test_run_operation_summarises_a_download_with_the_same_heights():
    """run_operation("datasets.buildings") is summarised by the serializers;
    it read Building.height too, so it also reported no heights."""
    from dtcc_core.datasets.buildings import BuildingCollection

    from dtcc_agent.serializers import serialize

    summary = serialize(BuildingCollection([_building(0, 0, estimated=12.0)]))

    assert summary["type"] == "BuildingCollection"
    assert summary["count"] == 1
    assert summary["height_stats"]["max"] == 12.0


@pytest.mark.parametrize("bounds", [None, "a1b2c3d4"])
def test_run_operation_passes_bounds_that_are_not_a_literal_box(monkeypatch, bounds):
    """Some operations default bounds to None or take a stored Bounds id."""
    from unittest.mock import MagicMock

    import dtcc_agent.dispatcher as dispatcher
    from dtcc_agent.object_store import ObjectStore

    op = MagicMock(category="builder", _callable=lambda **kw: None)
    op.name = "builder.build_terrain_raster"
    op.params = []
    monkeypatch.setattr(dispatcher, "get_operation", lambda name: op)

    result = dispatcher.run_operation(op.name, {"bounds": bounds}, store=ObjectStore())

    assert not str(result.get("error", "")).startswith("Invalid bounds")


def test_numpy_coordinates_are_a_valid_box():
    from dtcc_agent.dispatcher import bounds_error

    assert bounds_error([np.int64(0), np.float32(0), np.float64(10), 10]) is None

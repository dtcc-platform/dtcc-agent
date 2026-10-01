"""render_object's matplotlib renderer: a PNG, with no OpenGL or display (U9)."""

import numpy as np
import pytest

from dtcc_agent.renderer import render_to_file

model = pytest.importorskip("dtcc_core.model")

PNG = b"\x89PNG\r\n\x1a\n"
SQUARE = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)


def _objects():
    mesh = model.Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], float),
                      faces=np.array([[0, 1, 2]]))
    surface = model.Surface(vertices=SQUARE)
    return {
        "Mesh": mesh,
        "PointCloud": model.PointCloud(points=np.random.default_rng(0).random((500, 3))),
        "Raster": model.Raster(data=np.arange(12, dtype=float).reshape(3, 4)),
        "Surface": surface,
        "MultiSurface": model.MultiSurface(surfaces=[surface]),
        "LineString": model.LineString(vertices=SQUARE),
        "MultiLineString": model.MultiLineString(linestrings=[model.LineString(vertices=SQUARE)]),
        "Bounds": model.Bounds(xmin=0, ymin=0, xmax=5, ymax=5),
    }


@pytest.mark.parametrize("type_name", list(_objects()))
def test_each_geometry_type_renders_a_png(tmp_path, type_name):
    path = tmp_path / "out.png"
    assert render_to_file(_objects()[type_name], type_name, path, width=300, height=200)
    assert path.read_bytes().startswith(PNG)


def test_a_large_point_cloud_is_subsampled_not_refused(tmp_path):
    points = np.random.default_rng(1).random((300_000, 3))
    path = tmp_path / "big.png"
    assert render_to_file(model.PointCloud(points=points), "PointCloud", path)


@pytest.mark.parametrize("obj,type_name", [
    (model.Mesh(), "Mesh"),
    (model.LineString(), "LineString"),
    (model.MultiLineString(linestrings=[model.LineString()]), "MultiLineString"),
])
def test_empty_geometry_draws_nothing(tmp_path, obj, type_name):
    path = tmp_path / "empty.png"
    assert not render_to_file(obj, type_name, path)
    assert not path.exists()


def test_an_oversized_request_is_clamped(tmp_path):
    path = tmp_path / "huge.png"
    assert render_to_file(model.Bounds(xmin=0, ymin=0, xmax=1, ymax=1), "Bounds", path,
                          width=100_000, height=100_000)
    assert path.stat().st_size < 5_000_000


def _building():
    building = model.Building()
    building.add_geometry(model.Surface(vertices=SQUARE), model.GeometryType.LOD0)
    return building


def test_buildings_and_cities_render_their_footprints(tmp_path):
    city = model.City()
    city.add_buildings([_building()])
    collections = pytest.importorskip("dtcc_core.model.object.dataset_collections")
    cases = (
        (_building(), "Building"), ([_building()], "list"), (city, "City"),
        (collections.BuildingCollection(buildings=[_building()]), "BuildingCollection"),
        (collections.FootprintCollection(footprints=[model.Surface(vertices=SQUARE)]),
         "FootprintCollection"),
    )
    for obj, type_name in cases:
        path = tmp_path / f"{type_name}.png"
        assert render_to_file(obj, type_name, path)
        assert path.read_bytes().startswith(PNG)

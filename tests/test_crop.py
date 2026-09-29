"""Tests for spatial cropping of cached objects."""

import numpy as np
import pytest

from dtcc_agent.crop import crop_to_bounds


def test_crop_pointcloud_filters_points():
    """PointCloud-like object: only points within bounds are kept."""
    class FakePC:
        def __init__(self):
            self.points = np.array([
                [100.0, 200.0, 5.0],
                [150.0, 250.0, 6.0],
                [300.0, 400.0, 7.0],  # outside
            ])
            self.classification = np.array([1, 2, 3])

    pc = FakePC()
    cropped = crop_to_bounds(pc, [90, 190, 200, 300])

    assert len(cropped.points) == 2
    assert len(cropped.classification) == 2
    assert cropped.points[0][0] == 100.0


def test_crop_returns_original_if_unknown_type():
    """Unknown types are returned as-is (no cropping)."""
    obj = {"data": 123}
    result = crop_to_bounds(obj, [0, 0, 100, 100])
    assert result is obj


def _building(x, y, outline=None, id=None, attributes=None):
    from dtcc_core.model import Building, GeometryType, Surface

    b = Building() if id is None else Building(id=id)
    b.attributes.update(attributes or {})
    square = [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]]
    vertices = [[vx, vy, 0] for vx, vy in (outline or square)]
    b.add_geometry(Surface(vertices=np.array(vertices, float)), GeometryType.LOD0)
    return b


def test_crop_keeps_only_core_buildings_inside_bounds():
    """What datasets.buildings returns: a Core BuildingCollection (#39)."""
    from dtcc_core.datasets.buildings import BuildingCollection

    city = BuildingCollection([_building(0, 0), _building(500, 500)])
    cropped = crop_to_bounds(city, [-50, -50, 100, 100])

    assert len(cropped.buildings) == 1
    assert len(city.buildings) == 2


def test_crop_drops_a_core_building_it_cannot_place():
    """Without LOD0 geometry footprint() is None. Core's size filter drops such
    a building from a fresh download, so a crop must not count it anywhere."""
    from dtcc_core.datasets.buildings import BuildingCollection
    from dtcc_core.model import Building

    city = BuildingCollection([Building(), _building(0, 0)])
    cropped = crop_to_bounds(city, [-50, -50, 100, 100])

    assert len(cropped.buildings) == 1
    assert cropped.buildings[0].footprint() is not None


def test_crop_keeps_what_a_fresh_core_download_keeps():
    """Core keeps a footprint only when it lies wholly inside the bounds shrunk
    by 2 m. A building whose centre is inside but whose edge crosses them is
    not in a fresh download, so a crop drops it too."""
    from dtcc_core.datasets.buildings import BuildingCollection

    inside = _building(10, 10)          # 10..20, clear of the 2 m margin
    crossing = _building(95, 40)        # 95..105 crosses x = 100
    in_margin = _building(89, 60)       # 89..99 ends inside the 2 m margin
    city = BuildingCollection([inside, crossing, in_margin])

    cropped = crop_to_bounds(city, [0, 0, 100, 100])

    assert cropped.buildings == [inside]


def test_crop_returns_a_core_city_unchanged():
    """A City's buildings cannot be replaced; returning it as-is lets the
    cache treat a larger cached City as a miss rather than fail."""
    from dtcc_core.model import City

    city = City()
    assert crop_to_bounds(city, [0, 0, 100, 100]) is city


def test_crop_drops_every_part_of_a_footprint_that_crosses_the_bounds():
    """Core tests a multi-part footprint whole, then splits it into buildings
    that share the source id. One part outside drops them all."""
    from dtcc_core.datasets.buildings import BuildingCollection

    part_inside = _building(10, 10, id="feature-7")
    part_outside = _building(500, 500, id="feature-7")
    other = _building(30, 30)
    city = BuildingCollection([part_inside, part_outside, other])

    cropped = crop_to_bounds(city, [0, 0, 100, 100])

    assert cropped.buildings == [other]


def test_crop_tests_the_footprint_unsimplified():
    """footprint() simplifies by 1 cm. A 5 mm spike across the 2 m margin
    keeps the building out of a fresh download, so the crop must see it."""
    from dtcc_core.datasets.buildings import BuildingCollection

    spiked = _building(0, 0, outline=[
        [90, 40], [97.995, 40], [97.995, 49.99], [98.003, 50.0],
        [97.995, 50.01], [97.995, 60], [90, 60],
    ])
    city = BuildingCollection([spiked])

    assert crop_to_bounds(city, [0, 0, 100, 100]).buildings == []


@pytest.mark.parametrize("source_id", ["objektidentitet", "osm_id"])
def test_crop_groups_parts_by_the_source_feature_not_the_random_building_id(source_id):
    """LM and OSM parts get random Building ids; the source feature id they
    share lives in the copied properties (as a live download shows)."""
    from dtcc_core.datasets.buildings import BuildingCollection

    feature = {source_id: "3caf7f14"}
    part_inside = _building(10, 10, attributes=feature)
    part_outside = _building(500, 500, attributes=feature)
    other = _building(30, 30, attributes={source_id: "9d01aa20"})
    city = BuildingCollection([part_inside, part_outside, other])

    cropped = crop_to_bounds(city, [0, 0, 100, 100])

    assert cropped.buildings == [other]


def test_buildings_without_any_id_are_judged_one_by_one():
    """PR-Agent: an empty id must not tie unrelated buildings together."""
    from dtcc_core.datasets.buildings import BuildingCollection

    inside = _building(10, 10, id="")
    outside = _building(500, 500, id="")

    cropped = crop_to_bounds(BuildingCollection([inside, outside]), [0, 0, 100, 100])

    assert cropped.buildings == [inside]

"""A download over a set area is refused before it starts (#79).

The M2 baseline asked for a point cloud of "central Gothenburg"; the model
passed the city's whole box, about 20 x 30 km, and the MCP server ran out of
memory partway through 123 LiDAR tiles. Measured on the real run_operation
path: a 7 km square peaked near 6 GB, a 3 km square near 1.7 GB.
"""

import pytest

from dtcc_agent import dispatcher, provenance
from dtcc_agent.object_store import ObjectStore

# q09's bounds from the baseline: "Gothenburg, Sweden" as geocoded.
CITY = [308930.95, 6382798.34, 329583.16, 6413000.0]


def _side(km):
    """A square `km` on a side, in central Gothenburg (EPSG:3006)."""
    half = km * 1000 / 2
    return [319500 - half, 6399500 - half, 319500 + half, 6399500 + half]


@pytest.fixture
def lookups(monkeypatch):
    """Record operations looked up: past the area check, nothing else runs."""
    seen = []

    def get_operation(name):
        seen.append(name)
        raise KeyError("stop here")

    monkeypatch.setattr(dispatcher, "get_operation", get_operation)
    return seen


@pytest.mark.parametrize("bounds", [CITY, _side(3.3)], ids=["q09-city", "just-over"])
def test_a_lidar_download_over_the_cap_is_refused_before_it_starts(lookups, bounds):
    result = dispatcher.run_operation("datasets.point_cloud", {"bounds": bounds}, ObjectStore())
    assert result["error"].startswith("Refused: datasets.point_cloud")
    assert f"{dispatcher.MAX_DOWNLOAD_KM2:g} km²" in result["error"]
    assert lookups == []  # refused before the operation is even looked up
    assert provenance.error_category(result["error"]) == "Refused"


def test_a_download_within_the_cap_goes_ahead(lookups):
    dispatcher.run_operation("datasets.point_cloud", {"bounds": _side(3.0)}, ObjectStore())
    assert lookups == ["datasets.point_cloud"]


@pytest.mark.parametrize("name", sorted(dispatcher.LIGHT_DOWNLOADS))
def test_light_vector_downloads_are_not_capped(lookups, name):
    dispatcher.run_operation(name, {"bounds": CITY}, ObjectStore())
    assert lookups == [name]


def test_every_heavy_download_is_capped_by_default(lookups):
    # A dataset nobody has classified yet is capped, not trusted.
    result = dispatcher.run_operation("datasets.something_new", {"bounds": CITY}, ObjectStore())
    assert result["error"].startswith("Refused")


@pytest.mark.parametrize("raw, value", [(None, 10.0), ("25", 25.0), ("2.5", 2.5)])
def test_the_cap_is_configurable(raw, value):
    assert dispatcher._max_download_km2(raw) == value


@pytest.mark.parametrize("raw", ["0", "-1", "lots", "nan", "inf"])
def test_a_nonsense_cap_stops_the_server(raw):
    with pytest.raises(ValueError, match="DTCC_AGENT_MAX_AREA_KM2"):
        dispatcher._max_download_km2(raw)


def test_get_buildings_is_capped_too(monkeypatch, tmp_path):
    # Building heights come from LiDAR: 25 km² peaked at 5.3 GB, 100 km² was killed.
    import json

    from dtcc_agent import runner, server
    from dtcc_agent.disk_cache import DiskCache

    downloads = []
    monkeypatch.setattr(runner, "fetch_buildings", lambda **kw: downloads.append(kw) or [])
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))

    result = json.loads(server.get_buildings(bounds=CITY))
    assert result["error"].startswith("Refused: datasets.buildings")
    assert downloads == []

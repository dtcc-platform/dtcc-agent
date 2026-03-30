"""Tests for geojson_store module."""

import json

import pytest

from dtcc_agent.geojson_store import (
    load_geojson,
    query_geojson,
    summarize_geojson_property,
)

SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [11.97, 57.70]},
            "properties": {"name": "Building A", "energy_rating": "A", "annual_kwh": 80, "layer_type": "building"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [11.98, 57.71]},
            "properties": {"name": "Building B", "energy_rating": "D", "annual_kwh": 145, "layer_type": "building"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[11.96, 57.69], [11.99, 57.72]]},
            "properties": {"name": "Pipe 1", "diameter": 0.5, "layer_type": "district_heating"},
        },
    ],
}


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "test.geojson"
    path.write_text(json.dumps(SAMPLE_GEOJSON))
    return str(path)


class TestLoadGeojson:
    def test_load_success(self, sample_file):
        result = load_geojson(sample_file)
        assert "error" not in result
        assert result["summary"]["feature_count"] == 3
        assert "Point" in result["summary"]["geometry_types"]
        assert "LineString" in result["summary"]["geometry_types"]
        assert result["summary"]["property_schema"]["name"] == "str"
        assert result["summary"]["property_schema"]["annual_kwh"] == "int"
        assert result["summary"]["bounds"] is not None
        assert result["summary"]["layer_types"] == {"building": 2, "district_heating": 1}
        # Small dataset: features included
        assert "features" in result["summary"]
        assert len(result["summary"]["features"]) == 3

    def test_load_file_not_found(self):
        result = load_geojson("/nonexistent/path.geojson")
        assert "error" in result

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.geojson"
        path.write_text("not json")
        result = load_geojson(str(path))
        assert "error" in result

    def test_load_not_feature_collection(self, tmp_path):
        path = tmp_path / "point.geojson"
        path.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
        result = load_geojson(str(path))
        assert "error" in result


class TestQueryGeojson:
    def test_equality(self):
        result = query_geojson(SAMPLE_GEOJSON, "energy_rating", "==", "D")
        assert result["result"]["match_count"] == 1
        assert result["result"]["features"][0]["properties"]["name"] == "Building B"

    def test_greater_than(self):
        result = query_geojson(SAMPLE_GEOJSON, "annual_kwh", ">", 100)
        assert result["result"]["match_count"] == 1

    def test_contains(self):
        result = query_geojson(SAMPLE_GEOJSON, "name", "contains", "pipe")
        assert result["result"]["match_count"] == 1

    def test_no_matches(self):
        result = query_geojson(SAMPLE_GEOJSON, "energy_rating", "==", "Z")
        assert result["result"]["match_count"] == 0

    def test_invalid_operator(self):
        result = query_geojson(SAMPLE_GEOJSON, "name", "~", "x")
        assert "error" in result

    def test_returns_geojson(self):
        result = query_geojson(SAMPLE_GEOJSON, "layer_type", "==", "building")
        assert result["geojson"]["type"] == "FeatureCollection"
        assert len(result["geojson"]["features"]) == 2


class TestSummarizeProperty:
    def test_numeric(self):
        result = summarize_geojson_property(SAMPLE_GEOJSON, "annual_kwh")
        assert result["type"] == "numeric"
        assert result["count"] == 2
        assert result["min"] == 80.0
        assert result["max"] == 145.0

    def test_categorical(self):
        result = summarize_geojson_property(SAMPLE_GEOJSON, "energy_rating")
        assert result["type"] == "categorical"
        assert result["unique_values"] == 2
        assert result["value_counts"]["A"] == 1
        assert result["value_counts"]["D"] == 1

    def test_missing_property(self):
        result = summarize_geojson_property(SAMPLE_GEOJSON, "nonexistent")
        assert "error" in result

    def test_mixed_numeric(self):
        result = summarize_geojson_property(SAMPLE_GEOJSON, "diameter")
        # Only 1 feature has diameter; the others have None (filtered out)
        assert result["type"] == "numeric"
        assert result["count"] == 1

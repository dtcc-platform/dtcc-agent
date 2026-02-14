"""Tests for the geocoding module."""

import pytest
from dtcc_agent.geocode import geocode, KNOWN_BOUNDS


class TestHardcodedFallbacks:
    """Tests that use hardcoded bounds (no network needed)."""

    def test_lindholmen_exact(self):
        result = geocode("lindholmen")
        assert result["source"] == "hardcoded"
        assert result["bounds"] == KNOWN_BOUNDS["lindholmen"]

    def test_case_insensitive(self):
        result = geocode("Lindholmen")
        assert result["source"] == "hardcoded"

    def test_with_city_suffix(self):
        """'Lindholmen, Gothenburg' should match on the first word."""
        result = geocode("Lindholmen, Gothenburg")
        assert result["source"] == "hardcoded"
        assert result["bounds"] == KNOWN_BOUNDS["lindholmen"]

    def test_chalmers(self):
        result = geocode("Chalmers")
        assert result["source"] == "hardcoded"
        assert len(result["bounds"]) == 4

    def test_center_computed(self):
        result = geocode("lindholmen")
        b = result["bounds"]
        expected_cx = (b[0] + b[2]) / 2
        expected_cy = (b[1] + b[3]) / 2
        assert result["center"] == [expected_cx, expected_cy]

    def test_all_known_bounds_valid(self):
        """Every hardcoded entry should have minx < maxx, miny < maxy."""
        for name, b in KNOWN_BOUNDS.items():
            assert len(b) == 4, f"{name}: expected 4 bounds"
            assert b[0] < b[2], f"{name}: minx >= maxx"
            assert b[1] < b[3], f"{name}: miny >= maxy"


class TestNominatim:
    """Tests that hit the Nominatim API (marked external)."""

    @pytest.mark.external
    def test_unknown_place(self):
        """A place not in hardcoded list should fall back to Nominatim."""
        result = geocode("Stockholm Central Station")
        assert result["source"] == "nominatim"
        assert len(result["bounds"]) == 4
        # Stockholm is roughly x=674000, y=6580000 in EPSG:3006
        assert 670000 < result["center"][0] < 680000

    @pytest.mark.external
    def test_nonexistent_place(self):
        with pytest.raises(ValueError, match="No results"):
            geocode("xyzzy_nonexistent_place_12345")

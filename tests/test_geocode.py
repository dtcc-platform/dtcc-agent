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


class TestSwedenOnly:
    """#80: the agent covers Sweden only, so geocoding must never resolve a
    place to another country. Nominatim is stubbed: no network."""

    @pytest.fixture
    def nominatim(self, monkeypatch):
        """Answer each Nominatim query with `hits`; record the params sent."""
        import httpx
        from dtcc_agent import geocode as module

        sent = {}

        def stub(url, params=None, **kwargs):
            sent.update(params)
            return httpx.Response(200, json=stub.hits, request=httpx.Request("GET", url))

        stub.hits = []
        monkeypatch.setattr(module.httpx, "get", stub)
        return stub, sent

    def test_the_search_is_limited_to_sweden(self, nominatim):
        stub, sent = nominatim
        # What Nominatim answers for "central Gothenburg" with countrycodes=se.
        stub.hits = [{"display_name": "Göteborgs central, Nils Ericsonsplatsen, Göteborg",
                      "boundingbox": ["57.7043483", "57.7143483", "11.9681864", "11.9781864"]}]
        result = geocode("central Gothenburg")
        assert sent["countrycodes"] == "se"
        # Gothenburg is roughly x=319000, y=6399000 in EPSG:3006.
        assert 318000 < result["center"][0] < 321000
        assert 6398000 < result["center"][1] < 6401000

    def test_a_hit_outside_sweden_is_refused(self, nominatim):
        stub, _ = nominatim
        # What Nominatim answered without the filter: a café in Hamilton, New Zealand.
        stub.hits = [{"display_name": "Gothenburg Cafe Restaurant Bar, Hamilton East, New Zealand",
                      "boundingbox": ["-37.7902334", "-37.7901334", "175.2871785", "175.2872785"]}]
        with pytest.raises(ValueError, match="not in Sweden"):
            geocode("central Gothenburg")

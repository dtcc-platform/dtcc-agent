"""Geocoding: place names → EPSG:3006 bounding boxes.

Uses OpenStreetMap Nominatim for lookup, pyproj for CRS
transformation, and hardcoded fallbacks for key Gothenburg
districts where reliable bounding boxes are known.
"""

from __future__ import annotations

import httpx
from pyproj import Transformer

# WGS84 → SWEREF99 TM (EPSG:3006)
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)

# Default half-width in meters when Nominatim returns a point, not a box
_DEFAULT_RADIUS = 250.0

# Hardcoded bounding boxes for key Gothenburg districts (EPSG:3006).
# These are reliable fallbacks when Nominatim gives imprecise results
# or the network is unavailable.
KNOWN_BOUNDS: dict[str, list[float]] = {
    "lindholmen": [319800, 6398800, 320300, 6399300],
    "chalmers": [319200, 6397700, 319800, 6398300],
    "haga": [318800, 6398100, 319300, 6398600],
    "majorna": [317800, 6397600, 318500, 6398200],
    "nordstan": [319500, 6399200, 320000, 6399700],
    "järntorget": [318900, 6398500, 319400, 6399000],
    "kungsportsavenyn": [319300, 6398300, 319700, 6398800],
    "eriksberg": [318800, 6399100, 319500, 6399700],
    "frihamnen": [319900, 6399300, 320500, 6399800],
    "masthugget": [318600, 6398400, 319200, 6398900],
    "biskopsgården": [316900, 6399200, 317700, 6400000],
    "backaplan": [319000, 6399800, 319700, 6400400],
    "gamlestaden": [320600, 6399400, 321300, 6400000],
    "örgryte": [320500, 6397800, 321200, 6398500],
    "mölndal": [319800, 6395700, 320600, 6396500],
}


def _wgs84_to_3006(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 (lon, lat) to EPSG:3006 (x, y)."""
    return _transformer.transform(lon, lat)


def geocode(
    place_name: str,
    radius: float = _DEFAULT_RADIUS,
    timeout: float = 10.0,
) -> dict:
    """Convert a place name to an EPSG:3006 bounding box.

    Parameters
    ----------
    place_name : str
        Free-text place name, e.g. "Lindholmen, Gothenburg"
        or just "Lindholmen".
    radius : float
        Half-width of bounding box in meters when the geocoder
        returns a point instead of an area. Default 250m.
    timeout : float
        HTTP timeout for Nominatim request.

    Returns
    -------
    dict with keys:
        query: str — the original query
        bounds: list[float] — [minx, miny, maxx, maxy] in EPSG:3006
        center: list[float] — [x, y] center point in EPSG:3006
        source: str — "hardcoded" or "nominatim"
        display_name: str — resolved name from Nominatim (or the key)
    """
    # Check hardcoded fallbacks first
    key = place_name.strip().lower()
    # Also try just the first word (handles "Lindholmen, Gothenburg")
    first_word = key.split(",")[0].strip()

    for candidate in [key, first_word]:
        if candidate in KNOWN_BOUNDS:
            b = KNOWN_BOUNDS[candidate]
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            return {
                "query": place_name,
                "bounds": b,
                "center": [cx, cy],
                "source": "hardcoded",
                "display_name": candidate.title(),
            }

    # Nominatim lookup
    params = {
        "q": place_name,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "dtcc-agent/0.1 (research; chalmers.se)"}

    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Geocoding failed for '{place_name}': {exc}. "
            f"Known places: {', '.join(sorted(KNOWN_BOUNDS.keys()))}"
        ) from exc

    if not results:
        raise ValueError(
            f"No results for '{place_name}'. "
            f"Known hardcoded places: {', '.join(sorted(KNOWN_BOUNDS.keys()))}"
        )

    hit = results[0]
    display_name = hit.get("display_name", place_name)

    # Nominatim returns a boundingbox as [south, north, west, east] in WGS84
    if "boundingbox" in hit:
        south, north, west, east = [float(v) for v in hit["boundingbox"]]
        x_min, y_min = _wgs84_to_3006(west, south)
        x_max, y_max = _wgs84_to_3006(east, north)
        bounds = [
            round(x_min, 2),
            round(y_min, 2),
            round(x_max, 2),
            round(y_max, 2),
        ]
    else:
        # Point result — build box from radius
        lat = float(hit["lat"])
        lon = float(hit["lon"])
        cx, cy = _wgs84_to_3006(lon, lat)
        bounds = [
            round(cx - radius, 2),
            round(cy - radius, 2),
            round(cx + radius, 2),
            round(cy + radius, 2),
        ]

    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2

    return {
        "query": place_name,
        "bounds": bounds,
        "center": [round(cx, 2), round(cy, 2)],
        "source": "nominatim",
        "display_name": display_name,
    }

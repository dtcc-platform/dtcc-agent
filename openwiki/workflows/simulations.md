---
type: workflow
title: Simulations, runs and geocoding
description: The task-shaped tools dtcc-agent exposes directly. Geocoding turns a place into EPSG:3006 bounds, get_buildings gives a building inventory, and simulations run in-process or in a remote dtcc-sim, with scenario comparison and Run bookkeeping.
tags: [simulation, dtcc-sim, geocoding, runs, workflow]
sources:
  - id: openwiki-source-4fc133fdcc1bf230bdb18f76
    resource: repo://dtcc_agent/analysis.py
  - id: openwiki-source-4de2c7ae8f7f75886035ba34
    resource: repo://dtcc_agent/geocode.py
  - id: openwiki-source-32c33c58f635fb0708a0e8c6
    resource: repo://dtcc_agent/runner.py
  - id: openwiki-source-3cb4a6487d73410befc45a84
    resource: repo://dtcc_agent/serializers.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
generated: { by: "claude-code", at: "2026-09-29T19:33:34.851Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-29T19:33:34.851Z
---

# Simulations, runs and geocoding

A typical conversation begins with a place name and ends with a comparison of numbers:

1. `geocode` turns the place into bounds.
2. `get_buildings` describes the urban context.
3. `run_simulation` or `compare_scenarios` runs heat or air-quality simulations.

These tools are hardcoded in `server.py` and use `runner.py`, `geocode.py` and `analysis.py`. They sit beside the generic `run_operation` path described in [Dispatch, object references and serialization](../concepts/dispatch-and-object-store.md).

## Geocoding (`geocode.py`)

`geocode(place_name, radius=250)` returns `{query, bounds, center, source, display_name}` in **EPSG:3006** (SWEREF99 TM):

1. **Hardcoded districts first.** `KNOWN_BOUNDS` holds 15 Gothenburg districts (Lindholmen, Chalmers, Haga and others). The full lower-cased query is checked, then its first comma-separated part, so "Lindholmen, Gothenburg" matches. `source` is `"hardcoded"`.
2. **Otherwise Nominatim.** It sends `GET https://nominatim.openstreetmap.org/search` with a `dtcc-agent/0.1` User-Agent. A returned `boundingbox` (south, north, west, east in WGS84) is transformed with pyproj. A point result becomes a square of ±`radius` metres. `source` is `"nominatim"`.
3. **Failures.** HTTP errors raise `RuntimeError`, and an empty result raises `ValueError`. Both messages list the known places.

The chatbot's system prompt asks for the 250 m default, because large boxes download millions of points.

## Buildings (`get_buildings`)

The tool works in two steps so the download, not the answer, can be cached. `runner.fetch_buildings` downloads the Core `buildings` dataset as a `BuildingCollection`, only on a cache miss. `runner.summarize_buildings` turns the buildings of the area asked into a JSON summary on every call; a larger cached download is first cropped to that area (see [Disk cache](../concepts/disk-cache.md)). The summary holds:

- the building count;
- per-building height, ground height, footprint vertex count, and footprint area from the shoelace formula on `lod0`;
- height statistics over buildings with a positive height, each `None` when no building has one;
- total footprint area over every building, not only the `max_buildings` listed;
- a `truncated` flag when the count exceeds `max_buildings`.

A height is Core's `estimated_height`, else its `measured_height` (`serializers.building_height`, the same precedence Core's own meshing uses). `Building.height` is only the measurement, which a download leaves empty, so reading it reported every building as 0 m until #51. Live on Lindholmen the heights are 3.0 to 28.5 m.

`get_buildings`, `run_simulation` and `compare_scenarios` first check their bounds with `dispatcher.bounds_error` and return `{"error": "Invalid bounds ..."}` for a box with no area (inverted, zero-size, the wrong length or not finite), before any download or run.

The MCP tool checks the disk cache first and shares both its cache entry and its download flight with `datasets.buildings` for the same area and source. See [Disk cache](../concepts/disk-cache.md).

## Simulations (`runner.py`)

Only `urban_heat_simulation` and `air_quality_field` are treated as simulations (`_SIMULATION_NAMES`). The path depends on configuration:

- **Remote (mini-service mode)**, when `DTCC_SIM_SERVICE_URL` or `DTCC_REMOTE_SERVICES` is set:
  - The configured URLs are registered with dtcc-core's `register_remote_service`, each once it answers. `list_simulations`, `get_simulation_schema` and `run` register any service not yet registered on the spot. Separately, a background thread started with the operation catalogue asks every 30 s until each service has answered, so the same datasets also reach `list_operations` and `run_operation` without a request waiting on dtcc-sim (see [Operation catalogue](../concepts/operation-catalogue.md)).
  - Datasets are resolved from Core's registry and must carry a `base_url`.
  - `run()` validates and calls the remote descriptor's `build()`. This reuses Core's submit, status and result protocol rather than a copy of it.
  - It returns a `RemoteSimulationResult` (task id, size, format, `remote=True`). **The light container does not deserialise the FEniCSx output, so no field statistics are computed.**
- **Local (direct mode)**:
  - `import dtcc_sim.datasets` registers the simulations. The import is lazy, so the mini-service starts without FEniCSx.
  - `run()` calls the dataset in-process and returns a dolfinx `Function`, whose values are `result.x.array`.

`list_simulations`, `get_simulation_schema` (the dataset's `show_options()` JSON schema) and `run` all follow the same branch.

## Tools and Runs

| Tool | Behaviour |
|---|---|
| `list_simulations` | Name and description of each available simulation |
| `get_simulation_schema(name)` | Parameter JSON schema |
| `run_simulation(name, bounds, parameters, label)` | Runs and stores a Run; returns `run_id` and a field summary (local) or `remote_result` (remote) |
| `compare_scenarios(name, bounds, a, b, label_a, label_b)` | Runs A then B on the same bounds; locally returns per-scenario summaries plus B−A difference statistics |
| `list_past_runs(limit)` | Most recent Runs in this Session, with summaries |
| `get_run_summary(run_id)` | Re-summarises one Run |

`analysis.summarize_field` drops non-finite values and reports min, max, mean, std, median, the 5th and 95th percentiles, and the count. `compare_fields` requires equal value counts, meaning the same mesh; otherwise it returns an error. The field name is inferred as `temperature` or `concentration`.

**Run bookkeeping.** `_store_result` mints `run_id = str(uuid4())[:8]`, stores `{simulation, bounds, parameters, result, timestamp}` in the Session's `results`, and **also** stores the result in the Session's ObjectStore under a separate object id with `label=run_id`. The Run does not record that object id, and `get_run_summary` does not return it, so reaching the Object from a Run means listing objects and matching labels. ADR-0010 changes this so that a Run records the Object reference it yielded, with typed `run_…` and `obj_…` references. Runs are per Session. See [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Tests

- `tests/test_analysis.py` covers statistics, NaN handling and size mismatch.
- `tests/test_geocode.py` covers the hardcoded fallbacks offline and Nominatim under the `external` marker.
- `tests/test_server.py` covers the Run tools' error payloads and Run/Object id characterisation.

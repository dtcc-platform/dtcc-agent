---
type: concept
title: Artifacts and the file boundary
description: How files leave and enter dtcc-agent. Tools write renders and exports into a private per-Session folder and return a name, never a path; the chatbot serves them at /artifacts; run_operation refuses every file-path parameter; load_geojson reads only the shared results folder.
tags: [artifacts, filesystem, security, rendering, export, chatbot]
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-f4e2e41b78ef4f0591d5eac2
    resource: repo://chatbot/sessions.py
  - id: openwiki-source-58b44e3a6e999eaaad2363b4
    resource: repo://dtcc_agent/artifacts.py
  - id: openwiki-source-88dfce5777b4bb33b4b64cf7
    resource: repo://dtcc_agent/dispatcher.py
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-e00d8f9ae7bfdd9a24f35525
    resource: repo://dtcc_agent/renderer.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-45f696cdc913fb69ab9dbb96
    resource: repo://tests/test_artifact_tools.py
  - id: openwiki-source-104417024fe98bcbdd7d4cb6
    resource: repo://tests/test_artifacts.py
  - id: openwiki-source-f7a9ed42f310602857a88be4
    resource: repo://tests/test_path_refusal.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
---

# Artifacts and the file boundary

Two rules govern files since T7 (#63):

1. **No path from chat reaches the filesystem** (U1, #10). Nothing a user or the model types is ever opened, read or written as a path.
2. **Files a tool produces belong to one Session** (ADR-0004). They live in that Session's own folder, are named so they cannot be guessed, and are served only to that Session.

`dtcc_agent/artifacts.py` owns the layout. The MCP server writes through it and the chatbot reads through it, so both agree on where a file is even when they run as two services (T13): they mount the same `/data` volume.

## Per-Session artifact folders

- The root is `$DTCC_AGENT_ARTIFACTS_DIR`, defaulting to `<system temp>/dtcc_agent_artifacts`; the image sets `/data/artifacts`.
- Each Session writes under `<root>/<session id>/`, created `0700`. The Session id must match `[A-Za-z0-9_-]{1,64}`; anything else raises before a directory is made.
- `new_path(session, stem, suffix)` names a file `<32 hex token>_<stem>.<ext>`. The token comes from `secrets.token_hex(16)`; unsafe stem characters are flattened, so a stem like `../../etc/passwd` stays inside the folder.
- `describe(path)` is what a tool returns: `{"name": ..., "kind": "image" | "file"}`. The filesystem path never appears in a tool result.
- `find(session, name)` resolves a name only if it has the artifact name shape and exists as a regular file (not a symlink) in that Session's folder. Another Session's name, a traversal string or a malformed name returns `None`.
- `remove_session(session)` deletes the folder. The chatbot calls it when a session is removed or expires.

Which Session a tool writes for comes from `_Session.id`: the `X-DTCC-Session` header over HTTP, or `DTCC_AGENT_SESSION` (set by the chatbot) over stdio, else `"local"`. See [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Serving: the chatbot's `/artifacts` route and frames

- `GET /artifacts/{session_id}/{name}` returns 404 unless the session is live in `SessionManager` and `find()` resolves the name. Every response carries `Cache-Control: private, no-store` and `X-Content-Type-Options: nosniff`. PNGs are served inline as `image/png`; anything else is an attachment named after the artifact without its token (`download_name`).
- Until admission control (T14) the URL is the credential: the live session id plus the random token.
- After each tool result the chatbot runs `_artifact_frame`. It unwraps FastMCP's structured form (`{"result": "<tool JSON>"}`, which is what the Claude CLI actually delivers), checks that the named artifact exists in *this* session's folder, and sends the page an `image` frame or a `file` frame with the URL. The page appends an image to the message or a download link (`chatbot/static/index.html`).
- The old `/renders` static mount is gone: it served a fixed folder the renderer never wrote to (#38).

## Tools that write artifacts

**`render_object`** draws with matplotlib's Agg backend (`dtcc_agent/renderer.py`, U9). There is no OpenGL, display or main-thread requirement, so it runs in the worker pool like any tool.

| Object | View |
|---|---|
| City, BuildingCollection, FootprintCollection, Building, list of Buildings | plan view of footprints |
| Surface, MultiSurface, LineString, MultiLineString, RoadNetwork, Bounds, SensorCollection | plan view of polygons, lines or points |
| Mesh | 3D triangles, coloured by height, at most 20,000 faces |
| PointCloud, VolumeMesh | 3D points, at most 50,000 |
| Raster | image with a colour bar, nodata masked |

Width and height are clamped to 100–2400 px. Geometry with nothing to draw returns "Nothing to render" instead of a blank PNG. dtcc-viewer was set aside because it was never installed, requires `dtcc-core@develop` (conflicting with the pin) and needs a GL display. Whether the page draws geometry itself is still open in the rebuild plan (D1/D2, M3); only `renderer.py` would change.

**`export_object(object_ref, format)`** has no path parameter. It writes into the Session folder, deletes a partial file if the writer fails, and returns a `file` artifact. Formats come from `_EXPORT_DISPATCH` (type → format → `dtcc_core.io` writer):

| Type | Formats |
|---|---|
| PointCloud | csv, las, laz, json |
| Mesh | obj, ply, stl, vtk, vtu, gltf |
| VolumeMesh | obj, ply, stl, vtk, vtu |
| Raster | csv, tif, png, jpg |
| City, BuildingCollection, list of Buildings | geojson, gpkg (footprints with attributes; GeoJSON in WGS84), json (Core's City JSON) |
| SensorCollection | csv (written by the agent) |

Core's building writers are keyed on `City`, so a `BuildingCollection` or a list of Buildings is wrapped in one by `_as_city` first (#68).

## Path refusal in `run_operation`

`ParamInfo.is_path` (`dtcc_agent/registry.py`) marks a parameter as a file or directory path when its name is `path`, `filename` or `outfile`, ends in `_path` or `_dir`, or its type mentions `Path` or `format: path`. Before anything else runs, the dispatcher refuses an operation if a path parameter is required, or if an optional one is given a value other than `null` or `""`. The refusal names `export_object` and `load_geojson` as the supported routes.

Against the pinned Core this covers 18 operations: the `io.load_*` and `io.save_*` family and the two tetgen debug-folder options of `builder.build_city_volume_mesh` and `datasets.city_volume_mesh`. `tests/test_path_refusal.py` pins the exact set, so a Core upgrade that adds a path parameter fails the suite instead of becoming reachable.

## Reading: `load_geojson`

`load_geojson(name)` takes a path relative to `$SHARED_RESULTS_DIR`, where dtcc-sim writes results. `shared_result()` resolves it, follows symlinks, and refuses anything absolute, anything that resolves outside the folder, and any missing file. Without `SHARED_RESULTS_DIR` set, nothing can be loaded. Nothing in the agent yet tells the model which files dtcc-sim wrote.

## Tests

- `tests/test_artifacts.py`: folder layout and permissions, unguessable names, cross-Session and traversal lookups, invalid Session ids, removal, shared-results containment including a symlink escape.
- `tests/test_path_refusal.py`: the pinned set of path operations; required and given paths refused before Core is called; `null`/`""`/absent optional paths allowed.
- `tests/test_artifact_tools.py`: render and export return artifacts in the calling Session with no paths in the text; a failed export leaves no file; building exports in each format, in WGS84; `load_geojson` refusals.
- `tests/test_renderer.py`: every geometry type produces a PNG; subsampling; empty geometry draws nothing; size clamping.
- `tests/test_chatbot_app.py`: the route's 200/404 cases and headers; frames for image, file and FastMCP-wrapped results.

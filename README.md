# dtcc-agent

LLM agent interface for the [DTCC Platform](https://github.com/dtcc-platform).
Exposes urban digital twin simulations as [MCP](https://modelcontextprotocol.io/)
tools, enabling Claude and other LLM agents to discover datasets, inspect urban
context, run environmental simulations, and interpret results through natural
language.

## Architecture

```
LLM (Claude) ←→ MCP protocol ←→ dtcc-agent
                                    ├── geocode           (pyproj + Nominatim)
                                    ├── get_buildings      (dtcc_core.datasets)
                                    ├── run_simulation     (dtcc_sim)
                                    ├── compare_scenarios  (composes the above)
                                    ├── list_operations    ─┐
                                    ├── describe_operation  │ dynamic dispatch
                                    ├── run_operation       │ (109 operations)
                                    ├── list_objects        │ with object store
                                    ├── inspect_object     ─┘
                                    └── render_object       (dtcc_viewer, offscreen)
```

Unlike dtcc-deploy (which proxies HTTP calls to a FastAPI backend),
dtcc-agent calls `dtcc_core` and `dtcc_sim` **directly in-process**.
Simulation results stay in memory as numpy arrays — no file I/O
needed for the agent to reason about results.

## Prerequisites

dtcc-agent runs in the same conda environment as dtcc-sim. You need:

- FEniCSx (dolfinx) via conda
- dtcc-core and dtcc (pip)
- dtcc-sim (pip)
- dtcc-tetgen-wrapper (from source)

See the [dtcc-sim README](https://github.com/dtcc-platform/dtcc-sim)
for environment setup instructions.

## Installation

```bash
conda activate fenicsx-env   # or your dtcc environment
pip install -e .
```

## Usage

### With Claude Desktop / Claude Code

Add to your MCP configuration (`.mcp.json` or Claude Desktop settings):

```json
{
  "mcpServers": {
    "dtcc-agent": {
      "command": "conda",
      "args": ["run", "-n", "fenicsx-env", "python", "-m", "dtcc_agent"]
    }
  }
}
```

### Standalone

```bash
python -m dtcc_agent
```

## Available Tools

### Simulation tools (hardcoded)

| Tool | Description |
|------|-------------|
| `geocode` | Convert place names to EPSG:3006 bounding boxes |
| `get_buildings` | Inspect buildings in a bounding box |
| `list_simulations` | Discover available simulation types |
| `get_simulation_schema` | Get parameter schema for a simulation |
| `run_simulation` | Run a simulation and return summary statistics |
| `compare_scenarios` | Run two simulations and compare results |
| `list_past_runs` | List recent simulation runs |
| `get_run_summary` | Re-inspect a past simulation result |

### Dynamic dispatch tools (109 operations)

| Tool | Description |
|------|-------------|
| `list_operations` | Browse the full catalog of dtcc-core operations |
| `describe_operation` | Get parameter schema for any operation |
| `run_operation` | Execute any operation and store the result |
| `list_objects` | List objects in the in-memory store |
| `inspect_object` | Get detailed summary of a stored object |

### Visualization tools

| Tool | Description |
|------|-------------|
| `render_object` | Render a stored object as a PNG screenshot (offscreen 3D via dtcc-viewer) |

The dynamic dispatch tools expose **all** of dtcc-core:

| Category | Count | Examples |
|----------|-------|---------|
| `builder` | 73 | terrain rasters, surface meshes, building heights, tree detection, pointcloud filters |
| `io` | 18 | load/save pointcloud, mesh, raster, city; download data |
| `datasets` | 12 | point_cloud, buildings, terrain, trees, weather, simulations |
| `reproject` | 6 | reproject pointcloud, mesh, surface between CRS |

## How It Works

### Object store and pipelines

Results from `run_operation` are stored in memory with short 8-character
hex IDs. You pass these IDs as parameters to subsequent operations,
building multi-step pipelines:

```
run_operation("datasets.point_cloud", {"bounds": [...]})
  → result_id: "a1b2c3d4"   (PointCloud stored in memory)

run_operation("builder.build_terrain_raster", {"pc": "a1b2c3d4", "cell_size": 2.0})
  → result_id: "e5f6g7h8"   (Raster stored, built from the point cloud)
```

Parameters marked `is_object_ref: true` in `describe_operation` output
accept these IDs. The dispatcher resolves them from the store automatically.

The store uses LRU eviction (default 2 GB limit) to prevent unbounded
memory growth during long sessions.

### What gets returned

`run_operation` never returns raw array data. Instead, it returns
type-specific summaries:

- **PointCloud**: num_points, bounds, classification counts, z-stats
- **Mesh**: num_vertices, num_faces, bounds, marker info
- **Raster**: shape, cell_size, bounds, value statistics (min/max/mean/std)
- **City**: num_buildings, has_terrain, height stats
- **Tuples**: each element stored separately with linked IDs

Use `inspect_object(id)` to get a detailed summary of any stored object.

## Examples

### Example 1: Heatwave impact analysis

```
User: "How does a heatwave affect Lindholmen in Gothenburg?"

Agent calls: geocode("Lindholmen")
  → bounds [318866, 6399800, 319366, 6400300] in EPSG:3006

Agent calls: get_buildings(bounds)
  → 47 buildings, 3–45m height

Agent calls: compare_scenarios("urban_heat_simulation", bounds,
               {air_temperature: 20}, {air_temperature: 38})
  → baseline mean 17.8°C, heatwave mean 36.2°C, max 47.1°C

Agent: "Temperature increases by 18.4°C on average, with dangerous
        peaks of 47.1°C in street canyons..."
```

### Example 2: Terrain analysis pipeline

Download a point cloud, build a terrain raster, then compute slope:

```
User: "Analyze the terrain slope around Chalmers"

Agent calls: geocode("Chalmers")
  → bounds [319700, 6399500, 320200, 6400000]

Agent calls: list_operations(search="terrain")
  → shows datasets.point_cloud, builder.build_terrain_raster,
    builder.raster.slope_aspect, ...

Agent calls: run_operation("datasets.point_cloud",
               {"bounds": [319700, 6399500, 320200, 6400000]})
  → result_id: "a1b2c3d4", summary: 1.2M points, z range 5–48m

Agent calls: run_operation("builder.build_terrain_raster",
               {"pc": "a1b2c3d4", "cell_size": 2.0})
  → result_id: "e5f6g7h8", summary: 250×250 raster, elevation 5–48m

Agent calls: run_operation("builder.raster.slope_aspect",
               {"dem": "e5f6g7h8"})
  → result_ids: ["f9g0h1i2", "j3k4l5m6"]  (slope + aspect rasters)

Agent calls: inspect_object("f9g0h1i2")
  → slope raster: mean 0.12 rad (7°), max 0.8 rad (46°)

Agent: "The terrain around Chalmers is mostly gentle (mean slope 7°)
        with some steep sections reaching 46° near the hillside..."
```

### Example 3: Building and tree inventory

```
User: "What buildings and trees are in Lindholmen?"

Agent calls: geocode("Lindholmen")
  → bounds [318866, 6399800, 319366, 6400300]

Agent calls: run_operation("datasets.buildings",
               {"bounds": [318866, 6399800, 319366, 6400300]})
  → result_id: "b1c2d3e4", 47 buildings, heights 3–45m

Agent calls: run_operation("datasets.point_cloud",
               {"bounds": [318866, 6399800, 319366, 6400300]})
  → result_id: "f5g6h7i8"

Agent calls: run_operation("builder.trees_from_pointcloud",
               {"pc": "f5g6h7i8"})
  → result_id: "j9k0l1m2", 128 trees, height 4–18m

Agent calls: list_objects()
  → 3 objects, 45 MB total

Agent: "Lindholmen has 47 buildings (3–45m tall) and approximately
        128 detected trees (4–18m tall) from the LiDAR data..."
```

### Example 4: Visualizing buildings in 3D

Download buildings for an area and render a 3D screenshot:

```
User: "Show me the buildings in Lindholmen"

Agent calls: geocode("Lindholmen")
  → bounds [318866, 6399800, 319366, 6400300]

Agent calls: run_operation("datasets.buildings",
               {"bounds": [318866, 6399800, 319366, 6400300]})
  → result_id: "b1c2d3e4", 47 buildings, heights 3–45m

Agent calls: render_object("b1c2d3e4")
  → {"image_path": "/tmp/dtcc_screenshots_xxx/b1c2d3e4.png",
     "type": "City", "width": 1200, "height": 800}

Agent: "Here's a 3D rendering of the 47 buildings in Lindholmen.
        The tallest building reaches 45m..."
```

### Example 5: Terrain visualization pipeline

Build a terrain raster and render it:

```
User: "Visualize the terrain around Chalmers"

Agent calls: geocode("Chalmers")
  → bounds [319700, 6399500, 320200, 6400000]

Agent calls: run_operation("datasets.point_cloud",
               {"bounds": [319700, 6399500, 320200, 6400000]})
  → result_id: "a1b2c3d4", 1.2M points

Agent calls: run_operation("builder.build_terrain_raster",
               {"pc": "a1b2c3d4", "cell_size": 2.0})
  → result_id: "e5f6g7h8", 250×250 raster, elevation 5–48m

Agent calls: render_object("e5f6g7h8")
  → {"image_path": "/tmp/dtcc_screenshots_xxx/e5f6g7h8.png",
     "type": "Raster", "width": 1200, "height": 800}

Agent: "Here's the terrain raster for the Chalmers area. Elevation
        ranges from 5m near the waterfront to 48m on the hillside..."
```

### Example 6: Discovering operations

```
Agent calls: list_operations(category="builder")
  → 73 operations: build_terrain_raster, build_city_surface_mesh,
    trees_from_pointcloud, merge_meshes, slope_aspect, ...

Agent calls: describe_operation("builder.build_city_surface_mesh")
  → params: city (object_ref, required), max_mesh_size (float, default 10),
    min_mesh_angle (float, default 20.7), ...

Agent calls: describe_operation("builder.pc_filter.remove_vegetation")
  → params: pc (object_ref, required)
  → returns: PointCloud
```

## Web Chatbot

The chatbot provides a browser-based chat interface powered by Claude Agent SDK.

### Install

```bash
pip install -e ".[chatbot]"
```

### Run

```bash
python -m chatbot
```

Then open http://localhost:8000 in your browser.

### How it works

The chatbot uses the Claude Agent SDK to connect to dtcc-agent as an MCP server.
User messages are sent via WebSocket, and Claude's responses (including tool calls
and 3D renders) stream back in real-time.

## License

MIT License. This project is part of the
[DTCC Platform](https://github.com/dtcc-platform/) developed at the
[Digital Twin Cities Centre](https://dtcc.chalmers.se/).

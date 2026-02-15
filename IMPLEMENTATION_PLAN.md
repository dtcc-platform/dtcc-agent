# dtcc-agent Implementation Plan

## What this repo is

A standalone MCP server that gives LLMs direct in-process access to
dtcc-core and dtcc-sim. No HTTP, no FastAPI, no dtcc-atlas dependency.

```
dtcc-agent/
├── dtcc_agent/
│   ├── __init__.py
│   ├── __main__.py       ← entry point (python -m dtcc_agent)
│   ├── server.py         ← MCP server, all tool definitions
│   ├── geocode.py        ← place names → EPSG:3006 bounds
│   ├── analysis.py       ← summary stats from numpy arrays
│   └── runner.py         ← wraps dtcc_core/dtcc_sim dataset calls
├── tests/
│   ├── test_geocode.py   ← works standalone (no dtcc deps)
│   └── test_analysis.py  ← works standalone (pure numpy)
├── pyproject.toml
├── README.md
└── .mcp.json             ← Claude Desktop / Claude Code config
```

## Architecture vs dtcc-deploy

dtcc-deploy routes everything through HTTP to dtcc-atlas:

    MCP → httpx → FastAPI (dtcc-atlas) → dtcc-sim → file on disk → download

dtcc-agent calls Python directly:

    MCP → runner.py → dtcc_core / dtcc_sim → dolfinx Function in memory → numpy

This means:
- No HTTP latency between MCP and simulation
- Results stay in memory as numpy arrays (never serialized to XDMF)
- The agent can re-inspect past results without re-downloading files
- No changes needed in dtcc-atlas or dtcc-core

## MCP Tools provided

| Tool                  | What it does                                        | Calls               |
|-----------------------|-----------------------------------------------------|----------------------|
| `geocode`             | Place name → EPSG:3006 bounding box                 | Nominatim + pyproj   |
| `get_buildings`       | Inspect buildings in a bounding box                  | dtcc_core.datasets   |
| `list_simulations`    | Discover available simulation types                  | dtcc_core registry   |
| `get_simulation_schema` | Get parameter schema for a simulation              | Pydantic JSON Schema |
| `run_simulation`      | Run simulation, return summary stats                 | dtcc_sim + numpy     |
| `compare_scenarios`   | Run two sims on same area, return comparison         | dtcc_sim + numpy     |
| `list_past_runs`      | Show previous runs stored in memory                  | in-memory dict       |
| `get_run_summary`     | Re-inspect a past run's statistics                   | in-memory dict       |


## What's done

- [x] Full repo structure with pyproject.toml
- [x] Geocoding with hardcoded Gothenburg fallbacks + Nominatim
- [x] Analysis module (summarize_field, compare_fields)
- [x] Runner module wrapping dtcc_core dataset registry
- [x] MCP server with all 8 tools
- [x] In-memory result store with run_id references
- [x] Tests for geocode and analysis (no dtcc deps needed)
- [x] README with architecture diagram and example
- [x] .mcp.json for Claude Desktop / Claude Code


## What needs to be done before first demo

### 1. Integration test with real dtcc environment

The code is written but untested against the actual conda environment
with FEniCSx, dtcc-core, dtcc-sim, and dtcc-tetgen-wrapper installed.

```bash
conda activate fenicsx-env
cd dtcc-agent
pip install -e .

# Verify imports work
python -c "from dtcc_agent.runner import list_simulations; print(list_simulations())"

# Run a simulation
python -c "
from dtcc_agent.runner import run
from dtcc_agent.analysis import summarize_field
result = run('urban_heat_simulation', bounds=[319895.96, 6398909.72, 320095.96, 6399109.72])
print(summarize_field(result.x.array, 'temperature'))
"
```

**Likely issues to debug:**
- Import order: `dtcc_sim.datasets` auto-registers on import, but
  the registration may fail if dtcc-core isn't installed first.
  runner.py handles this by importing dtcc_sim.datasets at module level.
- MPI initialization: dolfinx uses mpi4py. In single-process mode
  this should be transparent, but verify.
- Memory: simulations on large domains produce large meshes. May need
  to limit bounding box size in the tools.

### 2. Docker integration (optional, for dtcc-deploy container)

If you want dtcc-agent available in the existing Docker container
alongside dtcc-atlas, add to the Dockerfile:

```dockerfile
# After dtcc-sim install
COPY dtcc-agent/pyproject.toml dtcc-agent/README.md ./dtcc-agent/
COPY dtcc-agent/dtcc_agent/ ./dtcc-agent/dtcc_agent/
RUN cd dtcc-agent && pip install --no-cache-dir .
```

But for the paper demo, running standalone in the conda env is
simpler and sufficient.

### 3. Nominatim network access

geocode.py calls `nominatim.openstreetmap.org`. If running in
a network-restricted environment, only the hardcoded Gothenburg
districts will work. For the paper demo this is fine since all
example scenarios are in Gothenburg.

### 4. Record demo transcript

Once integration works, run the full scenario from the paper:

```
User: "Compare urban heat in Lindholmen under current conditions
       versus a summer heatwave."

Expected tool call sequence:
  1. geocode("Lindholmen")
  2. list_simulations()
  3. get_simulation_schema("urban_heat_simulation")
  4. get_buildings(bounds)
  5. run_simulation("urban_heat_simulation", bounds, baseline params)
  6. run_simulation("urban_heat_simulation", bounds, heatwave params)
  → or: compare_scenarios(...) which does steps 5-6 in one call
```

Capture the full MCP tool call log for the paper appendix.


## Optional enhancements (lower priority)

### Result visualization

Add a `render_result_image(run_id)` tool that uses PyVista to
render a top-down heatmap PNG. This requires:
- PyVista with offscreen rendering (EGL or Xvfb)
- PIL for numpy→PNG conversion
- Returning base64-encoded image or saving to file

For the paper, static figures generated offline are sufficient.
The MCP tool would only matter for an interactive demo.

### Air quality scenario

The `air_quality_field` simulation is already registered. A demo
scenario could be:

```
User: "Show me current NO2 levels across central Gothenburg."

  1. geocode("central Gothenburg") → large bounding box
  2. run_simulation("air_quality_field", bounds, {phenomenon: "NO2"})
  → returns concentration field from SMHI sensor interpolation
```

This requires live SMHI API access, which may timeout. Worth
testing but urban_heat is the more reliable demo.

### Expanded geocoding

Add more Swedish cities to the hardcoded fallback dictionary,
or cache Nominatim results to avoid repeated lookups.


## Relationship to the paper

The paper ("AI-Orchestrated Urban Digital Twins...") needs to show
that an LLM agent can autonomously:

1. **Discover** available simulations and their parameters
2. **Ground** its reasoning in real urban data (buildings)
3. **Configure** simulations based on natural language intent
4. **Execute** simulations and read the results
5. **Compare** scenarios and synthesize recommendations

dtcc-agent provides tools for all five steps. The paper's
evaluation section should include:
- The full tool-call transcript for the heatwave comparison
- Statistics showing the agent chose reasonable parameters
- A figure showing baseline vs heatwave temperature fields
- Discussion of where the agent's reasoning was correct/incorrect

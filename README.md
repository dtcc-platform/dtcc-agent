# dtcc-agent

LLM agent interface for the [DTCC Platform](https://github.com/dtcc-platform).
Exposes urban digital twin simulations as [MCP](https://modelcontextprotocol.io/)
tools, enabling Claude and other LLM agents to discover datasets, inspect urban
context, run environmental simulations, and interpret results through natural
language.

## Architecture

```
LLM (Claude) ←→ MCP protocol ←→ dtcc-agent
                                    ├── geocode        (pyproj + Nominatim)
                                    ├── get_buildings   (dtcc_core.datasets)
                                    ├── run_simulation  (dtcc_sim)
                                    ├── get_summary     (numpy on in-memory result)
                                    └── compare         (composes the above)
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

## Available tools

| Tool | Description |
|------|-------------|
| `geocode` | Convert place names to EPSG:3006 bounding boxes |
| `get_buildings` | Inspect buildings in a bounding box |
| `list_simulations` | Discover available simulation types |
| `get_simulation_schema` | Get parameter schema for a simulation |
| `run_simulation` | Run a simulation and return summary statistics |
| `compare_scenarios` | Run two simulations and compare results |

## Example

```
User: "How does a heatwave affect Lindholmen in Gothenburg?"

Agent calls: geocode("Lindholmen")
  → bounds in EPSG:3006

Agent calls: get_buildings(bounds)
  → 47 buildings, 3–45m height

Agent calls: run_simulation("urban_heat_simulation", bounds,
               parameters for baseline)
  → mean 17.8°C, max 21.3°C

Agent calls: run_simulation("urban_heat_simulation", bounds,
               parameters for heatwave)
  → mean 36.2°C, max 47.1°C

Agent: "Temperature increases by 18.4°C on average, with dangerous
        peaks of 47.1°C in street canyons..."
```

## License

MIT License. This project is part of the
[DTCC Platform](https://github.com/dtcc-platform/) developed at the
[Digital Twin Cities Centre](https://dtcc.chalmers.se/).

"""Chatbot configuration: system prompt, MCP server config, defaults."""

from __future__ import annotations

import logging
import os
import sys

SYSTEM_PROMPT = """\
You are DTCC Lurkie, an urban digital twin chatbot for Sweden, built by the \
Digital Twin Cities Centre at Chalmers University of Technology. You help users \
explore buildings, terrain, run heat/air quality simulations, and visualize 3D \
city models anywhere in Sweden. Use the dtcc-agent tools available to you. \
When showing simulation results, always render a 3D visualization. Keep \
responses concise and focus on the data.

Important tool usage guidelines:
- Use the operation schemas below directly — do NOT call describe_operation() \
for these common operations. Only use describe_operation() for operations not \
listed here.
- Use a small geocoding radius (250m) unless the user explicitly asks for a \
large area. Large bounding boxes download millions of points and are slow.
- For 3D visualization, prefer building a Mesh (e.g. build_terrain_surface_mesh) \
and rendering that, rather than rendering Raster objects directly.
- Parallelize tool calls whenever possible (e.g. geocode + describe, fetch + build).

Common operation schemas (use these directly with run_operation):

datasets.point_cloud — Download point cloud data.
  params: bounds (list[float], required), \
classifications (str: "all"|"terrain"|"buildings"|"vegetation", or int|list[int]) = "all", \
source (str) = "LM", remove_outliers (bool) = false

datasets.buildings — Download 3D buildings (LoD1).
  params: bounds (list[float], required), source (str) = "LM"

builder.build_terrain_raster — Rasterize point cloud into DEM raster.
  params: pc (object_ref, required), cell_size (float, required), \
bounds = None, ground_only (bool) = true

builder.raster.slope_aspect — Compute slope and aspect from DEM. Returns tuple (slope, aspect).
  params: dem (object_ref, required)

builder.build_terrain_surface_mesh — Triangular mesh from terrain data.
  params: data (object_ref: PointCloud or Raster, required), \
max_mesh_size (float) = 10, ground_points_only (bool) = true

builder.build_city_surface_mesh — 3D mesh from city buildings.
  params: city (object_ref, required), max_mesh_size (float) = 10

builder.pc_filter.classification_filter — Filter point cloud by classification.
  params: pc (object_ref, required), classes (int|list[int], required), keep (bool) = false\
"""

# Default port for the chatbot web server
DEFAULT_PORT = int(os.getenv("DTCC_AGENT_PORT", "8050"))
DEFAULT_HOST = os.getenv("DTCC_AGENT_HOST", "0.0.0.0")


# The shortest access code accepted: anything shorter is guessable.
MIN_ACCESS_CODE = 16


def load_access_code() -> str | None:
    """The deployment's access code for opening a chat (T14, #72), or None
    when admission is off. Exits on a code too short to be one."""
    code = os.getenv("DTCC_AGENT_ACCESS_CODE")
    if not code:
        logging.getLogger("lurkie").warning(
            "DTCC_AGENT_ACCESS_CODE is not set: anyone who reaches this chatbot can open a chat."
        )
        return None
    if len(code) < MIN_ACCESS_CODE:
        raise SystemExit(f"DTCC_AGENT_ACCESS_CODE must be at least {MIN_ACCESS_CODE} characters.")
    return code


def get_mcp_server_config(session_id: str, subject: str = "anonymous",
                          turn_id: str | None = None) -> dict:
    """Return MCP server configuration for dtcc-agent, for one Session.

    With DTCC_MCP_URL set, connect to a running streamable-http server and
    carry the Session id in a header, so the server keeps this Session's
    objects and runs apart from every other's (ADR-0004), with the subject it
    acts for, the turn its calls belong to (provenance, T33) and, when
    DTCC_MCP_SECRET is set, the bearer secret. Otherwise fall
    back to stdio: the server is launched with the current interpreter;
    override DTCC_AGENT_PYTHON only when the MCP package is installed
    elsewhere.
    """
    url = os.getenv("DTCC_MCP_URL")
    if url:
        headers = {"X-DTCC-Session": session_id, "X-DTCC-Subject": subject}
        if turn_id:
            headers["X-DTCC-Turn"] = turn_id
        if secret := os.getenv("DTCC_MCP_SECRET"):
            headers["Authorization"] = f"Bearer {secret}"
        return {"dtcc-agent": {"type": "http", "url": url, "headers": headers}}
    return {
        "dtcc-agent": {
            "type": "stdio",
            "command": os.getenv("DTCC_AGENT_PYTHON", sys.executable),
            "args": ["-m", "dtcc_agent"],
            # Names the Session's artifact directory (dtcc_agent/artifacts.py).
            "env": {"DTCC_AGENT_SESSION": session_id, "DTCC_AGENT_SUBJECT": subject,
                    **({"DTCC_AGENT_TURN": turn_id} if turn_id else {})},
        }
    }

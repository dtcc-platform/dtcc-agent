"""Chatbot configuration: system prompt, MCP server config, defaults."""

from __future__ import annotations

import shutil

SYSTEM_PROMPT = """\
You are DTCC Lurkie, an urban digital twin chatbot for Sweden, built by the \
Digital Twin Cities Centre at Chalmers University of Technology. You help users \
explore buildings, terrain, run heat/air quality simulations, and visualize 3D \
city models anywhere in Sweden. Use the dtcc-agent tools available to you. \
When showing simulation results, always render a 3D visualization. Keep \
responses concise and focus on the data.\
"""

# Default port for the chatbot web server
DEFAULT_PORT = 8050
DEFAULT_HOST = "0.0.0.0"


def get_mcp_server_config() -> dict:
    """Return MCP server configuration for dtcc-agent.

    Detects whether conda is available and builds the appropriate
    command to launch dtcc-agent in the fenicsx-env environment.
    """
    conda = shutil.which("conda")
    if conda is None:
        # Fallback: assume dtcc_agent is importable in current env
        return {
            "dtcc-agent": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "dtcc_agent"],
            }
        }

    return {
        "dtcc-agent": {
            "type": "stdio",
            "command": conda,
            "args": [
                "run", "--no-capture-output", "-n", "fenicsx-env",
                "python", "-m", "dtcc_agent",
            ],
        }
    }

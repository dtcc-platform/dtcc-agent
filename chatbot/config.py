"""Chatbot configuration: system prompt, MCP server config, defaults.

SYSTEM_PROMPT is the pydantic-ai runtime's base prompt; the operation catalogue
follows it (chatbot/runtime/pai.py). The SDK runtime keeps M2's prompt.
"""

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
- Use a small geocoding radius (250m) unless the user explicitly asks for a \
large area. Large bounding boxes download millions of points and are slow.
- For 3D visualization, prefer building a Mesh (e.g. build_terrain_surface_mesh) \
and rendering that, rather than rendering Raster objects directly.
- Parallelize tool calls whenever possible (e.g. geocode + describe, fetch + build).\
"""

# Default port for the chatbot web server
DEFAULT_PORT = int(os.getenv("DTCC_AGENT_PORT", "8050"))
DEFAULT_HOST = os.getenv("DTCC_AGENT_HOST", "0.0.0.0")


# The model runs on Amazon Bedrock (#29). Claude has no on-demand model IDs in
# eu-north-1, only the eu.* cross-region inference profiles.
DEFAULT_MODEL = "eu.anthropic.claude-sonnet-5-5"
DEFAULT_REGION = "eu-north-1"


def load_model() -> str:
    """The Bedrock model ID the agent answers with: DTCC_AGENT_MODEL, or Sonnet 5.5."""
    return os.getenv("DTCC_AGENT_MODEL") or DEFAULT_MODEL


def bedrock_env() -> dict[str, str]:
    """Environment that points the Claude CLI at Bedrock. Credentials are
    inherited from the chatbot's own environment.

    The CLI also calls a small model for internal steps. It gets the same
    model as the answers: Haiku 4.5, its default, is refused on our account
    until Anthropic's use-case form is filed, and one model keeps every call
    on a model we know is enabled."""
    return {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": os.getenv("AWS_REGION") or DEFAULT_REGION,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": load_model(),
    }


def require_bedrock_credentials() -> None:
    """Exit unless some AWS credential is configured. Without one every chat
    turn would fail; better to refuse to start, and never fall back to
    another provider."""
    if not any(os.getenv(name) for name in
               ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE")):
        raise SystemExit("No Bedrock credentials: set AWS_BEARER_TOKEN_BEDROCK "
                         "(or standard AWS credentials) to start the chatbot.")


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

"""Smoke tests for the MCP server entry point.

The rest of the suite exercises the modules behind the tools but never imports
`server.py`, so the suite stayed green while `python -m dtcc_agent` could not
start at all: a fresh install resolved `mcp` 2.x, where the
`mcp.server.fastmcp` path server.py imports was removed.

These tests close that gap. They import the server module and enumerate its
tools, which is the smallest check that fails when the package cannot start.
"""

import asyncio

import pytest


def test_server_module_imports():
    """server.py imports — the check the suite was missing."""
    from dtcc_agent import server

    assert server.mcp is not None
    assert callable(server.main)


def test_server_exposes_tools():
    """The server constructs and exposes its tool surface."""
    from dtcc_agent.server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == 22, f"expected 22 tools, got {len(tools)}"


@pytest.mark.parametrize(
    "name",
    [
        "geocode",
        "list_simulations",
        "run_simulation",
        "compare_scenarios",
        "list_operations",
        "run_operation",
        "render_object",
    ],
)
def test_documented_tool_is_registered(name):
    """Tools the README and the MCP client config rely on are present."""
    from dtcc_agent.server import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert name in names


def test_entry_point_module_imports():
    """`python -m dtcc_agent` resolves."""
    import importlib.util

    spec = importlib.util.find_spec("dtcc_agent.__main__")
    assert spec is not None

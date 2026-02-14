"""Entry point for `python -m dtcc_agent`."""

# dtcc_core uses asyncio.run() internally for downloads (lidar, gpkg).
# The MCP server runs inside an async event loop, so we must patch asyncio
# to allow nested event loops before anything else is imported.
import nest_asyncio
nest_asyncio.apply()

from .server import main

main()

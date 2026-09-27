# Files

- [MCP server and tool execution](mcp-server-and-tool-execution.md) - How dtcc-agent registers its MCP tools, runs each call in a bounded worker pool bound to a Session, deduplicates concurrent downloads, and starts over stdio or stateless streamable-http.
- [Architecture overview](overview.md) - End-to-end picture of dtcc-agent, from the Lurkie web chatbot through the MCP server and generic dispatch to dtcc-core and dtcc-sim, with a module ownership map and the two deployment modes.
- [Sessions and isolation](sessions-and-isolation.md) - How dtcc-agent makes the Session the isolation unit (ADR-0004), covering the X-DTCC-Session header, per-Session object stores and runs, the eight-Session cap with LRU eviction, the local stdio Session, and the gaps that remain.

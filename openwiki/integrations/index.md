# Files

- [Lurkie chatbot](chatbot-lurkie.md) - The chatbot package's FastAPI app, Lurkie. It serves the page, admits chats with an access code, runs the /chat WebSocket protocol, hands each turn to the pydantic-ai runtime, stores memory, writes one answer record per turn, and serves artifacts while their session is live.
- [DTCC Engine integration](dtcc-engine.md) - How dtcc-agent runs simulations as DTCC Engine jobs when DTCC_ENGINE_URL is set. It submits, polls with backoff, downloads the result as a .dtccpkg read by Core, maps every Engine answer to a message the model can relay, and replaces the dtcc-sim path and its shared results folder with no fallback.

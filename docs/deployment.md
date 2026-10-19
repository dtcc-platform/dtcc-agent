# What dtcc-agent needs to run

Input for the deployment design. This lists what the agent needs, not how to deploy it.
`docker-compose.yml` is the reference: every variable named here is set there, and
`tests/test_deployment_doc.py` fails when the two drift apart.

## Shape

- **One image, two containers.** The image is `dtcc-agent`: `python:3.12-slim`, linux/amd64,
  about 900 MB, built from the repo `Dockerfile`.
  - **`dtcc-agent-mcp`, the tool server:** imports dtcc-core and serves the 22 tools.
  - **`dtcc-agent`, the chat server:** FastAPI and a websocket, with the static chat UI. It
    calls the model on Bedrock.
- **The two containers share one network namespace.** The chat reaches the tool server on
  `127.0.0.1:8051`. Nothing else may reach port 8051.
- **The only public port is 8050:** HTTP for the UI and `/health`, and the websocket on `/chat`.
  Put HTTPS in front of it.

## Secrets

| Name | Used by | What |
|---|---|---|
| `DTCC_AGENT_ACCESS_CODE` | chat | The code a user types to open a chat |
| `DTCC_MCP_SECRET` | both | Shared secret sent on every chat-to-tool call |
| `AWS_BEARER_TOKEN_BEDROCK` | chat | Bedrock API key. **The current key expires 2026-11-01.** In AWS, an IAM role with `bedrock:InvokeModel*` is better. |
| `DTCC_ENGINE_TOKEN` | tool server | The Engine's API token. Required when `DTCC_ENGINE_URL` is set, or the tool server won't start. |

Compose refuses to start if any of the first three is missing.

## Settings

| Name | Default | What |
|---|---|---|
| `AWS_REGION` | `eu-north-1` | Bedrock region |
| `DTCC_AGENT_MODEL` | `eu.anthropic.claude-sonnet-5-5` | The model. Changing the model is this one setting. |
| `DTCC_ENGINE_URL` | unset | **Engine mode:** the Engine's base URL. When set, simulations run as Engine jobs over HTTP, and dtcc-sim and the shared folder aren't used. |
| `DTCC_ENGINE_TIMEOUT` | `900` | Seconds one simulation may take before it times out |
| `DTCC_REMOTE_SERVICES` | `http://host.docker.internal:8001` | dtcc-sim's URL. Used only without the Engine. |
| `DTCC_SIM_SERVICE_URL` | unset | A single dtcc-sim URL that overrides `DTCC_REMOTE_SERVICES`. Used only without the Engine. |
| `DTCC_MCP_WORKERS` | `4` | Core operations running at once, across all chats. One chat gets at most half. |
| `DTCC_AGENT_MAX_AREA_KM2` | `10` | The largest area one building or point-cloud download may cover |
| `DTCC_MCP_MEM_LIMIT` | `5g` | The tool server's memory limit |
| `DTCC_AGENT_DATA` | `./data/agent` | Host folder mounted at `/data` in both containers |
| `DTCC_SHARED_RESULTS` | `../dtcc-sim/data/shared-results` | Host folder shared with dtcc-sim. Used only without the Engine. |
| `DTCC_AGENT_IMAGE` / `DTCC_AGENT_TAG` | `dtcc-agent` / `local` | The image to run |
| `DOCKER_PLATFORM` | `linux/amd64` | The image's platform |
| `APP_UID` / `APP_GID` / `EXTRAS` | `1000` / `1000` / `chatbot` | Build arguments: the user the image runs as, and the optional dependencies it installs |

## Storage

- **`/data`**, mounted in both containers, and persistent:
  - `cache/`: downloaded buildings and terrain. It can be deleted; it refills.
  - `artifacts/`: renders and exports a user downloads, kept per chat.
  - `logs/`: answer records (model, cost, tools).
  - `memory/`: conversation memory.
- **`/shared/results`**, tool server only, **not needed in Engine mode.** Without the Engine,
  the agent reads every simulation result from a folder dtcc-sim also mounts; only the job
  request goes over HTTP. So without the Engine, the agent and dtcc-sim need the same machine
  or a shared volume (for example EFS). With the Engine, results come back over HTTP and this
  folder goes away.

## Memory and CPU

- **Tool server:**
  - a 5 GB memory limit (`DTCC_MCP_MEM_LIMIT`);
  - a 2 GiB budget for objects kept in memory, shared by all chats, of which one chat may
    hold at most half. An object is anything a chat keeps between steps, such as downloaded
    buildings, meshes or simulation results;
  - each download capped at 10 km² (`DTCC_AGENT_MAX_AREA_KM2`).
- **Measured** (2026-10-09, local, emulated amd64): 10 chats at once peaked at 4.15 of 5 GiB,
  with no restarts. **For 20 users, plan about 8 GB and raise the object budget to match.**
  The object budget is a constant in `dtcc_agent/server.py` today, not a setting, so raising
  it takes a code change.
- **Chat server:** small, mostly waiting on Bedrock.
- **CPU:** Core work (meshing, cropping) is CPU-bound, and 4 workers keep 4 cores busy.
  Simulations run in dtcc-sim or the Engine, not in the agent.

## State, and why it matters for a gateway

- **Chats live in memory.**
  - The chat server keeps a chat's conversation until it has been idle for 1 hour.
  - The tool server keeps each chat's objects. It drops a chat idle for 1 hour, and beyond
    that keeps the 8 most recently used idle chats. A chat in use is never dropped.
- **Not stateless.** With more than one agent instance behind a gateway, **every request of
  a chat must reach the same instance.** The websocket already stays on one instance per
  connection; a reconnect must too, so routing must be sticky per chat.
- A restart loses open chats. The cache and artifacts on `/data` survive.

## Health and restarts

- **Tool server healthy:** port 8051 accepts connections. That happens only after the
  catalogue is built, which takes up to about 2 minutes on start.
- **Chat healthy:** `GET /health` answers and the tool server is reachable. On failure the
  chat stops itself, and `restart: unless-stopped` brings it back inside the tool server's
  network.
- **Logs:** to stdout, plus answer records in `/data/logs`.

## Outbound network

- Bedrock (`bedrock-runtime.eu-north-1.amazonaws.com`).
- Lantmäteriet and OSM, through dtcc-core's downloads.
- Nominatim (`nominatim.openstreetmap.org`), to turn place names into coordinates.
- dtcc-sim or the Engine.

## Open, for the design

1. **The Engine.** Does the deployment get a dev Engine? Then the agent sends simulations
   there and needs no shared folder.
2. **Gateway keys.** Either one service key for the agent, with per-user limits kept inside
   the agent, or one key per end user from the platform's backend.
3. **Bedrock.** An IAM role or a key. If a key: who renews it, and does QA get its own?
4. **The object budget.** Make it a setting before sizing for 20 users, or change the
   constant in the image.

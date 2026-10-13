# Files

- [Deployment, configuration and CI](deployment-and-ci.md) - Covers the Docker image and two-service compose layout, Bedrock credentials, admission, memory limits and restarts, every environment variable the server, chatbot and cache read (including the DTCC Engine settings), and the GitHub workflows (build-tests, dtcc-core contract, PR-Agent).
- [Provenance and evaluation](provenance-and-evaluation.md) - How every answer can be traced and measured. The chatbot writes one answers.jsonl line per turn (model, prompt version, tokens, cost, error and HTTP status), the MCP server one operations.jsonl line per tool call, joined on the turn id, and eval/measure.py runs a fixed question set through the chat to report latency, tokens and cost, compared only against a same-day control.

FROM python:3.12-slim

ARG APP_UID=1000
ARG APP_GID=1000
# The SDK runtime (`sdk` extra) is M3's rollback, kept for one milestone.
# Build with EXTRAS=chatbot for an image with no SDK and no claude CLI (#86).
ARG EXTRAS=chatbot,sdk

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/dtcc-agent \
    XDG_CACHE_HOME=/data/cache \
    DTCC_AGENT_ARTIFACTS_DIR=/data/artifacts \
    DTCC_AGENT_HOST=0.0.0.0 \
    DTCC_AGENT_PORT=8050

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" dtcc-agent && \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home dtcc-agent

WORKDIR /app
COPY . /app

# dtcc-core comes from the commit pinned in pyproject.toml, the only one the
# contract workflow tests (U10, #14). The check fails the build if pip
# installed anything else.
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[${EXTRAS}]" && \
    python -c "import json, tomllib, importlib.metadata as m; \
pin = next(d for d in tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies'] if d.startswith('dtcc-core')).rsplit('@', 1)[1]; \
got = json.loads(m.distribution('dtcc-core').read_text('direct_url.json'))['vcs_info']['commit_id']; \
assert got == pin, f'dtcc-core {got} is not the pinned {pin}'; \
print('dtcc-core', got)" && \
    mkdir -p /data/artifacts /data/cache /data/logs /data/memory /shared/results /home/dtcc-agent/.cache && \
    chown -R dtcc-agent:dtcc-agent /app /data /shared /home/dtcc-agent

USER dtcc-agent

EXPOSE 8050

CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8050"]

FROM python:3.12-slim

ARG DTCC_CORE_REF=develop
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/dtcc-agent \
    XDG_CACHE_HOME=/data/cache \
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

RUN pip install --upgrade pip setuptools wheel && \
    pip install "dtcc-core @ git+https://github.com/dtcc-platform/dtcc-core.git@${DTCC_CORE_REF}" && \
    pip install -e ".[chatbot]" && \
    mkdir -p /data/cache /data/logs /data/memory /shared/results /home/dtcc-agent/.cache && \
    chown -R dtcc-agent:dtcc-agent /app /data /shared /home/dtcc-agent

USER dtcc-agent

EXPOSE 8050

CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8050"]

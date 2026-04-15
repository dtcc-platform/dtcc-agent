FROM python:3.12-slim

ARG DTCC_CORE_REF=develop

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DTCC_AGENT_HOST=0.0.0.0 \
    DTCC_AGENT_PORT=8050

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install "dtcc-core @ git+https://github.com/dtcc-platform/dtcc-core.git@${DTCC_CORE_REF}" && \
    pip install -e ".[chatbot]"

EXPOSE 8050

CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8050"]

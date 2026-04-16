#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DTCC_AGENT_IMAGE="${DTCC_AGENT_IMAGE:-dtcc-agent}"
export DTCC_AGENT_TAG="${DTCC_AGENT_TAG:-local}"
export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DTCC_CORE_REF="${DTCC_CORE_REF:-develop}"
export APP_UID="${APP_UID:-1000}"
export APP_GID="${APP_GID:-1000}"

echo "Building ${DTCC_AGENT_IMAGE}:${DTCC_AGENT_TAG} via docker compose"

cd "${SCRIPT_DIR}"
docker compose build dtcc-agent

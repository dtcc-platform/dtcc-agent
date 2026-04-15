#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${DTCC_AGENT_IMAGE:-dtcc-agent}"
IMAGE_TAG="${DTCC_AGENT_TAG:-local}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
DTCC_CORE_REF="${DTCC_CORE_REF:-develop}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${SCRIPT_DIR}/Dockerfile"

docker build \
  --platform "${PLATFORM}" \
  --build-arg "DTCC_CORE_REF=${DTCC_CORE_REF}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

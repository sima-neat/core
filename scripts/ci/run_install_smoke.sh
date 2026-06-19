#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-install-smoke}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$(mktemp -d /tmp/sima-neat-install-XXXXXX)}"
CONSUMER_BUILD_DIR="${CONSUMER_BUILD_DIR:-build-install-smoke-consumer}"

echo "[install-smoke] configuring project..."
cmake -S . -B "${BUILD_DIR}"

echo "[install-smoke] building core target..."
cmake --build "${BUILD_DIR}" --target sima_neat -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[install-smoke] installing to ${INSTALL_PREFIX}..."
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}" --component core
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}" --component dev

echo "[install-smoke] configuring downstream consumer..."
cmake -S tests/install_smoke -B "${CONSUMER_BUILD_DIR}" \
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release

echo "[install-smoke] building downstream consumer..."
cmake --build "${CONSUMER_BUILD_DIR}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[install-smoke] running downstream consumer..."
"${CONSUMER_BUILD_DIR}/install_smoke_app"

echo "[install-smoke] passed."

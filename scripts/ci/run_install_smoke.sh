#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-install-smoke}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$(mktemp -d /tmp/sima-neat-install-XXXXXX)}"
CONSUMER_BUILD_DIR="${CONSUMER_BUILD_DIR:-build-install-smoke-consumer}"

echo "[install-smoke] configuring project..."
cmake -S . -B "${BUILD_DIR}"

echo "[install-smoke] validating development package dependencies..."
if ! grep -q "simaai-memory-lib-dev" "${BUILD_DIR}/CPackConfig.cmake"; then
  echo "sima-neat-dev must depend on simaai-memory-lib-dev so downstream C++ apps can link SimaNeat::sima_neat on a DevKit." >&2
  exit 1
fi

echo "[install-smoke] building core and development targets..."
cmake --build "${BUILD_DIR}" --target sima_neat sima_neat_static -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[install-smoke] installing to ${INSTALL_PREFIX}..."
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}" --component core
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}" --component dev

EXPECTED_ABI="$(sed -n 's/^set(SimaNeat_ABI_VERSION "\([0-9][0-9]*\)")$/\1/p' \
  "${BUILD_DIR}/SimaNeatConfig.cmake")"
if [[ ! "${EXPECTED_ABI}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[install-smoke] failed to read a valid public ABI from SimaNeatConfig.cmake." >&2
  exit 1
fi
if [[ ! -L "${INSTALL_PREFIX}/lib/libsima_neat.so.${EXPECTED_ABI}" ]]; then
  echo "[install-smoke] missing libsima_neat.so.${EXPECTED_ABI} SONAME link." >&2
  exit 1
fi
LINKER_TARGET="$(readlink "${INSTALL_PREFIX}/lib/libsima_neat.so")"
if [[ "$(basename "${LINKER_TARGET}")" != "libsima_neat.so.${EXPECTED_ABI}" ]]; then
  echo "[install-smoke] libsima_neat.so targets '${LINKER_TARGET}', expected ABI ${EXPECTED_ABI}." >&2
  exit 1
fi

echo "[install-smoke] configuring downstream consumer..."
cmake -S tests/install_smoke -B "${CONSUMER_BUILD_DIR}" \
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release

echo "[install-smoke] building downstream consumer..."
cmake --build "${CONSUMER_BUILD_DIR}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[install-smoke] running downstream consumer..."
"${CONSUMER_BUILD_DIR}/install_smoke_app"

echo "[install-smoke] passed."

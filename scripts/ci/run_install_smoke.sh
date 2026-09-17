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
if grep 'CPACK_DEBIAN_DEV_PACKAGE_DEPENDS' "${BUILD_DIR}/CPackConfig.cmake" |
    grep -Eq 'neat-internals-dev|simaai-memory-lib-dev|sima-lmm-dev|simaai-heap-dev'; then
  echo "Shared customers must not require implementation development packages." >&2
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
  -DCMAKE_BUILD_TYPE=Release \
  -DSIMANEAT_SMOKE_STATIC=ON

echo "[install-smoke] building downstream consumer..."
cmake --build "${CONSUMER_BUILD_DIR}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[install-smoke] running downstream consumer..."
"${CONSUMER_BUILD_DIR}/install_smoke_app"

echo "[install-smoke] running statically linked downstream consumer..."
"${CONSUMER_BUILD_DIR}/install_smoke_static_app"

echo "[install-smoke] checking static dependency paths..."
if grep -nE '/[^";> ]*lib(z|cpp-httplib|simaaimem)\.(so|a)' \
    "${INSTALL_PREFIX}/lib/cmake/SimaNeat/SimaNeatStaticTargets.cmake"; then
  echo "Static targets contain a build-host dependency path." >&2
  exit 1
fi

echo "[install-smoke] passed."

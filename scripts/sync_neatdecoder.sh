#!/usr/bin/env bash
set -euo pipefail

SRC_DIR_DEFAULT="/home/sima/stable_pipeline_session/Session/build-neatdecoder-new"
SRC_DIR="${1:-$SRC_DIR_DEFAULT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
LIB_NAME="libgstneatdecoder.so"
SRC_PATH="${SRC_DIR}/${LIB_NAME}"

if [[ ! -f "${SRC_PATH}" ]]; then
  echo "Neatdecoder binary not found: ${SRC_PATH}" >&2
  echo "Pass the build directory as the first argument or update SRC_DIR_DEFAULT." >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script installs system-wide to ${PLUGIN_DIR} and requires root." >&2
  echo "Re-run with sudo: sudo ${BASH_SOURCE[0]} ${SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${PLUGIN_DIR}"
cp -f "${SRC_PATH}" "${PLUGIN_DIR}/"

echo "Copied ${LIB_NAME} -> ${PLUGIN_DIR}"

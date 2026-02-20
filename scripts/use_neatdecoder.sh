#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="${SIMA_GST_PLUGIN_DIR:-/usr/lib/aarch64-linux-gnu/neat/gst-plugins}"
LIB_NAME="libgstneatdecoder.so"

if [[ ! -d "${PLUGIN_DIR}" ]]; then
  echo "Missing NEAT plugin directory: ${PLUGIN_DIR}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "${PLUGIN_DIR}/${LIB_NAME}" ]]; then
  echo "Missing ${PLUGIN_DIR}/${LIB_NAME}." >&2
  return 1 2>/dev/null || exit 1
fi

export SIMA_GST_PLUGIN_DIR="${PLUGIN_DIR}"
export GST_PLUGIN_PATH_1_0="${PLUGIN_DIR}"
unset GST_PLUGIN_PATH
if [[ -n "${SIMA_SET_GST_SYSTEM_PATH:-}" ]]; then
  export GST_PLUGIN_SYSTEM_PATH_1_0=""
fi

echo "SIMA_GST_PLUGIN_DIR=${SIMA_GST_PLUGIN_DIR}"
echo "GST_PLUGIN_PATH_1_0=${GST_PLUGIN_PATH_1_0}"
if [[ -n "${SIMA_SET_GST_SYSTEM_PATH:-}" ]]; then
  echo "GST_PLUGIN_SYSTEM_PATH_1_0=${GST_PLUGIN_SYSTEM_PATH_1_0}"
fi

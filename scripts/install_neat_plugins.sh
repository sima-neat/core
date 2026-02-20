#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR_DEFAULT="/usr/lib/aarch64-linux-gnu/neat/gst-plugins"
PLUGIN_DIR="${1:-$PLUGIN_DIR_DEFAULT}"
SYS_PLUGIN_DIRS=(
  "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
  "/lib/aarch64-linux-gnu/gstreamer-1.0"
)
LIBS=("libgstneatdecoder.so" "libgstneatencoder.so")

missing=()
for lib in "${LIBS[@]}"; do
  if [[ ! -f "${PLUGIN_DIR}/${lib}" ]]; then
    missing+=("${lib}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing staged plugin(s) in ${PLUGIN_DIR}: ${missing[*]}" >&2
  echo "Build and copy them into ${PLUGIN_DIR} first." >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script installs system-wide to: ${SYS_PLUGIN_DIRS[*]}" >&2
  echo "Re-run with sudo: sudo ${BASH_SOURCE[0]} ${PLUGIN_DIR}" >&2
  exit 1
fi

for dir in "${SYS_PLUGIN_DIRS[@]}"; do
  mkdir -p "${dir}"
  for lib in "${LIBS[@]}"; do
    cp -f "${PLUGIN_DIR}/${lib}" "${dir}/"
    echo "Copied ${lib} -> ${dir}"
  done
done

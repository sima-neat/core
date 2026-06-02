#!/usr/bin/env bash
set -euo pipefail

# Pre-install cleanup for self-hosted Modalix runners. Keep this limited to
# storage only; runtime services and installed package state may not exist yet.
TARGET_COPROCESSING_DIR="/data/simaai/coprocessing"
ROOT_FREE_SPACE_THRESHOLD_MB="${ROOT_FREE_SPACE_THRESHOLD_MB:-500}"

if [[ $# -gt 0 ]]; then
  pass="$1"
else
  pass="${DEVKIT_PASSWORD:-edgeai}"
fi

run_step() {
  local label="$1"
  shift
  printf "[storage-cleanup] %s...\n" "$label"
  printf '%s\n' "$pass" | sudo -S -p '' "$@"
  local rc=$?
  printf "[storage-cleanup] %s rc=%d\n" "$label" "$rc"
  return "$rc"
}

cleanup_coprocessing() {
  if [[ "${TARGET_COPROCESSING_DIR}" != "/data/simaai/coprocessing" ]]; then
    printf "[storage-cleanup] coprocessing cleanup skipped: unexpected target %s\n" "${TARGET_COPROCESSING_DIR}"
    return 1
  fi

  if [[ ! -d "${TARGET_COPROCESSING_DIR}" ]]; then
    printf "[storage-cleanup] coprocessing cleanup skipped: directory not found %s\n" "${TARGET_COPROCESSING_DIR}"
    return 0
  fi

  run_step "empty coprocessing directory" find "${TARGET_COPROCESSING_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
}

root_free_space_mb() {
  df -Pm / | awk 'NR==2 {print $4}'
}

cleanup_tmp_simaai() {
  run_step "remove /tmp/simaai-*" bash -c '
    shopt -s nullglob
    paths=(/tmp/simaai-*)
    if (( ${#paths[@]} > 0 )); then
      rm -rf -- "${paths[@]}"
    fi
  '
}

cleanup_tmp_sima_if_root_low_space() {
  local free_mb
  free_mb="$(root_free_space_mb)"

  if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
    printf "[storage-cleanup] /tmp/sima_* cleanup skipped: unable to read free space on /\n"
    return 1
  fi

  if (( free_mb >= ROOT_FREE_SPACE_THRESHOLD_MB )); then
    printf "[storage-cleanup] /tmp/sima_* cleanup skipped: root free space is %sMB (threshold %sMB)\n" "$free_mb" "$ROOT_FREE_SPACE_THRESHOLD_MB"
    return 0
  fi

  printf "[storage-cleanup] root free space is %sMB (< %sMB), removing /tmp/sima_*\n" "$free_mb" "$ROOT_FREE_SPACE_THRESHOLD_MB"
  run_step "remove /tmp/sima_* for low root free space" bash -c '
    shopt -s nullglob
    paths=(/tmp/sima_*)
    if (( ${#paths[@]} > 0 )); then
      rm -rf -- "${paths[@]}"
    fi
  '
}

report_root_space() {
  local free_mb
  free_mb="$(root_free_space_mb)"
  if [[ "$free_mb" =~ ^[0-9]+$ ]]; then
    printf "[storage-cleanup] root free space: %sMB (threshold %sMB)\n" "$free_mb" "$ROOT_FREE_SPACE_THRESHOLD_MB"
  else
    printf "[storage-cleanup] root free space: unknown\n"
  fi
}

report_root_space
cleanup_coprocessing
cleanup_tmp_simaai
cleanup_tmp_sima_if_root_low_space
report_root_space

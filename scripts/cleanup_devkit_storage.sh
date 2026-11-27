#!/usr/bin/env bash
set -euo pipefail

# Pre-install cleanup for self-hosted Modalix runners. Keep this limited to
# storage only; runtime services and installed package state may not exist yet.
TARGET_COPROCESSING_DIR="/data/simaai/coprocessing"
TMP_ROOT="${DEVKIT_STORAGE_TMP_ROOT:-/tmp}"
TMP_FREE_SPACE_THRESHOLD_MB="${TMP_FREE_SPACE_THRESHOLD_MB:-1024}"

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

free_space_mb() {
  df -Pm "$1" | awk 'NR==2 {print $4}'
}

remove_tmp_matches() {
  local label="$1"
  shift
  # shellcheck disable=SC2016
  run_step "$label" bash -c '
    shopt -s nullglob
    root="$1"
    shift
    paths=()
    for pattern in "$@"; do
      paths+=("${root}"/${pattern})
    done
    if (( ${#paths[@]} > 0 )); then
      rm -rf -- "${paths[@]}"
    fi
  ' bash "${TMP_ROOT}" "$@"
}

cleanup_tmp_simaai() {
  remove_tmp_matches "remove ${TMP_ROOT}/simaai-*" 'simaai-*'
}

# Cancelled jobs are killed before their EXIT traps run; this runner is idle now.
cleanup_tmp_interrupted_runs() {
  remove_tmp_matches "remove scratch left in ${TMP_ROOT} by interrupted installs and tests" \
    'sima-neat-*' 'neat-apps-*' 'neat-sysroot-deps-*' 'core-scheduler-admission-*' \
    'sima-cli-artifact-install.*' 'install-neat-from-a-branch.*' 'pytest-of-*'
}

cleanup_tmp_sima_if_tmp_low_space() {
  local free_mb
  free_mb="$(free_space_mb "${TMP_ROOT}")"

  if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
    printf "[storage-cleanup] %s/sima_* cleanup skipped: unable to read free space on %s\n" "${TMP_ROOT}" "${TMP_ROOT}"
    return 1
  fi

  if (( free_mb >= TMP_FREE_SPACE_THRESHOLD_MB )); then
    printf "[storage-cleanup] %s/sima_* cleanup skipped: %s free space is %sMB (threshold %sMB)\n" "${TMP_ROOT}" "${TMP_ROOT}" "$free_mb" "$TMP_FREE_SPACE_THRESHOLD_MB"
    return 0
  fi

  printf "[storage-cleanup] %s free space is %sMB (< %sMB), removing %s/sima_*\n" "${TMP_ROOT}" "$free_mb" "$TMP_FREE_SPACE_THRESHOLD_MB" "${TMP_ROOT}"
  remove_tmp_matches "remove ${TMP_ROOT}/sima_* for low ${TMP_ROOT} free space" 'sima_*'
}

report_space() {
  local path free_mb
  for path in / "${TMP_ROOT}"; do
    free_mb="$(free_space_mb "$path")"
    if [[ "$free_mb" =~ ^[0-9]+$ ]]; then
      printf "[storage-cleanup] %s free space: %sMB\n" "$path" "$free_mb"
    else
      printf "[storage-cleanup] %s free space: unknown\n" "$path"
    fi
  done
}

report_space
cleanup_coprocessing
cleanup_tmp_simaai
cleanup_tmp_interrupted_runs
cleanup_tmp_sima_if_tmp_low_space
report_space

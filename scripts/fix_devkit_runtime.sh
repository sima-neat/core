#!/usr/bin/env bash
set +e

# Recovery script to bring a devkit out of a bad runtime state by
# restarting remote processors and related services.
TARGET_COPROCESSING_DIR="/data/simaai/coprocessing"
ROOT_FREE_SPACE_THRESHOLD_MB=500

if [[ $# -gt 0 ]]; then
  pass="$1"
else
  pass="${DEVKIT_PASSWORD:-edgeai}"
fi
run_step() {
  local label="$1"
  shift
  printf "[recovery] %s...\n" "$label"
  printf '%s\n' "$pass" | sudo -S -p '' "$@"
  local rc=$?
  printf "[recovery] %s rc=%d\n" "$label" "$rc"
  return $rc
}

cleanup_tmp_sima_if_root_low_space() {
  local free_mb
  free_mb="$(df -Pm / | awk 'NR==2 {print $4}')"

  if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
    printf "[recovery] /tmp cleanup skipped: unable to read free space on /\n"
    return 1
  fi

  if (( free_mb >= ROOT_FREE_SPACE_THRESHOLD_MB )); then
    printf "[recovery] /tmp cleanup skipped: root free space is %sMB (threshold %sMB)\n" "$free_mb" "$ROOT_FREE_SPACE_THRESHOLD_MB"
    return 0
  fi

  printf "[recovery] root free space is %sMB (< %sMB), removing /tmp/sima_*\n" "$free_mb" "$ROOT_FREE_SPACE_THRESHOLD_MB"
  run_step "remove /tmp/sima_* for low root free space" bash -c '
    shopt -s nullglob
    paths=(/tmp/sima_*)
    if (( ${#paths[@]} > 0 )); then
      rm -rf -- "${paths[@]}"
    fi
  '
}

repair_stale_global_dispatcher_lib() {
  local global_lib="/usr/lib/aarch64-linux-gnu/libneatdispatchercore.so"
  local runtime_lib="/usr/lib/aarch64-linux-gnu/neat/runtime/libneatdispatchercore.so"

  if [[ ! -e "${runtime_lib}" ]]; then
    printf "[recovery] dispatcher lib repair skipped: runtime lib not found at %s\n" "${runtime_lib}"
    return 0
  fi

  if [[ -L "${global_lib}" && "$(readlink -f "${global_lib}")" == "${runtime_lib}" ]]; then
    printf "[recovery] dispatcher lib repair skipped: %s already points at runtime lib\n" "${global_lib}"
    return 0
  fi

  if [[ -e "${global_lib}" ]]; then
    if dpkg-query -S "${global_lib}" >/dev/null 2>&1; then
      printf "[recovery] dispatcher lib repair skipped: %s is package-owned\n" "${global_lib}"
      return 0
    fi

    local backup="${global_lib}.bak-$(date -u +%Y%m%dT%H%M%SZ)"
    run_step "quarantine stale unowned ${global_lib}" mv -f "${global_lib}" "${backup}"
  fi

  if [[ ! -e "${global_lib}" ]]; then
    run_step "link ${global_lib} to packaged runtime lib" ln -s "${runtime_lib}" "${global_lib}"
  fi
}

empty_coprocessing() {
  if [[ "${TARGET_COPROCESSING_DIR}" != "/data/simaai/coprocessing" ]]; then
    printf "[recovery] empty coprocessing skipped: unexpected target %s\n" "${TARGET_COPROCESSING_DIR}"
    return 1
  fi

  if [[ ! -d "${TARGET_COPROCESSING_DIR}" ]]; then
    printf "[recovery] empty coprocessing skipped: directory not found %s\n" "${TARGET_COPROCESSING_DIR}"
    return 0
  fi

  run_step "empty coprocessing directory" find "${TARGET_COPROCESSING_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  run_step "remove /tmp/simaai-*" bash -c '
    shopt -s nullglob
    paths=(/tmp/simaai-*)
    if (( ${#paths[@]} > 0 )); then
      rm -rf -- "${paths[@]}"
    fi
  '
}

stop_runtime_services() {
  run_step "stop simaai-pipeline-manager.service" systemctl stop simaai-pipeline-manager.service
  run_step "stop rctd.service" systemctl stop rctd.service
  run_step "stop simaai-appcomplex.service" systemctl stop simaai-appcomplex.service
  run_step "terminate stale runtime processes" pkill -TERM -f '(/usr/bin/)?(mlashmcomplex|simaai_pipeline_handler_new|rctd)( |$)'
  sleep 1
  run_step "kill stale runtime processes" pkill -KILL -f '(/usr/bin/)?(mlashmcomplex|simaai_pipeline_handler_new|rctd)( |$)'
}

repair_stale_global_dispatcher_lib
stop_runtime_services
empty_coprocessing
cleanup_tmp_sima_if_root_low_space
run_step "remoteproc0 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc1 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc1 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc0 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc status" sh -c 'for rp in /sys/class/remoteproc/remoteproc0 /sys/class/remoteproc/remoteproc1; do echo "$rp: $(cat $rp/name) state=$(cat $rp/state)"; done'
run_step "init_mla_memory" /usr/bin/init_mla_memory.sh
run_step "restart simaai-appcomplex.service" systemctl restart simaai-appcomplex.service
run_step "restart simaai-pipeline-manager.service" systemctl restart simaai-pipeline-manager.service
run_step "restart rctd.service" systemctl restart rctd.service

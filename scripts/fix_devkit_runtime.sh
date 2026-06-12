#!/usr/bin/env bash
set +e

# Recovery script to bring a devkit out of a bad runtime state by
# restarting remote processors and related services.
TARGET_COPROCESSING_DIR="/data/simaai/coprocessing"
ROOT_FREE_SPACE_THRESHOLD_MB=500
EV74_FIRMWARE_INSTALLER="/usr/libexec/sima-neat-firmware/install.sh"
EV74_FIRMWARE_TARGET="/lib/firmware/modalix-cvu-fw"
EV74_FIRMWARE_SHA_FILE="/usr/share/sima-neat-firmware/modalix-cvu-fw.sha256"

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

staged_ev74_firmware_sha() {
  [[ -f "${EV74_FIRMWARE_SHA_FILE}" ]] || return 1
  tr -d '[:space:]' < "${EV74_FIRMWARE_SHA_FILE}"
}

active_ev74_firmware_sha() {
  [[ -f "${EV74_FIRMWARE_TARGET}" ]] || return 1
  sha256sum "${EV74_FIRMWARE_TARGET}" | awk '{print $1}'
}

activate_staged_ev74_firmware_if_needed() {
  local expected active rc

  if [[ ! -x "${EV74_FIRMWARE_INSTALLER}" || ! -f "${EV74_FIRMWARE_SHA_FILE}" ]]; then
    printf "[recovery] EV74 firmware activation skipped: staged firmware package not installed\n"
    return 0
  fi

  expected="$(staged_ev74_firmware_sha || true)"
  if [[ -z "${expected}" ]]; then
    printf "[recovery] EV74 firmware activation failed: unable to read staged firmware sha from %s\n" "${EV74_FIRMWARE_SHA_FILE}"
    return 1
  fi

  active="$(active_ev74_firmware_sha || true)"
  if [[ "${active}" == "${expected}" ]]; then
    printf "[recovery] EV74 firmware activation skipped: active firmware already matches staged sha=%s\n" "${expected}"
    return 0
  fi

  printf "[recovery] active EV74 firmware sha=%s does not match staged sha=%s; activating staged firmware\n" \
    "${active:-<missing>}" "${expected}"
  run_step "activate staged EV74 firmware" "${EV74_FIRMWARE_INSTALLER}" --activate
  rc=$?

  active="$(active_ev74_firmware_sha || true)"
  if [[ "${active}" == "${expected}" ]]; then
    # Compatibility with older neat-ev74-firmware packages: activation can
    # return non-zero solely because simaai-log.service is masked on CI/lab
    # devkits, after the firmware was copied and EV74 dispatch health passed.
    # Treat the SHA match as sufficient here; the rest of this recovery script
    # will restart the runtime services again before tests run.
    printf "[recovery] staged EV74 firmware is active after activation (helper rc=%d)\n" "${rc}"
    return 0
  fi

  printf "[recovery] EV74 firmware activation failed: active sha=%s expected staged sha=%s helper rc=%d\n" \
    "${active:-<missing>}" "${expected}" "${rc}"
  return 1
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
if ! activate_staged_ev74_firmware_if_needed; then
  exit 1
fi
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

#!/usr/bin/env bash
set +e

# Recovery script to bring a devkit out of a bad runtime state by
# restarting remote processors and related services.
TARGET_COPROCESSING_DIR="/data/simaai/coprocessing"
ROOT_FREE_SPACE_THRESHOLD_MB=500
EV74_FIRMWARE_INSTALLER="/usr/libexec/sima-neat-firmware/install.sh"
EV74_FIRMWARE_TARGET="/lib/firmware/modalix-cvu-fw"
EV74_FIRMWARE_SHA_FILE="/usr/share/sima-neat-firmware/modalix-cvu-fw.sha256"
DISPATCHER_GLOBAL_LIB_DIR="${NEAT_RECOVERY_DISPATCHER_GLOBAL_LIB_DIR:-/usr/lib/aarch64-linux-gnu}"
DISPATCHER_QUARANTINE_DIR="${NEAT_RECOVERY_DISPATCHER_QUARANTINE_DIR:-/var/lib/sima-neat/quarantine/dispatcher}"
RECOVERY_STEP_TIMEOUT_SECONDS="${NEAT_RECOVERY_STEP_TIMEOUT_SECONDS:-120}"
RECOVERY_STEP_KILL_GRACE_SECONDS="${NEAT_RECOVERY_STEP_KILL_GRACE_SECONDS:-10}"

if [[ $# -gt 0 ]]; then
  pass="$1"
else
  pass="${DEVKIT_PASSWORD:-edgeai}"
fi
run_step() {
  local label="$1"
  shift
  printf "[recovery] %s...\n" "$label"
  printf '%s\n' "$pass" | timeout \
    --foreground \
    --kill-after="${RECOVERY_STEP_KILL_GRACE_SECONDS}s" \
    "${RECOVERY_STEP_TIMEOUT_SECONDS}s" \
    sudo -S -p '' "$@"
  local rc=$?
  if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
    printf "[recovery] %s timed out after %ss (kill grace %ss)\n" \
      "$label" "$RECOVERY_STEP_TIMEOUT_SECONDS" "$RECOVERY_STEP_KILL_GRACE_SECONDS" >&2
  fi
  printf "[recovery] %s rc=%d\n" "$label" "$rc"
  return $rc
}

run_optional_service_step() {
  local label="$1"
  local action="$2"
  local unit="$3"
  run_step "$label" systemctl "$action" "$unit"
  local rc=$?
  if [[ "$rc" -eq 5 ]]; then
    printf "[recovery] %s skipped: systemd unit %s is not installed on this devkit image\n" \
      "$label" "$unit"
    return 0
  fi
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

quarantine_stale_global_dispatcher_libs() {
  local candidate owner destination timestamp
  local -a candidates=()

  if [[ ! -d "${DISPATCHER_GLOBAL_LIB_DIR}" ]]; then
    printf "[recovery] dispatcher quarantine skipped: loader directory not found at %s\n" \
      "${DISPATCHER_GLOBAL_LIB_DIR}"
    return 0
  fi

  mapfile -t candidates < <(
    find "${DISPATCHER_GLOBAL_LIB_DIR}" -maxdepth 1 -mindepth 1 \
      -name 'libneatdispatchercore.so*' -print | sort
  )
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    printf "[recovery] dispatcher quarantine skipped: no global dispatcher paths found\n"
    return 0
  fi

  # Refuse the complete migration before moving anything when another package
  # owns one of these paths. Its package must perform the ownership transition.
  for candidate in "${candidates[@]}"; do
    owner="$(dpkg-query -S "${candidate}" 2>/dev/null || true)"
    if [[ -n "${owner}" ]]; then
      printf "[recovery] refusing package-owned global dispatcher path %s: %s\n" \
        "${candidate}" "${owner}" >&2
      return 1
    fi
  done

  run_step "create dispatcher quarantine" mkdir -p "${DISPATCHER_QUARANTINE_DIR}" || return 1
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for candidate in "${candidates[@]}"; do
    destination="${DISPATCHER_QUARANTINE_DIR}/$(basename "${candidate}")"
    if [[ -e "${destination}" || -L "${destination}" ]]; then
      destination="${destination}.${timestamp}.$$"
    fi
    run_step "quarantine stale unowned ${candidate}" \
      mv -f -- "${candidate}" "${destination}" || return 1
  done

  run_step "refresh dynamic loader cache" ldconfig || return 1
  printf "[recovery] quarantined %d global dispatcher path(s); no global alias was created\n" \
    "${#candidates[@]}"
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
  run_optional_service_step \
    "stop simaai-pipeline-manager.service" stop simaai-pipeline-manager.service
  run_optional_service_step "stop rctd.service" stop rctd.service
  run_optional_service_step "stop simaai-appcomplex.service" stop simaai-appcomplex.service
  run_step "terminate stale runtime processes" pkill -TERM -f '(/usr/bin/)?(mlashmcomplex|simaai_pipeline_handler_new|rctd)( |$)'
  sleep 1
  run_step "kill stale runtime processes" pkill -KILL -f '(/usr/bin/)?(mlashmcomplex|simaai_pipeline_handler_new|rctd)( |$)'
}

if [[ "${NEAT_RECOVERY_FUNCTIONS_ONLY:-OFF}" == "ON" ]]; then
  return 0 2>/dev/null || exit 0
fi

if ! quarantine_stale_global_dispatcher_libs; then
  exit 1
fi
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
run_optional_service_step "restart simaai-appcomplex.service" restart simaai-appcomplex.service
run_optional_service_step \
  "restart simaai-pipeline-manager.service" restart simaai-pipeline-manager.service
run_optional_service_step "restart rctd.service" restart rctd.service

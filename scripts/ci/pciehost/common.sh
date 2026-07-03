#!/usr/bin/env bash

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
WORK_DIR="${SIMAPCIE_WORK_DIR:-${WORKSPACE}/_work/pciehost-hardware}"
PACKAGE_DIR="${SIMAPCIE_PACKAGE_DIR:-${WORK_DIR}/package}"
EXTRAS_DIR="${SIMAPCIE_EXTRAS_DIR:-${WORK_DIR}/extras}"
ARTIFACT_DIR="${SIMAPCIE_ARTIFACT_DIR:-${WORKSPACE}/_work/pciehost-artifacts}"
STATE_FILE="${WORK_DIR}/state.env"
ASSET_ENV_FILE="${WORK_DIR}/assets.env"

VULCAN_ENV="${VULCAN_ENV:-production}"
REF_NAME="${REF_NAME:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}}"
SHORT_SHA="${SHORT_SHA:-${GITHUB_SHA:-}}"
SHORT_SHA="${SHORT_SHA:0:12}"

CARD_HOST="${SIMAPCIE_CARD_HOST:-10.0.0.2}"
CARD_USER="${SIMAPCIE_USER:-sima}"
CARD_ID="${SIMAPCIE_CARD_ID:-0}"
QUEUE="${SIMAPCIE_QUEUE:-0}"
STRESS_QUEUES="${SIMAPCIE_STRESS_QUEUES:-0 1 2 3}"
SYNC_ITERATIONS="${SIMAPCIE_SYNC_ITERATIONS:-50}"
TEST_ITERATIONS="${SIMAPCIE_TEST_ITERATIONS:-1000}"
STRESS_ITERATIONS="${SIMAPCIE_STRESS_ITERATIONS:-${TEST_ITERATIONS}}"
READINESS_TIMEOUT_MS="${SIMAPCIE_READINESS_TIMEOUT_MS:-180000}"
PULL_TIMEOUT_MS="${SIMAPCIE_PULL_TIMEOUT_MS:-30000}"

shell_quote() {
  printf '%q' "$1"
}

write_state_var() {
  local name="$1"
  local value="$2"

  mkdir -p "$(dirname "${STATE_FILE}")"
  grep -v "^${name}=" "${STATE_FILE}" 2>/dev/null >"${STATE_FILE}.tmp" || true
  printf '%s=%q\n' "${name}" "${value}" >>"${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "${STATE_FILE}"
}

load_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
  fi
  if [[ -f "${ASSET_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ASSET_ENV_FILE}"
    export SIMAPCIE_YOLOV8_MODEL SIMAPCIE_TEST_IMAGE SIMAPCIE_BOXDECODE_IMAGE
  fi
}

sanitize_path() {
  local clean_path
  clean_path="$(
    printf '%s' "${PATH}" \
      | tr ':' '\n' \
      | grep -viE '(^|/)(conda|mambaforge|miniconda|anaconda)(/|$)' \
      | paste -sd: -
  )"
  export PATH="${clean_path}"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
}

run_host_sudo() {
  local password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo is required to run: $*" >&2
    exit 1
  elif sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [[ -n "${password}" ]]; then
    printf '%s\n' "${password}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

host_sudo_is_available() {
  local password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
  [[ "$(id -u)" -eq 0 ]] || sudo -n true 2>/dev/null || [[ -n "${password}" ]]
}

run_host_sudo_if_available() {
  if host_sudo_is_available; then
    run_host_sudo "$@"
  else
    "$@"
  fi
}

ssh_card() {
  ssh \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=accept-new \
    "${CARD_USER}@${CARD_HOST}" "$@"
}

remote_sudo_wrapper_script() {
  cat <<'REMOTE_SUDO'
setup_remote_sudo_wrapper() {
  local password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
  if [[ -z "${password}" || ! -x /usr/bin/sudo ]]; then
    return 0
  fi

  REMOTE_SUDO_WRAPPER_DIR="$(mktemp -d /tmp/sima-neat-sudo-wrapper.XXXXXX)"
  export REMOTE_SUDO_WRAPPER_DIR
  cat > "${REMOTE_SUDO_WRAPPER_DIR}/sudo" <<'SUDO_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
real_sudo="/usr/bin/sudo"
password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
if "${real_sudo}" -n true 2>/dev/null; then
  exec "${real_sudo}" "$@"
fi
if [[ -z "${password}" ]]; then
  exec "${real_sudo}" "$@"
fi
printf '%s\n' "${password}" | "${real_sudo}" -S -p '' "$@"
SUDO_WRAPPER
  chmod 0700 "${REMOTE_SUDO_WRAPPER_DIR}/sudo"
  export PATH="${REMOTE_SUDO_WRAPPER_DIR}:${PATH}"
}
REMOTE_SUDO
}

cleanup_host_pcie_device() {
  local dev="/dev/sima_mla_c${CARD_ID}"
  local known_pattern='[t]est_tensor_run|[t]est_image_run|[t]est_image_boxdecode_run|[t]est_queue_blocker|[g]st-launch-1.0'

  if [[ ! -e "${dev}" ]]; then
    return 0
  fi

  echo "Host PCIe device owners before cleanup: ${dev}"
  if command -v fuser >/dev/null 2>&1; then
    run_host_sudo_if_available fuser -v "${dev}" || true
  else
    echo "WARN: fuser is not available; host PCIe owner cleanup will use process names only." >&2
  fi

  if command -v pgrep >/dev/null 2>&1; then
    mapfile -t host_pids < <(
      for pid in $(pgrep -f "${known_pattern}" || true); do
        [[ "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        first_arg="${cmdline%% *}"
        base="$(basename "${first_arg}")"
        case "${base}" in
          test_tensor_run|test_image_run|test_image_boxdecode_run|test_queue_blocker|gst-launch-1.0)
            printf '%s\n' "${pid}"
            ;;
        esac
      done
    )
    if [[ "${#host_pids[@]}" -gt 0 ]]; then
      echo "Stopping known host PCIe hardware test processes: ${host_pids[*]}"
      kill "${host_pids[@]}" 2>/dev/null || true
      sleep 1
      mapfile -t host_pids < <(
        for pid in "${host_pids[@]}"; do
          kill -0 "${pid}" 2>/dev/null && printf '%s\n' "${pid}"
        done
      )
      if [[ "${#host_pids[@]}" -gt 0 ]]; then
        echo "Force-stopping known host PCIe hardware test processes: ${host_pids[*]}"
        kill -9 "${host_pids[@]}" 2>/dev/null || true
      fi
    fi
  fi

  if command -v fuser >/dev/null 2>&1 &&
    run_host_sudo_if_available fuser "${dev}" >/dev/null 2>&1; then
    echo "Host PCIe device still busy after known-process cleanup; killing remaining owners: ${dev}"
    run_host_sudo_if_available fuser -k -TERM "${dev}" || true
    sleep 1
    if run_host_sudo_if_available fuser "${dev}" >/dev/null 2>&1; then
      run_host_sudo_if_available fuser -k "${dev}" || true
    fi
  fi
}

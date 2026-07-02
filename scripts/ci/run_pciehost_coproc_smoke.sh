#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
WORK_DIR="${SIMAPCIE_WORK_DIR:-${WORKSPACE}/_work/pciehost-coproc}"
PACKAGE_DIR="${SIMAPCIE_PACKAGE_DIR:-${WORK_DIR}/package}"
EXTRAS_DIR="${SIMAPCIE_EXTRAS_DIR:-${WORK_DIR}/extras}"
ARTIFACT_DIR="${SIMAPCIE_ARTIFACT_DIR:-${WORKSPACE}/_work/pciehost-artifacts}"

VULCAN_ENV="${VULCAN_ENV:-production}"
REF_NAME="${REF_NAME:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}}"
SHORT_SHA="${SHORT_SHA:-${GITHUB_SHA:-}}"
SHORT_SHA="${SHORT_SHA:0:12}"

CARD_HOST="${SIMAPCIE_CARD_HOST:-10.0.0.2}"
CARD_USER="${SIMAPCIE_USER:-sima}"
CARD_ID="${SIMAPCIE_CARD_ID:-0}"
QUEUE="${SIMAPCIE_QUEUE:-0}"
READINESS_TIMEOUT_MS="${SIMAPCIE_READINESS_TIMEOUT_MS:-180000}"
PULL_TIMEOUT_MS="${SIMAPCIE_PULL_TIMEOUT_MS:-30000}"
MODE="run"

usage() {
  cat <<'EOF'
Usage: run_pciehost_coproc_smoke.sh

Installs the exact published PCIe host package, refreshes the card runtime, and
runs the PCIe host hardware CTest smoke suite on the host/card runner.

Important environment:
  VULCAN_ENV                 Vulcan artifact env, e.g. dev/stg/production
  REF_NAME                   Branch/tag name (default: GitHub ref)
  SHORT_SHA                  12-char commit folder (default: GITHUB_SHA prefix)
  SIMAPCIE_CARD_HOST         Card management IP (default: 10.0.0.2)
  SIMAPCIE_USER              Card SSH user (default: sima)
  DEVKIT_PASSWORD            Optional sudo password used by installers
  SUDO_PASSWORD              Optional host sudo password; sudo may prompt if unset
  SIMAPCIE_CARD_INSTALL_CMD  Optional command to install card runtime
  SIMAPCIE_CARD_INSTALL_DIR  Optional card-side working dir for sima-cli install
  SIMAPCIE_CARD_GST_DEBUG    Optional card-side GST debug string
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--cleanup-only" ]]; then
  MODE="cleanup-only"
elif [[ $# -gt 0 ]]; then
  echo "ERROR: unknown option: $1" >&2
  usage >&2
  exit 1
fi

if [[ "${MODE}" != "cleanup-only" && ( -z "${REF_NAME}" || -z "${SHORT_SHA}" ) ]]; then
  echo "ERROR: REF_NAME and SHORT_SHA/GITHUB_SHA are required." >&2
  exit 1
fi

shell_quote() {
  printf '%q' "$1"
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
    mapfile -t host_pids < <(pgrep -f "${known_pattern}" || true)
    if [[ "${#host_pids[@]}" -gt 0 ]]; then
      echo "Stopping known host PCIe smoke processes: ${host_pids[*]}"
      kill "${host_pids[@]}" 2>/dev/null || true
      sleep 1
      mapfile -t host_pids < <(pgrep -f "${known_pattern}" || true)
      if [[ "${#host_pids[@]}" -gt 0 ]]; then
        echo "Force-stopping known host PCIe smoke processes: ${host_pids[*]}"
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

preflight() {
  sanitize_path
  require_cmd sima-cli
  require_cmd /usr/bin/gst-inspect-1.0
  require_cmd ctest
  require_cmd ssh
  require_cmd tar
  require_cmd find

  rm -rf "${WORK_DIR}"
  mkdir -p "${PACKAGE_DIR}" "${EXTRAS_DIR}" "${ARTIFACT_DIR}"

  ping -c 1 -W 5 "${CARD_HOST}" >/dev/null
}

install_host_package() {
  local package_spec="core/pciehost/amd64@${REF_NAME}:${SHORT_SHA}"
  echo "Installing PCIe host package ${package_spec} from Vulcan env ${VULCAN_ENV}"
  SIMA_CLI_CHECK_FOR_UPDATE=0 \
    sima-cli neat install \
      --env "${VULCAN_ENV}" \
      --install-dir "${PACKAGE_DIR}" \
      "${package_spec}"

  /usr/bin/gst-inspect-1.0 neatpciehost >/dev/null
}

install_host_test_runtime_deps() {
  if ldconfig -p 2>/dev/null | grep -q 'libopencv_imgcodecs'; then
    return
  fi

  echo "Installing PCIe hardware test runtime dependencies"
  run_host_sudo apt-get update
  run_host_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev
}

ensure_card_sima_cli() {
  if ssh_card 'export PATH="/data/sima-cli/.venv/bin:${HOME}/.sima-cli/.venv/bin:${PATH}"; command -v sima-cli >/dev/null 2>&1'; then
    return
  fi

  local installer="${WORKSPACE}/scripts/ci/install_sima_cli_main.sh"
  if [[ ! -f "${installer}" ]]; then
    echo "ERROR: unable to bootstrap sima-cli on card; missing ${installer}" >&2
    exit 1
  fi

  echo "Installing sima-cli on PCIe card"
  local remote_installer="/tmp/sima-neat-install-sima-cli-main.sh"
  ssh_card "cat > $(shell_quote "${remote_installer}")" < "${installer}"
  ssh_card \
    "REMOTE_INSTALLER=$(shell_quote "${remote_installer}") \
     DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     SUDO_PASSWORD=$(shell_quote "${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}") \
     bash -s" <<REMOTE_BOOTSTRAP
set -euo pipefail
$(remote_sudo_wrapper_script)
setup_remote_sudo_wrapper
trap 'rm -rf "\${REMOTE_SUDO_WRAPPER_DIR:-}" "\${REMOTE_INSTALLER}"' EXIT
bash "\${REMOTE_INSTALLER}"
REMOTE_BOOTSTRAP
}

install_card_runtime() {
  local package_spec="core@${REF_NAME}:${SHORT_SHA}"
  if [[ -n "${SIMAPCIE_CARD_INSTALL_CMD:-}" ]]; then
    echo "Installing card runtime with SIMAPCIE_CARD_INSTALL_CMD"
    export VULCAN_ENV REF_NAME SHORT_SHA CARD_HOST CARD_USER package_spec
    export DEVKIT_PASSWORD="${DEVKIT_PASSWORD:-}"
    export SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD}}"
    bash -lc "${SIMAPCIE_CARD_INSTALL_CMD}"
    return
  fi

  ensure_card_sima_cli

  echo "Installing card runtime ${package_spec} on ${CARD_USER}@${CARD_HOST} from Vulcan env ${VULCAN_ENV}"
  ssh_card \
    "REMOTE_VULCAN_ENV=$(shell_quote "${VULCAN_ENV}") \
     REMOTE_PACKAGE_SPEC=$(shell_quote "${package_spec}") \
     REMOTE_CARD_INSTALL_DIR=$(shell_quote "${SIMAPCIE_CARD_INSTALL_DIR:-}") \
     REMOTE_DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     REMOTE_SUDO_PASSWORD=$(shell_quote "${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}") \
     bash -s" <<'REMOTE_INSTALL'
set -euo pipefail

export PATH="/data/sima-cli/.venv/bin:${HOME}/.sima-cli/.venv/bin:${PATH}"
if ! command -v sima-cli >/dev/null 2>&1; then
  echo "ERROR: sima-cli is not installed on the PCIe card after bootstrap." >&2
  exit 1
fi

card_install_dir="${REMOTE_CARD_INSTALL_DIR}"
if [[ -z "${card_install_dir}" ]]; then
  if [[ -d /workspace ]]; then
    card_install_dir="/workspace"
  else
    card_install_dir="${HOME:-/tmp}/sima-neat-card-install"
  fi
fi
mkdir -p "${card_install_dir}"

export SIMA_CLI_CHECK_FOR_UPDATE=0
export DEVKIT_PASSWORD="${REMOTE_DEVKIT_PASSWORD}"
export SUDO_PASSWORD="${REMOTE_SUDO_PASSWORD}"
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
setup_remote_sudo_wrapper
trap 'rm -rf "${REMOTE_SUDO_WRAPPER_DIR:-}"' EXIT
cd "${card_install_dir}"
sima-cli install --neat --env "${REMOTE_VULCAN_ENV}" "${REMOTE_PACKAGE_SPEC}" -t all
REMOTE_INSTALL
}

extract_extras() {
  local extras_tar
  mapfile -t extras_archives < <(
    find "${PACKAGE_DIR}" "${WORKSPACE}" -type f \
      -name 'sima-pcie-host-*-Linux-amd64-extras.tar.gz' \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk '{print $2}'
  )
  if [[ "${#extras_archives[@]}" -eq 0 ]]; then
    echo "ERROR: unable to locate PCIe host extras archive after install." >&2
    find "${PACKAGE_DIR}" -maxdepth 3 -type f -printf '  %p\n' >&2 || true
    exit 1
  fi

  extras_tar="${extras_archives[0]}"
  echo "Extracting ${extras_tar}"
  rm -rf "${EXTRAS_DIR}"
  mkdir -p "${EXTRAS_DIR}"
  tar -xzf "${extras_tar}" -C "${EXTRAS_DIR}"

  mapfile -t ctest_files < <(
    find "${EXTRAS_DIR}" -type f -name CTestTestfile.cmake \
      -path '*/tests/CTestTestfile.cmake' \
      -print | sort
  )
  if [[ "${#ctest_files[@]}" -eq 0 ]]; then
    echo "ERROR: unable to locate PCIe host hardware CTest files in extras." >&2
    find "${EXTRAS_DIR}" -maxdepth 5 -type f -printf '  %p\n' >&2 || true
    exit 1
  fi

  TEST_DIR="$(dirname "${ctest_files[0]}")"
  export TEST_DIR
  echo "Using PCIe host hardware test dir: ${TEST_DIR}"
}

resolve_assets() {
  local env_file="${WORK_DIR}/pciehost-test-assets.env"
  "${WORKSPACE}/pcie_host/scripts/resolve_hardware_test_assets.sh" \
    --workspace "${WORKSPACE}" \
    --extras-root "${EXTRAS_DIR}" \
    --cache-dir "${WORK_DIR}/assets" \
    --env-file "${env_file}"
  # shellcheck disable=SC1090
  source "${env_file}"
  export SIMAPCIE_YOLOV8_MODEL SIMAPCIE_TEST_IMAGE SIMAPCIE_BOXDECODE_IMAGE
}

verify_card_plugins() {
  ssh_card "set -e; \
    command -v pcie-pipeline-builder >/dev/null; \
    gst-inspect-1.0 neatpciesrc >/dev/null; \
    gst-inspect-1.0 neatpciesink >/dev/null"
}

cleanup_queue() {
  cleanup_host_pcie_device

  ssh_card "REMOTE_QUEUE=$(shell_quote "${QUEUE}") bash -s" <<'REMOTE_CLEANUP' || true
set -euo pipefail

pattern="[p]cie-pipeline-builder.*--queue([ =])?${REMOTE_QUEUE}([^0-9]|$)"
if command -v pgrep >/dev/null 2>&1; then
  mapfile -t pids < <(pgrep -f "${pattern}" || true)
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    sleep 1
    mapfile -t pids < <(pgrep -f "${pattern}" || true)
    if [[ "${#pids[@]}" -gt 0 ]]; then
      kill -9 "${pids[@]}" 2>/dev/null || true
    fi
  fi
fi

rm -f \
  "/run/sima-neat/pcie/q${REMOTE_QUEUE}.pid" \
  "/run/sima-neat/pcie/q${REMOTE_QUEUE}.status" \
  "/tmp/q${REMOTE_QUEUE}-card.gst.log" \
  2>/dev/null || true
REMOTE_CLEANUP
}

run_ctest() {
  export SIMAPCIE_CARD_HOST="${CARD_HOST}"
  export SIMAPCIE_USER="${CARD_USER}"
  export SIMAPCIE_CARD_ID="${CARD_ID}"
  export SIMAPCIE_QUEUE="${QUEUE}"
  export SIMAPCIE_READINESS_TIMEOUT_MS="${READINESS_TIMEOUT_MS}"
  export SIMAPCIE_PULL_TIMEOUT_MS="${PULL_TIMEOUT_MS}"
  export SIMAPCIE_CARD_GST_DEBUG="${SIMAPCIE_CARD_GST_DEBUG:-}"
  export SIMAPCIE_CARD_GST_DEBUG_FILE="${SIMAPCIE_CARD_GST_DEBUG_FILE:-/tmp/q${QUEUE}-card.gst.log}"

  ctest \
    --test-dir "${TEST_DIR}" \
    --output-on-failure \
    --no-tests=error
}

main() {
  trap cleanup_queue EXIT
  preflight
  install_host_package
  install_host_test_runtime_deps
  cleanup_queue
  install_card_runtime
  extract_extras
  resolve_assets
  verify_card_plugins
  cleanup_queue
  run_ctest
}

if [[ "${MODE}" == "cleanup-only" ]]; then
  cleanup_queue
  exit 0
fi

main "$@"

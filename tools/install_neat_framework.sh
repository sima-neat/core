#!/usr/bin/env bash
set -euo pipefail

# install_neat_framework.sh
#
# Purpose:
# - Install SiMa NEAT wheel into a Python virtual environment (board mode only).
# - Install NEAT runtime .deb packages on a Modalix board, natively in the ROS2
#   SDK, or into an eLxr SDK sysroot.
#
# Behavior:
# - Auto-detects environment:
#   - ROS2 SDK when /etc/sdk-release reports SDK Type=ros2-sdk.
#   - eLxr SDK when /etc/sdk-release reports SDK Version, or SYSROOT points to a
#     valid directory.
#   - Modalix board when /etc/buildinfo reports MACHINE=modalix.
# - In Modalix board mode, installs .deb packages with apt (sudo).
# - In ROS2 SDK mode, installs .deb packages into native system paths with apt
#   without running board lifecycle or firmware operations.
# - In eLxr SDK mode, caches install artifacts under the sysroot
#   neat-install-packages folder for paired DevKit sync.
# - In eLxr SDK mode, exposes the sysroot-installed neat command at
#   /usr/local/bin/neat for convenient use inside the SDK container.
# - In Modalix board mode, installs python3-venv with apt when python3 lacks
#   ensurepip (Platform 3.0.0 images ship without it).
# - On devices with writable /media/nvme, creates the venv at /media/nvme/pyneat and
#   exposes it via $HOME/pyneat for consistent activation instructions.
# - On devices without /media/nvme, creates the venv directly at $HOME/pyneat.
#
# Expected working directory:
# - Directory containing:
#   - one .whl file
#   - the Debian packages delivered by the selected Internals, LLiMa, and Core artifacts
#   - neat-install-manifest.txt when installed from a packaged release
#
# Environment overrides:
# - FORCE_INSTALL: set to 1 to skip platform compatibility checks and request the
#   EV74 kernel-build compatibility override, including on a paired DevKit.
#   Set by sima-cli neat install -f; other values do not force. Requires an
#   override-aware firmware package; does not establish runtime compatibility.
#   Firmware integrity, ownership, DMA safety, and package safeguards still apply.
# - PYNEAT_VENV_DIR: Python virtualenv path
# - SUDO_PASSWORD / DEVKIT_PASSWORD: sudo password (preferred non-interactive override)
#   Also passed to the Sentinel installer.
# - DEFAULT_SUDO_PASSWORD: fallback password (default: edgeai)
# - SYSROOT: SDK sysroot path override (default: /opt/toolchain/aarch64/modalix)
# - ELXR_SDK_RELEASE_FILE: SDK release metadata path (default: /etc/sdk-release)
# - NEAT_OS_RELEASE_FILE: operating-system metadata path (default: /etc/os-release)
# - NEAT_BUILDINFO_FILE: DevKit build metadata path (default: /etc/buildinfo)
# - NEAT_PACKAGE_MANIFEST: package manifest filename/path (default: manifest.json)
# - NEAT_INSTALLER_SKIP_PLATFORM_CHECK: ON/OFF (default: OFF) explicit escape hatch
#   for development installs on nonstandard images; does not override EV74 checks.
# - DEVKIT_SYNC_DEVKIT_IP: paired DevKit IP for SDK->DevKit artifact sync
# - DEVKIT_DEPLOY_USER: DevKit SSH user (default: sima)
# - DEVKIT_SYNC_REQUIRED: ON/OFF (default: ON) fail hard if paired DevKit sync fails
# - NEAT_INSTALLER_SKIP_DEVKIT_SYNC: ON/OFF (default: OFF) skip SDK->DevKit sync
# - NEAT_INSTALL_MANIFEST: install manifest filename (default: neat-install-manifest.txt)
# - CODEX_HOME: optional Codex home override for skill install target
# - CLAUDE_HOME: optional Claude home override for skill install target
# - NEAT_INSTALLER_INSTALL_CODEX_SKILL: ON/OFF (default: ON)
# - NEAT_INSTALLER_INSTALL_CLAUDE_SKILL: ON/OFF (default: ON)
# - NEAT_INSTALLER_ALLOW_DPKG_FALLBACK: ON/OFF (default: OFF) allow direct
#   dpkg fallback after apt-get has had a chance to resolve dependencies.
# - NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL: ON/OFF (default: OFF) allow the
#   destructive remove-and-retry path after apt rejects the local package set.
#   Keep OFF for normal installs so an incompatible bundle cannot remove the
#   board's currently working runtime and transitive platform packages.
# - NEAT_INSTALLER_APT_UPDATE: AUTO/ON/OFF (default: AUTO) controls whether the
#   board installer refreshes APT metadata before installing local DEBs. AUTO
#   refreshes only when /var/lib/apt/lists has no package index files.
# - NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD: ON/OFF (default: ON) activate
#   staged EV74 firmware from the selected package set after board replacement
#   and keep it selected across reboots.
# - NEAT_INSTALLER_SENTINEL_REQUIRED: ON/OFF (default: OFF) fail the board
#   install when Sentinel, the telemetry daemon Model::benchmark() reads,
#   cannot be installed. OFF warns and continues; CI sets ON.

SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"
DEVKIT_DEPLOY_USER="${DEVKIT_DEPLOY_USER:-sima}"
DEVKIT_SYNC_REQUIRED="${DEVKIT_SYNC_REQUIRED:-ON}"
NEAT_INSTALLER_SKIP_DEVKIT_SYNC="${NEAT_INSTALLER_SKIP_DEVKIT_SYNC:-OFF}"
NEAT_INSTALL_MANIFEST="${NEAT_INSTALL_MANIFEST:-neat-install-manifest.txt}"
NEAT_INSTALLER_INSTALL_CODEX_SKILL="${NEAT_INSTALLER_INSTALL_CODEX_SKILL:-ON}"
NEAT_INSTALLER_INSTALL_CLAUDE_SKILL="${NEAT_INSTALLER_INSTALL_CLAUDE_SKILL:-ON}"
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK="${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK:-OFF}"
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL="${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL:-OFF}"
NEAT_INSTALLER_APT_UPDATE="${NEAT_INSTALLER_APT_UPDATE:-AUTO}"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD="${NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD:-ON}"
NEAT_INSTALLER_SENTINEL_REQUIRED="${NEAT_INSTALLER_SENTINEL_REQUIRED:-OFF}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
NEAT_OS_RELEASE_FILE="${NEAT_OS_RELEASE_FILE:-/etc/os-release}"
NEAT_BUILDINFO_FILE="${NEAT_BUILDINFO_FILE:-/etc/buildinfo}"
NEAT_PACKAGE_MANIFEST="${NEAT_PACKAGE_MANIFEST:-manifest.json}"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK="${NEAT_INSTALLER_SKIP_PLATFORM_CHECK:-OFF}"
INSTALLER_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
# Installs the Sentinel telemetry daemon on boards; ships next to this installer.
SENTINEL_INSTALLER_PATH="$(dirname "${INSTALLER_SCRIPT_PATH}")/install_sentinel.sh"
SCHEDULER_PLATFORM_ADMISSION_PATH="$(dirname "${INSTALLER_SCRIPT_PATH}")/scheduler_platform_admission.py"
BOARD_PLATFORM_ADMISSION_DIR=""
GREEN=$'\033[0;32m'
RESET=$'\033[0m'
INSTALLER_START_DIR="${PWD}"
INSTALLER_TMP_DIRS=()

cleanup_installer_tmp_dirs() {
  local dir
  cd -- "${INSTALLER_START_DIR}" 2>/dev/null || true
  for dir in "${INSTALLER_TMP_DIRS[@]}"; do
    [[ -n "${dir}" && ( -e "${dir}" || -L "${dir}" ) ]] && rm -rf -- "${dir}"
  done
}
trap cleanup_installer_tmp_dirs EXIT

usage() {
  cat <<'EOF'
Usage: install_neat_framework.sh [--local] [-h|--help]

Options:
  --local      Install only into the current environment from artifacts in cwd.
               Disables paired SDK->DevKit sync behavior.
  -h, --help   Show this help.

Environment:
  FORCE_INSTALL=1  Skip platform compatibility checks and request the EV74
                   kernel-build compatibility override (also on a paired DevKit).
                   Set automatically by sima-cli neat install core -f.
                   Requires an override-aware firmware package. A mismatched
                   runtime may be unsafe; force does not establish compatibility.
                   Firmware integrity, ownership, DMA safety, and package
                   dependency/removal checks still apply.

Example:
  FORCE_INSTALL=1 bash install_neat_framework.sh --local
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local)
        NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

log() {
  printf '[install_neat_framework] %s\n' "$*"
}

relation_field_has_package() {
  local field="$1"
  local expected_package="$2"
  local relation package

  while IFS= read -r relation; do
    relation="${relation#"${relation%%[![:space:]]*}"}"
    relation="${relation%"${relation##*[![:space:]]}"}"
    package="${relation%%[[:space:](]*}"
    package="${package%%:*}"
    if [[ "${package}" == "${expected_package}" ]]; then
      return 0
    fi
  done < <(printf '%s\n' "${field}" | tr ',' '\n')
  return 1
}

relation_field_provides_exact_version() {
  local field="$1"
  local expected_package="$2"
  local expected_version="$3"
  local relation

  while IFS= read -r relation; do
    relation="$(printf '%s' "${relation}" |
      sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/ /g')"
    if [[ "${relation}" == "${expected_package} (= ${expected_version})" ]]; then
      return 0
    fi
  done < <(printf '%s\n' "${field}" | tr ',' '\n')
  return 1
}

relation_field_matches_version() {
  local field="$1" expected_package="$2" installed_version="$3"
  local relation
  local pattern="^${expected_package} \((<<|<=|=|>=|>>) ([^[:space:]()]+)\)$"
  while IFS= read -r relation; do
    relation="$(printf '%s' "${relation}" |
      sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/ /g')"
    if [[ "${relation}" =~ ${pattern} ]] &&
       dpkg --compare-versions "${installed_version}" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"; then
      return 0
    fi
  done < <(printf '%s\n' "${field}" | tr ',' '\n')
  return 1
}

find_verified_bundled_replacement() {
  local removed_package="$1"
  local installed_version="$2"
  shift 2
  local deb_path provides replaces conflicts

  for deb_path in "$@"; do
    [[ -f "${deb_path}" ]] || continue
    provides="$(dpkg-deb -f "${deb_path}" Provides 2>/dev/null || true)"
    replaces="$(dpkg-deb -f "${deb_path}" Replaces 2>/dev/null || true)"
    conflicts="$(dpkg-deb -f "${deb_path}" Conflicts 2>/dev/null || true)"
    if relation_field_provides_exact_version \
         "${provides}" "${removed_package}" "${installed_version}" &&
       relation_field_has_package "${replaces}" "${removed_package}" &&
       relation_field_has_package "${conflicts}" "${removed_package}"; then
      basename "${deb_path}"
      return 0
    fi
  done
  return 1
}

# Scheduler declares retirement of the monolithic runtime/plugin files. Runner
# supplies execution in the same cohort, not ownership of Scheduler's old files.
# Prove bounded retirement without claiming compatibility with the legacy API.
find_verified_runner_retirement() {
  local removed_package="$1" installed_version="$2"
  shift 2
  case "${removed_package}" in neat-runtime|neat-gst-plugins) ;; *) return 1 ;; esac
  python3 - "$(resolve_package_manifest_path)" <<'PYRETIRE' || return 1
import json, re, sys
try:
    with open(sys.argv[1]) as stream:
        version = json.load(stream).get("platform-version", "")
    raise SystemExit(0 if re.fullmatch(r"3[.]0[.][0-9]+", str(version)) else 1)
except (OSError, ValueError):
    raise SystemExit(1)
PYRETIRE
  dpkg --compare-versions "${installed_version}" lt '0.6.1~' || return 1
  local deb field relations expected role runner="" scheduler="" admitted
  local legacy_expected=""
  expected="${removed_package} (<< 0.6.1~)"
  # Preserve previously admitted local cohorts with the older, narrower bound.
  if dpkg --compare-versions "${installed_version}" lt '0.6~'; then
    legacy_expected="${removed_package} (<< 0.6~)"
  fi
  for deb in "$@"; do
    [[ -f "${deb}" ]] || continue
    role="$(dpkg-deb -f "${deb}" Package 2>/dev/null)"
    case "${role}" in
      simaai-runner)
        [[ -z "${runner}" ]] || return 1
        runner="${deb}"
        continue
        ;;
      simaai-scheduler) ;;
      *) continue ;;
    esac
    admitted=1
    for field in Breaks Replaces; do
      relations="$(dpkg-deb -f "${deb}" "${field}" 2>/dev/null | tr ',' '\n' |
        sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/ /g')"
      if ! grep -Fxq -- "${expected}" <<< "${relations}" &&
         ! { [[ -n "${legacy_expected}" ]] && grep -Fxq -- "${legacy_expected}" <<< "${relations}"; }; then
        admitted=0
        break
      fi
    done
    if [[ "${admitted}" == 1 ]]; then
      [[ -z "${scheduler}" ]] || return 1
      scheduler="${deb}"
    fi
  done
  # The complete runtime still requires both real providers. Authenticated
  # cohort and transaction admission separately bind exact dependencies and
  # every transferred file; Runner need not retire an upper package to install.
  if [[ "${removed_package}" == neat-runtime && -n "${runner}" && -n "${scheduler}" ]]; then
    printf '%s + %s\n' "$(basename "${runner}")" "$(basename "${scheduler}")"
    return 0
  fi
  if [[ "${removed_package}" == neat-gst-plugins && -n "${scheduler}" ]]; then
    basename "${scheduler}"
    return 0
  fi
  return 1
}

# Firmware replacement is retirement, not compatibility with the legacy image.
# Admit only the selected authenticated native owner and its bounded relations;
# Scheduler separately checks the full transaction and transferred file owners.
find_verified_firmware_retirement() {
  local removed_package="$1" installed_version="$2"
  shift 2
  [[ "${removed_package}" == neat-ev74-firmware ]] || return 1
  local required_control deb replaces breaks conflicts identity archive_sha
  required_control="$(required_board_firmware_control)" || return 1
  [[ -n "${required_control}" ]] || return 1
  for deb in "$@"; do
    [[ -f "${deb}" ]] || continue
    archive_sha="$(sha256sum "${deb}")" || return 1
    printf -v identity '%s\t%s\t%s\t%s' \
      "$(dpkg-deb -f "${deb}" Package)" "$(dpkg-deb -f "${deb}" Version)" \
      "$(dpkg-deb -f "${deb}" Architecture)" "${archive_sha%% *}"
    [[ "${identity}" == "${required_control}" ]] || continue
    replaces="$(dpkg-deb -f "${deb}" Replaces)" || return 1
    breaks="$(dpkg-deb -f "${deb}" Breaks)" || return 1
    conflicts="$(dpkg-deb -f "${deb}" Conflicts)" || return 1
    relation_field_matches_version "${replaces}" "${removed_package}" "${installed_version}" || return 1
    if ! relation_field_matches_version "${breaks}" "${removed_package}" "${installed_version}" &&
       ! relation_field_matches_version "${conflicts}" "${removed_package}" "${installed_version}"; then
      return 1
    fi
    basename "${deb}"
    return 0
  done
  return 1
}

verify_simulated_package_removals() {
  local simulation_log="$1"
  shift
  local -a install_specs=("$@")
  local -a removed_packages=()
  local -a verified_replacements=()
  local package package_name installed_version replacement_deb

  mapfile -t removed_packages < <(awk '$1 == "Remv" {print $2}' "${simulation_log}")
  for package in "${removed_packages[@]}"; do
    package_name="${package%%:*}"
    case "${package_name}" in
      sima-neat | sima-neat-dev | neat-internals-dev | sima-lmm-dev)
        continue
        ;;
      neat-libcamera | neat-libcamera-dev | neat-libcamera-tools)
        # Retired recovery packages: the platform owns libcamera since
        # internals#190, so APT removing them is the intended retirement.
        continue
        ;;
    esac

    installed_version="$(
      dpkg-query -W -f='${Version}' "${package_name}" 2>/dev/null || true
    )"
    replacement_deb=""
    if [[ -n "${installed_version}" ]]; then
      replacement_deb="$(
        find_verified_bundled_replacement \
          "${package_name}" "${installed_version}" "${install_specs[@]}" || true
      )"
    fi
    if [[ -z "${replacement_deb}" && -n "${installed_version}" ]]; then
      replacement_deb="$(find_verified_runner_retirement \
        "${package_name}" "${installed_version}" "${install_specs[@]}" || true)"
    fi
    if [[ -z "${replacement_deb}" && -n "${installed_version}" ]]; then
      replacement_deb="$(find_verified_firmware_retirement \
        "${package_name}" "${installed_version}" "${install_specs[@]}" || true)"
    fi
    if [[ -n "${replacement_deb}" ]]; then
      verified_replacements+=(
        "${package_name}=${installed_version} -> ${replacement_deb}"
      )
      continue
    fi
    cat "${simulation_log}" >&2
    echo "Refusing to install because APT would remove ${package} without a bundled package that Provides its exact installed version and explicitly Replaces and Conflicts with it." >&2
    return 1
  done

  if [[ "${#verified_replacements[@]}" -gt 0 ]]; then
    log "Verified platform package replacements:"
    printf '  %s\n' "${verified_replacements[@]}"
  fi
}

# When retired neat-libcamera recovery packages are still installed, extend
# the transaction with explicit name=version specs derived from the installed
# palette's version: the pinned palette cannot be removed, and explicit
# versions select the SiMa repository packages even where a shared name has a
# foreign candidate. Emits nothing on boards without retired packages,
# keeping their transaction unchanged.
collect_board_heal_specs() {
  local target palette_version
  local -a retired_installed=()

  for target in neat-libcamera neat-libcamera-dev neat-libcamera-tools; do
    if deb_package_is_installed "${target}"; then
      retired_installed+=("${target}")
    fi
  done
  if [[ "${#retired_installed[@]}" -eq 0 ]]; then
    return 0
  fi

  palette_version=""
  if deb_package_is_installed simaai-palette-modalix; then
    palette_version="$(
      dpkg-query -W -f='${Version}' simaai-palette-modalix 2>/dev/null || true
    )"
  fi
  if [[ -z "${palette_version}" ]]; then
    echo "Retired neat-libcamera packages are installed, but the simaai-palette-modalix version is unavailable; continuing without platform pins." >&2
    return 0
  fi

  printf 'simaai-palette-modalix=%s\n' "${palette_version}"
  for target in "${retired_installed[@]}"; do
    printf '%s=%s\n' "${target#neat-}" "${palette_version}"
  done
}

log_green() {
  printf '%s[install_neat_framework] %s%s\n' "${GREEN}" "$*" "${RESET}"
}

print_green_banner() {
  printf '\n%s=============================================================%s\n' "${GREEN}" "${RESET}"
  printf '%s PYNEAT VIRTUAL ENV INSTALLED AT:%s\n' "${GREEN}" "${RESET}"
  printf '%s %s%s\n' "${GREEN}" "$1" "${RESET}"
  printf '%s ACTIVATE WITH:%s\n' "${GREEN}" "${RESET}"
  printf '%s source %s%s\n' "${GREEN}" "$2" "${RESET}"
  printf '%s=============================================================%s\n\n' "${GREEN}" "${RESET}"
}

resolve_venv_dir() {
  if [[ -n "${PYNEAT_VENV_DIR:-}" ]]; then
    printf '%s\n' "${PYNEAT_VENV_DIR}"
    return 0
  fi

  if [[ -d /media/nvme && -w /media/nvme ]]; then
    printf '%s\n' "/media/nvme/pyneat"
    return 0
  fi

  printf '%s\n' "$HOME/pyneat"
}

ensure_home_pyneat_symlink() {
  local venv_dir="$1"
  local home_pyneat="$HOME/pyneat"

  # Only NVMe-backed installs need a stable home path. Devices that install
  # directly into $HOME/pyneat should not create or rewrite any symlink.
  if [[ "${venv_dir}" != "/media/nvme/pyneat" ]]; then
    return 0
  fi

  # Preserve an existing correct symlink. If $HOME/pyneat is still a real
  # directory from an older install layout, replace it so activation can always
  # use ~/pyneat/bin/activate on NVMe-backed devices.
  if [[ -L "${home_pyneat}" ]]; then
    local target
    target="$(readlink "${home_pyneat}")"
    if [[ "${target}" == "${venv_dir}" ]]; then
      return 0
    fi
    rm -f "${home_pyneat}"
  elif [[ -d "${home_pyneat}" ]]; then
    rm -rf "${home_pyneat}"
    log "Removed existing directory ${home_pyneat} before creating symlink"
  elif [[ -e "${home_pyneat}" ]]; then
    echo "Cannot create ${home_pyneat} symlink; path already exists and is not a directory or symlink." >&2
    exit 1
  fi

  # Keep activation instructions consistent across devices:
  #   source ~/pyneat/bin/activate
  ln -sfn "${venv_dir}" "${home_pyneat}"
  log "Created symlink ${home_pyneat} -> ${venv_dir}"
}

activation_path_for_display() {
  local venv_dir="$1"

  if [[ "${venv_dir}" == "/media/nvme/pyneat" ]]; then
    printf '%s\n' '$HOME/pyneat/bin/activate'
    return 0
  fi

  printf '%s\n' "${venv_dir}/bin/activate"
}

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return $?
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "This installation modifies system packages and requires root or sudo." >&2
    exit 1
  fi

  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return $?
  fi

  local pw="${SUDO_PASSWORD}"
  if [[ -z "${pw}" ]]; then
    pw="${DEFAULT_SUDO_PASSWORD}"
  fi

  if printf '%s\n' "${pw}" | sudo -S -v >/dev/null 2>&1; then
    printf '%s\n' "${pw}" | sudo -S "$@"
    return $?
  fi

  if [[ -t 0 ]]; then
    read -r -s -p "Enter sudo password: " pw
    echo
    if [[ -z "${pw}" ]]; then
      echo "sudo password is required." >&2
      exit 1
    fi
    printf '%s\n' "${pw}" | sudo -S -v >/dev/null
    printf '%s\n' "${pw}" | sudo -S "$@"
    return $?
  fi

  echo "Unable to authenticate sudo. Set SUDO_PASSWORD or DEVKIT_PASSWORD." >&2
  exit 1
}

require_sentinel_installer() {
  if [[ ! -f "${SENTINEL_INSTALLER_PATH}" ]]; then
    echo "Missing ${SENTINEL_INSTALLER_PATH}; it ships with this installer and installs the Sentinel telemetry daemon." >&2
    return 1
  fi
}

# Model::benchmark() reads board telemetry from the Sentinel daemon but runs
# without it, so a failed Sentinel install only warns unless
# NEAT_INSTALLER_SENTINEL_REQUIRED=ON. Run the script through bash because
# downloads may drop its executable bit. Sentinel's installer reads the sudo
# password from SUDO_PASSWORD, which may hold the DEVKIT_PASSWORD value.
install_board_sentinel() {
  if require_sentinel_installer &&
    SUDO_PASSWORD="${SUDO_PASSWORD}" bash "${SENTINEL_INSTALLER_PATH}"; then
    return 0
  fi

  if [[ "${NEAT_INSTALLER_SENTINEL_REQUIRED}" == "ON" ]]; then
    echo "Sentinel installation failed and NEAT_INSTALLER_SENTINEL_REQUIRED=ON." >&2
    exit 1
  fi
  log "WARNING: Sentinel is not installed. Model::benchmark() reports power and board telemetry as unavailable until you run 'sima-cli neat install sentinel'."
}

read_metadata_field() {
  local metadata_file="$1"
  local requested_key="$2"
  awk -v requested_key="${requested_key}" '
    {
      separator = index($0, "=")
      if (separator == 0) {
        next
      }
      key = substr($0, 1, separator - 1)
      value = substr($0, separator + 1)
      sub(/^[[:space:]]+/, "", key)
      sub(/[[:space:]]+$/, "", key)
      if (key != requested_key) {
        next
      }
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if (value ~ /^".*"$/) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
      exit
    }
  ' "${metadata_file}" 2>/dev/null || true
}

run_scp() {
  if command -v scp >/dev/null 2>&1; then
    scp "$@"
    return 0
  fi
  if command -v sima-scp >/dev/null 2>&1; then
    sima-scp "$@"
    return 0
  fi
  echo "Neither scp nor sima-scp is available for DevKit sync." >&2
  return 1
}

run_ssh() {
  if command -v ssh >/dev/null 2>&1; then
    ssh "$@"
    return 0
  fi
  if command -v sima-ssh >/dev/null 2>&1; then
    sima-ssh "$@"
    return 0
  fi
  echo "Neither ssh nor sima-ssh is available for DevKit sync." >&2
  return 1
}

detect_env_mode() {
  local sdk_release_unrecognized=0
  if [[ -f "${ELXR_SDK_RELEASE_FILE}" ]]; then
    local sdk_type product_name
    sdk_type="$(read_metadata_field "${ELXR_SDK_RELEASE_FILE}" "SDK Type")"
    product_name="$(read_metadata_field "${ELXR_SDK_RELEASE_FILE}" "Product Name")"
    if [[ "${sdk_type}" == "ros2-sdk" ||
          "${product_name}" == "SiMa.ai ROS2 SDK" ]]; then
      echo "ros2-sdk"
      return 0
    fi
    if [[ -z "${sdk_type}" &&
          -n "$(read_metadata_field "${ELXR_SDK_RELEASE_FILE}" "SDK Version")" ]]; then
      echo "elxr-sdk"
      return 0
    fi
    sdk_release_unrecognized=1
  fi
  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ "${sdk_release_unrecognized}" -eq 1 ]]; then
    echo "unsupported"
    return 0
  fi
  if [[ -f "${NEAT_BUILDINFO_FILE}" ]] && grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' "${NEAT_BUILDINFO_FILE}"; then
    echo "modalix-board"
    return 0
  fi
  # Preserve the legacy fallback for DevKit images that do not provide
  # /etc/buildinfo. An unrecognized SDK release file is rejected above.
  echo "modalix-board"
}

resolve_package_manifest_path() {
  if [[ "${NEAT_PACKAGE_MANIFEST}" == /* ]]; then
    printf '%s\n' "${NEAT_PACKAGE_MANIFEST}"
    return 0
  fi
  if [[ -f "./${NEAT_PACKAGE_MANIFEST}" ]]; then
    printf '%s\n' "./${NEAT_PACKAGE_MANIFEST}"
    return 0
  fi
  if [[ -f "./deps/${NEAT_PACKAGE_MANIFEST}" ]]; then
    printf '%s\n' "./deps/${NEAT_PACKAGE_MANIFEST}"
    return 0
  fi
  if [[ -f "$(dirname "${INSTALLER_SCRIPT_PATH}")/../deps/${NEAT_PACKAGE_MANIFEST}" ]]; then
    printf '%s\n' "$(dirname "${INSTALLER_SCRIPT_PATH}")/../deps/${NEAT_PACKAGE_MANIFEST}"
    return 0
  fi
  printf '%s\n' "./${NEAT_PACKAGE_MANIFEST}"
}

read_manifest_platform_version() {
  local manifest_path="$1"
  python3 - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(f"missing package manifest: {manifest_path}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid package manifest JSON: {manifest_path}: {exc}")

version = str(data.get("platform-version", "")).strip()
if not version:
    raise SystemExit(f"missing or empty platform-version in package manifest: {manifest_path}")
print(version.split("+", 1)[0])
PY
}

read_sdk_platform_version() {
  local release_file="$1"
  awk -F'=' '
    $1 ~ /^[[:space:]]*Platform Base[[:space:]]*$/ {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if (value != "") {
        print value
        found=1
        exit
      }
    }
    $1 ~ /^[[:space:]]*SDK Version[[:space:]]*$/ {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      sub(/_.*/, "", value)
      sdk_version=value
    }
    END {
      if (!found && sdk_version != "") {
        print sdk_version
      }
    }
  ' "${release_file}" 2>/dev/null || true
}

sdk_platform_version_label() {
  local release_file="$1"
  if grep -qE '^[[:space:]]*Platform Base[[:space:]]*=[[:space:]]*[^[:space:]]' "${release_file}" 2>/dev/null; then
    printf '%s\n' "Platform Base"
  else
    printf '%s\n' "SDK Version"
  fi
}

read_ros2_sdk_platform_version() {
  local release_file="$1"
  local value
  value="$(read_metadata_field "${release_file}" "Platform Base")"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  read_metadata_field "${release_file}" "Platform Version"
}

ros2_sdk_platform_version_label() {
  local release_file="$1"
  if [[ -n "$(read_metadata_field "${release_file}" "Platform Base")" ]]; then
    printf '%s\n' "Platform Base"
  else
    printf '%s\n' "Platform Version"
  fi
}

read_devkit_platform_version() {
  local buildinfo_file="$1"
  awk -F'=' '
    $1 ~ /^[[:space:]]*DISTRO_VERSION[[:space:]]*$/ {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' "${buildinfo_file}" 2>/dev/null || true
}

ensure_platform_compatible() {
  if [[ "${FORCE_INSTALL:-}" == "1" ]]; then
    log "FORCE_INSTALL=1; skipping platform compatibility check. Package may not work on this platform."
    return 0
  fi
  if [[ "${NEAT_INSTALLER_SKIP_PLATFORM_CHECK}" == "ON" ]]; then
    log "NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON; skipping platform compatibility check."
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to read the Neat package manifest before install." >&2
    exit 1
  fi

  local manifest_path expected actual source_label source_file
  manifest_path="$(resolve_package_manifest_path)"
  if ! expected="$(read_manifest_platform_version "${manifest_path}")"; then
    echo "Unable to verify Neat package platform compatibility." >&2
    exit 1
  fi

  case "${ENV_MODE}" in
    elxr-sdk)
      source_file="${ELXR_SDK_RELEASE_FILE}"
      if [[ ! -f "${source_file}" ]]; then
        echo "Cannot verify eLxr SDK compatibility: missing ${source_file}." >&2
        echo "Set ELXR_SDK_RELEASE_FILE or NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON for an explicit development override." >&2
        exit 1
      fi
      source_label="$(sdk_platform_version_label "${source_file}")"
      actual="$(read_sdk_platform_version "${source_file}")"
      ;;
    ros2-sdk)
      source_file="${ELXR_SDK_RELEASE_FILE}"
      if [[ ! -f "${source_file}" ]]; then
        echo "Cannot verify ROS2 SDK compatibility: missing ${source_file}." >&2
        exit 1
      fi
      source_label="$(ros2_sdk_platform_version_label "${source_file}")"
      actual="$(read_ros2_sdk_platform_version "${source_file}")"
      ;;
    modalix-board)
      source_label="DISTRO_VERSION"
      source_file="${NEAT_BUILDINFO_FILE}"
      if [[ ! -f "${source_file}" ]]; then
        echo "Cannot verify Modalix DevKit compatibility: missing ${source_file}." >&2
        echo "This installer only supports Modalix DevKit targets or eLxr SDK environments." >&2
        exit 1
      fi
      if ! grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' "${source_file}"; then
        echo "Cannot verify Modalix DevKit compatibility: ${source_file} does not report MACHINE=modalix." >&2
        exit 1
      fi
      actual="$(read_devkit_platform_version "${source_file}")"
      ;;
    *)
      echo "Unsupported installation environment." >&2
      echo "Expected SDK Type=ros2-sdk, an eLxr SDK release file, or MACHINE=modalix build metadata." >&2
      exit 1
      ;;
  esac

  if [[ -z "${actual}" ]]; then
    echo "Cannot verify platform compatibility: ${source_label} is missing in ${source_file}." >&2
    exit 1
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Incompatible platform version for this Neat package." >&2
    echo "  Package platform-version: ${expected} (${manifest_path})" >&2
    echo "  Detected ${source_label}: ${actual} (${source_file})" >&2
    echo "Refusing to install before modifying Python, apt, or sysroot packages." >&2
    exit 1
  fi

  log "Platform compatibility verified: ${actual}"
}

install_skill_for_agent() {
  local source_dir="$1"
  local agent_name="$2"
  local agent_home="$3"

  local target_dir="${agent_home}/skills/sima-neat"
  mkdir -p "$(dirname "${target_dir}")"
  rm -rf "${target_dir}"
  cp -a "${source_dir}" "${target_dir}"
  log "Installed ${agent_name} skill to: ${target_dir}"
}

install_agent_skills_for_current_user() {
  local source_dir="$1"

  if [[ ! -d "${source_dir}" ]]; then
    log "Agent skill source not found; skipping skill install: ${source_dir}"
    return 0
  fi

  if [[ "${NEAT_INSTALLER_INSTALL_CODEX_SKILL}" == "ON" ]]; then
    install_skill_for_agent "${source_dir}" "Codex" "${CODEX_HOME:-$HOME/.codex}"
  else
    log "NEAT_INSTALLER_INSTALL_CODEX_SKILL=${NEAT_INSTALLER_INSTALL_CODEX_SKILL}; skipping Codex skill install."
  fi

  if [[ "${NEAT_INSTALLER_INSTALL_CLAUDE_SKILL}" == "ON" ]]; then
    install_skill_for_agent "${source_dir}" "Claude" "${CLAUDE_HOME:-$HOME/.claude}"
  else
    log "NEAT_INSTALLER_INSTALL_CLAUDE_SKILL=${NEAT_INSTALLER_INSTALL_CLAUDE_SKILL}; skipping Claude skill install."
  fi
}

append_matching_files() {
  local out_array_name="$1"
  local search_dir="$2"
  local pattern="$3"
  local -n out_array="${out_array_name}"
  local -a matches=()
  local manifest_path="${search_dir}/${NEAT_INSTALL_MANIFEST}"

  if [[ -f "${manifest_path}" ]]; then
    local line basename file
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line%$'\r'}"
      [[ -n "${line}" ]] || continue
      basename="$(basename "${line}")"
      [[ "${basename}" == ${pattern} ]] || continue
      file="${search_dir}/${basename}"
      if [[ ! -f "${file}" ]]; then
        echo "Install manifest references missing file: ${basename}" >&2
        exit 1
      fi
      matches+=("${file}")
    done < "${manifest_path}"
  else
    mapfile -t matches < <(find "${search_dir}" -maxdepth 1 -type f -name "${pattern}" | sort)
  fi

  out_array+=("${matches[@]}")
}

collect_wheel_files() {
  local search_dir="$1"
  local out_array_name="$2"
  local -n out_array="${out_array_name}"
  local manifest_path="${search_dir}/${NEAT_INSTALL_MANIFEST}"
  out_array=()

  if [[ -f "${manifest_path}" ]]; then
    local line basename file
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line%$'\r'}"
      [[ -n "${line}" ]] || continue
      basename="$(basename "${line}")"
      [[ "${basename}" == *.whl ]] || continue
      file="${search_dir}/${basename}"
      if [[ ! -f "${file}" ]]; then
        echo "Install manifest references missing file: ${basename}" >&2
        exit 1
      fi
      out_array+=("${file}")
    done < "${manifest_path}"
  else
    mapfile -t out_array < <(find "${search_dir}" -maxdepth 1 -type f -name '*.whl' | sort)
  fi
}

collect_debs_in_install_order() {
  local search_dir="$1"
  local out_array_name="$2"
  local -n out_array="${out_array_name}"
  out_array=()
  append_matching_files "${out_array_name}" "${search_dir}" '*.deb'
}

# Disk-backed locations come before /tmp, which is RAM-backed on Modalix boards.
bundle_stage_parent_candidates() {
  printf '%s\n' /media/nvme "${TMPDIR:-}" /var/tmp /tmp "${INSTALLER_START_DIR}"
}

create_bundle_stage_dir() {
  local required_kb="$1"
  local parent available_kb stage_dir
  while IFS= read -r parent; do
    [[ -n "${parent}" && -d "${parent}" && -w "${parent}" ]] || continue
    available_kb="$(df -Pk "${parent}" 2>/dev/null | awk 'NR == 2 {print $4}')"
    [[ "${available_kb}" =~ ^[0-9]+$ && "${available_kb}" -ge "${required_kb}" ]] || continue
    stage_dir="$(mktemp -d "${parent}/sima-neat-install.XXXXXX" 2>/dev/null)" || continue
    printf '%s\n' "${stage_dir}"
    return 0
  done < <(bundle_stage_parent_candidates)
  return 1
}

# sima-cli downloads every Core bundle into the same directory, so files from
# earlier installs can sit next to this one. Scheduler admission rejects any
# .deb the install manifest does not list, so install from a fresh directory
# that holds only this bundle.
stage_bundle_in_fresh_dir() {
  [[ -f "./${NEAT_INSTALL_MANIFEST}" ]] || return 0
  local -a bundle_files=()
  append_matching_files bundle_files "." '*'
  bundle_files+=("./${NEAT_INSTALL_MANIFEST}")
  if [[ -f ./resolved-deps-manifest.json ]]; then
    bundle_files+=(./resolved-deps-manifest.json)
  fi
  # An absolute package manifest path still resolves after the directory change.
  local package_manifest=""
  if [[ "${NEAT_PACKAGE_MANIFEST}" != /* ]]; then
    package_manifest="$(resolve_package_manifest_path)"
    [[ -f "${package_manifest}" ]] || package_manifest=""
  fi

  local required_kb stage_dir
  required_kb="$(du -ckL -- "${bundle_files[@]}" ${package_manifest:+"${package_manifest}"} |
    awk 'END {print $1}')"
  if ! stage_dir="$(create_bundle_stage_dir "${required_kb}")"; then
    echo "No writable location has ${required_kb} KB free for a fresh copy of the install bundle." >&2
    echo "Checked: $(bundle_stage_parent_candidates | paste -sd ' ' -)" >&2
    exit 1
  fi
  INSTALLER_TMP_DIRS+=("${stage_dir}")
  # mktemp creates an owner-only directory; APT's sandbox user reads local .deb files.
  chmod 0755 "${stage_dir}"
  cp -p -- "${bundle_files[@]}" "${stage_dir}/"
  if [[ -n "${package_manifest}" ]]; then
    cp -p -- "${package_manifest}" "${stage_dir}/${NEAT_PACKAGE_MANIFEST}"
  fi
  log "Installing from a fresh copy of the bundle: ${stage_dir}"
  cd -- "${stage_dir}"
}

sysroot_path() {
  printf '%s\n' "${SYSROOT:-/opt/toolchain/aarch64/modalix}"
}

sysroot_neat_install_packages_dir() {
  printf '%s\n' "$(sysroot_path)/neat-install-packages"
}

ensure_sima_lmm_sysroot_deps() {
  local sysroot="$1"
  # libsima_neat.so links cpp-httplib, which links brotli.
  local -a dependencies=(
    "usr/include/nlohmann/json.hpp:nlohmann-json3-dev"
    "usr/lib/aarch64-linux-gnu/libbrotlicommon.so.1:libbrotli1:arm64"
    "usr/lib/aarch64-linux-gnu/libcpp-httplib.so.0.18:libcpp-httplib0.18:arm64"
  )
  if compgen -G './sima-lmm-*.deb' >/dev/null 2>&1; then
    dependencies+=(
      "usr/lib/aarch64-linux-gnu/libfmt.so.10:libfmt10:arm64"
      "usr/lib/aarch64-linux-gnu/libspdlog.so.1.15:libspdlog1.15:arm64"
      "usr/lib/aarch64-linux-gnu/libfftw3.so.3:libfftw3-double3:arm64"
      "usr/lib/aarch64-linux-gnu/libavcodec.so.61:libavcodec61:arm64"
      "usr/lib/aarch64-linux-gnu/libavformat.so.61:libavformat61:arm64"
      "usr/lib/aarch64-linux-gnu/libavutil.so.59:libavutil59:arm64"
      "usr/lib/aarch64-linux-gnu/libswresample.so.5:libswresample5:arm64"
    )
  fi
  local dependency
  local -a missing_packages=()
  for dependency in "${dependencies[@]}"; do
    [[ -e "${sysroot}/${dependency%%:*}" ]] || missing_packages+=("${dependency#*:}")
  done
  [[ "${#missing_packages[@]}" -gt 0 ]] || return 0

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/neat-sysroot-deps-XXXXXX)"
  log "Installing customer SDK dependencies: ${missing_packages[*]}"
  if ! (cd "${tmp_dir}" && apt-get download "${missing_packages[@]}"); then
    rm -rf "${tmp_dir}"
    echo "Failed to download customer SDK dependencies." >&2
    exit 1
  fi
  local dep_deb
  for dep_deb in "${tmp_dir}"/*.deb; do
    if ! dpkg-deb -x "${dep_deb}" "${sysroot}" 2>/dev/null; then
      run_sudo dpkg-deb -x "${dep_deb}" "${sysroot}"
    fi
  done
  rm -rf "${tmp_dir}"
  for dependency in "${dependencies[@]}"; do
    if [[ ! -e "${sysroot}/${dependency%%:*}" ]]; then
      echo "Missing customer SDK dependency after install: ${dependency#*:}" >&2
      exit 1
    fi
  done
}

ensure_sdk_neat_cli_symlink() {
  local sysroot
  sysroot="$(sysroot_path)"
  local target="${sysroot}/usr/bin/neat"
  local link="/usr/local/bin/neat"

  if [[ ! -x "${target}" ]]; then
    return 0
  fi
  if [[ -d "${link}" && ! -L "${link}" ]]; then
    echo "Cannot create ${link} symlink; path already exists as a directory." >&2
    exit 1
  fi

  run_sudo mkdir -p "$(dirname "${link}")"
  run_sudo ln -sfn "${target}" "${link}"
  log "Created SDK neat command symlink ${link} -> ${target}"
}

cache_install_artifacts_in_sysroot() {
  local cache_dir
  cache_dir="$(sysroot_neat_install_packages_dir)"

  log "Caching SDK install artifacts in sysroot: ${cache_dir}"
  run_sudo mkdir -p "${cache_dir}"
  run_sudo rm -f \
    "${cache_dir}"/*.deb \
    "${cache_dir}"/*.whl \
    "${cache_dir}/${NEAT_INSTALL_MANIFEST}" \
    "${cache_dir}/${NEAT_PACKAGE_MANIFEST}" \
    "${cache_dir}"/install_neat_framework.sh \
    "${cache_dir}"/install_sentinel.sh \
    "${cache_dir}"/scheduler_platform_admission.py \
    "${cache_dir}"/resolved-deps-manifest.json

  local file
  for file in "${DEBS[@]}"; do
    run_sudo cp -f "${file}" "${cache_dir}/"
  done

  local -a wheel_files=()
  collect_wheel_files "." wheel_files
  for file in "${wheel_files[@]}"; do
    run_sudo cp -f "${file}" "${cache_dir}/"
  done

  if [[ -f "./${NEAT_INSTALL_MANIFEST}" ]]; then
    run_sudo cp -f "./${NEAT_INSTALL_MANIFEST}" "${cache_dir}/"
  fi
  local package_manifest
  package_manifest="$(resolve_package_manifest_path)"
  if [[ -f "${package_manifest}" ]]; then
    run_sudo cp -f "${package_manifest}" "${cache_dir}/${NEAT_PACKAGE_MANIFEST}"
  fi
  run_sudo cp -f "${INSTALLER_SCRIPT_PATH}" "${cache_dir}/install_neat_framework.sh"
  run_sudo chmod 0755 "${cache_dir}/install_neat_framework.sh"
  require_sentinel_installer || exit 1
  run_sudo cp -f "${SENTINEL_INSTALLER_PATH}" "${cache_dir}/install_sentinel.sh"
  run_sudo chmod 0755 "${cache_dir}/install_sentinel.sh"
  run_sudo cp -f "${SCHEDULER_PLATFORM_ADMISSION_PATH}" "${cache_dir}/"
  if [[ -f ./resolved-deps-manifest.json ]]; then
    run_sudo cp -f ./resolved-deps-manifest.json "${cache_dir}/"
  fi
}

collect_cached_devkit_deploy_files() {
  local cache_dir
  cache_dir="$(sysroot_neat_install_packages_dir)"

  if [[ "${ENV_MODE:-}" != "elxr-sdk" ]]; then
    echo "Paired DevKit sync from sysroot cache is only supported in eLxr SDK mode." >&2
    exit 1
  fi
  if [[ ! -d "${cache_dir}" ]]; then
    echo "Missing SDK install artifact cache: ${cache_dir}" >&2
    exit 1
  fi

  local -a cached_core_debs=()
  local -a cached_dev_debs=()
  mapfile -t cached_core_debs < <(find "${cache_dir}" -maxdepth 1 -type f -name 'sima-neat-*-Linux-core.deb' | sort)
  mapfile -t cached_dev_debs < <(find "${cache_dir}" -maxdepth 1 -type f -name 'sima-neat-*-Linux-dev.deb' | sort)
  collect_debs_in_install_order "${cache_dir}" CACHED_DEBS
  collect_wheel_files "${cache_dir}" CACHED_WHEELS
  local cached_installer="${cache_dir}/install_neat_framework.sh"
  local cached_sentinel_installer="${cache_dir}/install_sentinel.sh"
  local cached_manifest="${cache_dir}/${NEAT_INSTALL_MANIFEST}"
  local cached_package_manifest="${cache_dir}/${NEAT_PACKAGE_MANIFEST}"

  if [[ "${#cached_core_debs[@]}" -lt 1 ]]; then
    echo "No cached sima-neat core DEB found for paired DevKit sync in: ${cache_dir}" >&2
    exit 1
  fi
  if [[ "${#cached_dev_debs[@]}" -lt 1 ]]; then
    echo "No cached sima-neat dev DEB found for paired DevKit sync in: ${cache_dir}" >&2
    exit 1
  fi
  if [[ "${#CACHED_DEBS[@]}" -lt 1 ]]; then
    echo "No cached DEB files found for paired DevKit sync in: ${cache_dir}" >&2
    exit 1
  fi
  if [[ "${#CACHED_WHEELS[@]}" -lt 1 ]]; then
    echo "No cached PyNeat wheel found for paired DevKit sync in: ${cache_dir}" >&2
    exit 1
  fi
  local script
  for script in "${cached_installer}" "${cached_sentinel_installer}"; do
    if [[ ! -f "${script}" ]]; then
      echo "Cached installer script not found for paired DevKit sync: ${script}" >&2
      exit 1
    fi
  done

  CACHED_DEPLOY_FILES=("${CACHED_DEBS[@]}" "${CACHED_WHEELS[@]}")
  local admission_resource
  for admission_resource in scheduler_platform_admission.py resolved-deps-manifest.json; do
    if [[ -f "${cache_dir}/${admission_resource}" ]]; then
      CACHED_DEPLOY_FILES+=("${cache_dir}/${admission_resource}")
    elif [[ "${admission_resource}" == scheduler_platform_admission.py ]]; then
      echo "Cached Platform admission delegate is missing." >&2
      exit 1
    fi
  done
  if [[ -f "${cached_manifest}" ]]; then
    CACHED_DEPLOY_FILES+=("${cached_manifest}")
  fi
  if [[ -f "${cached_package_manifest}" ]]; then
    CACHED_DEPLOY_FILES+=("${cached_package_manifest}")
  fi
  CACHED_DEPLOY_FILES+=("${cached_installer}" "${cached_sentinel_installer}")
}

apt_package_database_is_healthy() {
  local apt_check_log
  apt_check_log="$(mktemp /tmp/sima-neat-apt-check-XXXXXX)"

  if run_sudo apt-get check >"${apt_check_log}" 2>&1; then
    rm -f "${apt_check_log}"
    return 0
  fi

  # Keep the authoritative diagnostic in the install/CI log, not just a generic
  # unhealthy-state message. The temporary capture itself is still removed.
  cat "${apt_check_log}" >&2
  rm -f "${apt_check_log}"

  # Observation only: admission still decides whether recovery is allowed.
  # Preserve full pending/abnormal states and trigger fields without dumping
  # the healthy package inventory or masking the original APT failure.
  local dpkg_states
  if dpkg_states="$(dpkg-query -W -f='${binary:Package}\t${Status}\t${db:Status-Abbrev}\t${Version}\t${Architecture}\t${Triggers-Pending}\t${Triggers-Awaited}\n')"; then
    if ! printf '%s\n' "${dpkg_states}" | awk -F '\t' '
      BEGIN { print "Package\tStatus\tStatus-Abbrev\tVersion\tArchitecture\tTriggers-Pending\tTriggers-Awaited" }
      $2 != "" && $2 != "install ok installed" && $2 != "hold ok installed" &&
          $2 !~ / (not-installed|config-files)$/ {
        count++
        if (count <= 100) print $0
      }
      END { printf "Incomplete dpkg records: %d; omitted: %d\n", count, (count > 100 ? count - 100 : 0) }
    ' >&2; then
      echo "Unable to format incomplete dpkg state; retaining the original APT failure." >&2
    fi
  else
    echo "Unable to read incomplete dpkg state; retaining the original APT failure." >&2
  fi
  return 1
}

# apt-get check validates dependencies, not completion of unpack/configure.
# Scheduler owns pending-cohort eligibility. Only the intermediate recovery
# check permits normal queued triggers; final installation must drain them.
dpkg_package_database_is_configured() {
  local states state
  states="$(dpkg-query -W -f='${Status}\n')" || return 2
  while IFS= read -r state; do
    case "${state}" in
      "install ok installed"|"hold ok installed"|*" not-installed"|*" config-files"|"") ;;
      "install ok triggers-pending"|"install ok triggers-awaited")
        [[ "${1:-}" == before-reinstall ]] || return 1
        ;;
      *) return 1 ;;
    esac
  done <<< "${states}"
}

apt_package_lists_are_populated() {
  [[ -d /var/lib/apt/lists ]] || return 1
  find /var/lib/apt/lists -maxdepth 1 -type f -name '*_Packages' -print -quit 2>/dev/null | grep -q .
}

refresh_apt_metadata_for_board_install() {
  case "${NEAT_INSTALLER_APT_UPDATE}" in
    OFF|off|0|false|FALSE)
      log "NEAT_INSTALLER_APT_UPDATE=${NEAT_INSTALLER_APT_UPDATE}; skipping apt metadata refresh."
      return 0
      ;;
    ON|on|1|true|TRUE|AUTO|auto|"") ;;
    *)
      echo "Invalid NEAT_INSTALLER_APT_UPDATE=${NEAT_INSTALLER_APT_UPDATE}; expected AUTO, ON, or OFF." >&2
      exit 1
      ;;
  esac

  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  local should_update=0
  if [[ "${NEAT_INSTALLER_APT_UPDATE}" =~ ^(ON|on|1|true|TRUE)$ ]]; then
    should_update=1
  elif ! apt_package_lists_are_populated; then
    log "APT package lists have no Packages indexes; refreshing metadata before local DEB install."
    should_update=1
  else
    log "APT package lists already contain Packages indexes; skipping apt-get update."
  fi

  [[ "${should_update}" -eq 1 ]] || return 0

  if run_sudo apt-get update; then
    return 0
  fi

  log "apt-get update failed; continuing so apt-get install reports the authoritative dependency error."
  return 0
}

deb_package_is_installed() {
  dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null | grep -q '^ii '
}

verify_canonical_palette_installation() {
  if ! deb_package_is_installed simaai-palette-modalix; then
    echo "simaai-palette-modalix is not installed after the native Modalix transaction." >&2
    return 1
  fi
}

# Preserve the native platform package identity across direct-runtime installs.
# Palette 3 no longer promises an OTA executable; do not discover or invoke one.
capture_direct_platform_identity() {
  dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\t${Status}\n' \
    simaai-palette-modalix
}

remove_installed_local_deb_packages() {
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    return 0
  fi

  local -a packages=()
  local -A seen=()
  local deb package idx

  # Remove in reverse install order so packages that depend on lower-level
  # runtime packages are removed before their dependencies.
  for ((idx = ${#DEBS[@]} - 1; idx >= 0; idx--)); do
    deb="${DEBS[$idx]}"
    package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    [[ -n "${package}" ]] || continue
    [[ -z "${seen[${package}]:-}" ]] || continue
    seen["${package}"]=1
    if dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^i'; then
      packages+=("${package}")
    fi
  done

  if [[ "${#packages[@]}" -eq 0 ]]; then
    return 0
  fi

  if board_platform_admission_is_active || board_has_scheduler_cohort; then
    local removal_simulation
    removal_simulation="$(mktemp /tmp/sima-neat-removal-simulate-XXXXXX)"
    INSTALLER_TMP_DIRS+=("${removal_simulation}")
    run_sudo apt-get remove -y --simulate "${packages[@]}" >"${removal_simulation}" 2>&1 || return 1
    verify_board_platform_plan "${removal_simulation}" || return 1
    # Maintenance may remove the selected products, never unrelated dependents.
    local removed selected
    while read -r removed; do
      [[ -n "${removed}" ]] || continue
      removed="${removed%%:*}"
      selected=0
      for package in "${packages[@]}"; do
        [[ "${package}" != "${removed}" ]] || selected=1
      done
      if [[ "${selected}" != 1 ]]; then
        echo "Maintenance removal would remove unrelated package: ${removed}" >&2
        return 1
      fi
    done < <(awk '$1 == "Remv" {print $2}' "${removal_simulation}")
  fi

  log "Removing installed NEAT packages before retrying apt downgrade/repair:"
  printf '  %s\n' "${packages[@]}"
  if run_sudo apt-get remove -y "${packages[@]}"; then
    return 0
  fi

  log "apt-get remove failed; falling back to forced dpkg removal before apt repair."
  run_sudo dpkg --remove --force-depends "${packages[@]}"
}

board_runtime_is_legacy() {
  python3 - "$(resolve_package_manifest_path)" "${NEAT_BUILDINFO_FILE}" <<'PYPROFILE'
import json
import re
import sys
from pathlib import Path

try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not re.fullmatch(r"2[.]1[.][0-9]+", str(manifest.get("platform-version", ""))):
        raise SystemExit(1)
    fields = {}
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
        if len(line) > 4096:
            raise SystemExit(1)
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key in ("MACHINE", "DISTRO_VERSION"):
            if key in fields:
                raise SystemExit(1)
            fields[key] = value.strip()
    raise SystemExit(0 if fields.get("MACHINE") == "modalix" and re.fullmatch(
        r"2[.]1[.][0-9]+([.~+_-][A-Za-z0-9_.+~-]+)?", fields.get("DISTRO_VERSION", "")
    ) else 1)
except (OSError, ValueError, AttributeError, TypeError):
    raise SystemExit(1)
PYPROFILE
}

stop_board_runtime_before_install() {
  if ! board_runtime_is_legacy; then
    log "Direct or unidentified profile: skipping legacy runtime lifecycle action."
    return 0
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  log "Stopping NEAT runtime services before package replacement."
  local svc
  for svc in \
      simaai-pipeline-manager.service \
      rctd.service \
      encoder.service \
      decoder.service \
      simaai-log.service; do
    if systemctl cat "${svc}" >/dev/null 2>&1; then
      run_sudo systemctl stop "${svc}" >/dev/null 2>&1 || true
      run_sudo systemctl reset-failed "${svc}" >/dev/null 2>&1 || true
    fi
  done
}

activate_board_runtime_after_install() {
  if [[ "${NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD}" != ON ]]; then
    log "EV74 firmware activation skipped."
    return 0
  fi

  # Activate only a helper delivered by this transaction, never ambient firmware
  # left behind by a previously installed runtime or a different cohort.
  local helper=/usr/libexec/sima-neat-firmware/install.sh
  local deb listing selected="" package version architecture installed expected owner staged
  local required_control=""
  required_control="$(required_board_firmware_control)" || return 1
  if board_has_scheduler_cohort && [[ -z "${required_control}" ]]; then
    log "EV74 firmware activation skipped: not required by the selected Scheduler cohort."
    return 0
  fi
  local required_package="${required_control%%$'\t'*}"
  for deb in "${DEBS[@]}"; do
    if [[ -n "${required_control}" ]]; then
      package="$(dpkg-deb -f "${deb}" Package)" || return 1
      [[ "${package}" == "${required_package}" ]] || continue
    else
      if ! listing="$(dpkg-deb --fsys-tarfile "${deb}" | tar -tf -)"; then
        echo "Cannot inspect selected DEB firmware payload: ${deb}" >&2
        return 1
      fi
      grep -Fxq -e ".${helper}" -e "${helper#/}" \
        -e ".${helper}/" -e "${helper#/}/" <<< "${listing}" || continue
    fi
    if [[ -n "${selected}" ]]; then
      echo "Multiple selected DEBs own the EV74 activation helper." >&2
      return 1
    fi
    selected="${deb}"
  done
  if [[ -z "${selected}" ]]; then
    if [[ -n "${required_control}" ]]; then
      echo "Required native EV74 firmware package does not deliver the activation helper." >&2
      return 1
    fi
    log "EV74 firmware activation skipped: no selected package delivers the helper."
    return 0
  fi

  package="$(dpkg-deb -f "${selected}" Package)" || return 1
  version="$(dpkg-deb -f "${selected}" Version)" || return 1
  architecture="$(dpkg-deb -f "${selected}" Architecture)" || return 1
  [[ -n "${package}" && -n "${version}" && -n "${architecture}" ]] || return 1
  if [[ -n "${required_control}" ]]; then
    local selected_sha
    selected_sha="$(sha256sum "${selected}")" || return 1
    printf -v expected '%s\t%s\t%s\t%s' "${package}" "${version}" "${architecture}" "${selected_sha%% *}"
    if [[ "${required_control}" != "${expected}" ]]; then
      echo "Selected EV74 firmware differs from the authenticated Scheduler cohort." >&2
      return 1
    fi
  fi
  installed="$(dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\t${db:Status-Abbrev}' "${package}")" || return 1
  printf -v expected '%s\t%s\t%s\tii ' "${package}" "${version}" "${architecture}"
  owner="$(dpkg-query -S "${helper}")" || return 1
  if [[ "${installed}" != "${expected}" ||
        ( "${owner}" != "${package}: ${helper}" &&
          "${owner}" != "${package}:${architecture}: ${helper}" ) ]]; then
    echo "Installed EV74 activation helper is not owned by the exact configured selected package." >&2
    return 1
  fi

  staged="$(mktemp -d /tmp/sima-neat-firmware-admission-XXXXXX)" || return 1
  INSTALLER_TMP_DIRS+=("${staged}")
  dpkg-deb -x "${selected}" "${staged}" || return 1
  local packaged_helper="${staged}${helper}"
  if [[ ! -f "${packaged_helper}" || ! -x "${packaged_helper}" || -L "${packaged_helper}" ||
        ! -f "${helper}" || ! -x "${helper}" || -L "${helper}" ]] ||
     ! cmp -s "${packaged_helper}" "${helper}" ||
     [[ "$(stat -c '%a' "${packaged_helper}")" != "$(stat -c '%a' "${helper}")" ]]; then
    echo "Installed EV74 activation helper does not match the selected executable payload." >&2
    return 1
  fi

  local force_install=0
  if [[ "${FORCE_INSTALL:-}" == "1" ]]; then
    force_install=1
  fi
  log "Activating staged EV74 firmware from ${package} ${version}."
  # Scope the normalized override to activation, after sudo environment filtering.
  # Do not preserve unrelated caller variables or expand SKIP_PLATFORM_CHECK.
  run_sudo env "FORCE_INSTALL=${force_install}" "${helper}" --activate || return $?

  # --activate selects the native image only until reboot. The EV74 boots
  # modalix-cvu-fw, which the loader finds in /lib/firmware/updates before the
  # platform file; a dangling link falls back to the platform image.
  local boot_link=/lib/firmware/updates/modalix-cvu-fw
  run_sudo mkdir -p "${boot_link%/*}" || return $?
  run_sudo ln -sfn /lib/firmware/sima-neat-native-cvu-fw "${boot_link}"
}

restart_board_codec_services() {
  if ! board_runtime_is_legacy; then
    log "Direct or unidentified profile: skipping legacy runtime lifecycle action."
    return 0
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  local -a services=()
  local service
  for service in encoder.service decoder.service; do
    if systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
      services+=("${service}")
    fi
  done

  if [[ "${#services[@]}" -eq 0 ]]; then
    return 0
  fi

  log "Restarting codec services after package replacement."
  run_sudo systemctl daemon-reload || true
  run_sudo systemctl enable "${services[@]}" || true
  if ! run_sudo systemctl restart "${services[@]}"; then
    echo "Failed to restart codec services after NEAT package installation." >&2
    run_sudo systemctl --no-pager --full status "${services[@]}" >&2 || true
    run_sudo journalctl -u encoder.service -u decoder.service --no-pager -n 80 >&2 || true
    exit 1
  fi
}

verify_board_codec_services() {
  if ! board_runtime_is_legacy; then
    log "Direct or unidentified profile: skipping legacy runtime lifecycle action."
    return 0
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  local service
  for service in encoder.service decoder.service; do
    if ! systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
      continue
    fi

    if ! systemctl is-active --quiet "${service}"; then
      log "${service} is not active after package install; attempting to start it once."
      run_sudo systemctl start "${service}" || true
      sleep 1
    fi

    if ! systemctl is-active --quiet "${service}"; then
      echo "${service} is not active after NEAT package installation." >&2
      run_sudo systemctl --no-pager --full status "${service}" >&2 || true
      run_sudo journalctl -u "${service}" --no-pager -n 80 >&2 || true
      exit 1
    fi

    log "Verified ${service} is active."
  done
}

dispatcher_multiarch_triplet() {
  dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || printf '%s\n' 'aarch64-linux-gnu'
}

dispatcher_global_lib_dir() {
  printf '/usr/lib/%s\n' "$(dispatcher_multiarch_triplet)"
}

dispatcher_private_runtime_dir() {
  printf '%s/neat/runtime\n' "$(dispatcher_global_lib_dir)"
}

dispatcher_quarantine_root() {
  printf '%s\n' '/var/lib/sima-neat/quarantine/dispatcher'
}

collect_stale_global_dispatcher_paths() {
  local out_array_name="$1"
  local -n out_array="${out_array_name}"
  local lib_dir
  lib_dir="$(dispatcher_global_lib_dir)"
  out_array=()
  [[ -d "${lib_dir}" ]] || return 0
  mapfile -t out_array < <(
    find "${lib_dir}" -maxdepth 1 -mindepth 1 \
      -name 'libneatdispatchercore.so*' \
      -print | sort
  )
}

migrate_stale_global_dispatcher_libs() {
  local quarantine_dir path owner
  local -a stale_paths=()
  collect_stale_global_dispatcher_paths stale_paths
  [[ "${#stale_paths[@]}" -gt 0 ]] || return 0

  for path in "${stale_paths[@]}"; do
    owner="$(dpkg-query -S "${path}" 2>/dev/null || true)"
    if [[ -n "${owner}" ]]; then
      echo "Refusing to quarantine package-owned global dispatcher path ${path}: ${owner}" >&2
      echo "The dispatcher package must migrate that ownership explicitly." >&2
      return 1
    fi
  done

  quarantine_dir="$(dispatcher_quarantine_root)/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  run_sudo mkdir -p "${quarantine_dir}"
  for path in "${stale_paths[@]}"; do
    log "Quarantining stale global dispatcher path outside loader directories: ${path}"
    run_sudo mv -- "${path}" "${quarantine_dir}/$(basename "${path}")"
  done
  run_sudo ldconfig >/dev/null 2>&1 || true
}

dispatcher_path_is_owned_by_neat_runtime() {
  local path="$1" owner
  owner="$(dpkg-query -S "${path}" 2>/dev/null || true)"
  [[ "${owner}" =~ ^neat-runtime(:[^:[:space:]]+)?:[[:space:]] ]]
}

verify_private_dispatcher_runtime() {
  local runtime_dir runtime_file soname soname_path resolved
  local -a runtime_files=()
  local -a stale_paths=()
  runtime_dir="$(dispatcher_private_runtime_dir)"
  if [[ ! -d "${runtime_dir}" ]]; then
    echo "Packaged private dispatcher runtime directory is missing: ${runtime_dir}" >&2
    return 1
  fi
  mapfile -t runtime_files < <(
    find "${runtime_dir}" -maxdepth 1 -type f \
      -name 'libneatdispatchercore.so.[0-9]*' -print | sort
  )
  if [[ "${#runtime_files[@]}" -ne 1 ]]; then
    echo "Expected exactly one versioned private dispatcher runtime, found ${#runtime_files[@]} in ${runtime_dir}." >&2
    return 1
  fi
  runtime_file="${runtime_files[0]}"
  soname="$(LC_ALL=C readelf -d "${runtime_file}" 2>/dev/null |
    sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p' | head -n1)"
  if [[ ! "${soname}" =~ ^libneatdispatchercore\.so\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Private dispatcher must have a versioned SONAME; got ${soname:-<missing>} from ${runtime_file}." >&2
    return 1
  fi
  soname_path="${runtime_dir}/${soname}"
  resolved="$(readlink -f "${soname_path}" 2>/dev/null || true)"
  if [[ "${resolved}" != "${runtime_file}" ]]; then
    echo "Private dispatcher SONAME path does not resolve to its packaged runtime." >&2
    echo "  SONAME path: ${soname_path}" >&2
    echo "  expected:    ${runtime_file}" >&2
    echo "  resolved:    ${resolved:-<missing>}" >&2
    return 1
  fi
  if ! dispatcher_path_is_owned_by_neat_runtime "${runtime_file}" ||
      ! dispatcher_path_is_owned_by_neat_runtime "${soname_path}"; then
    echo "Private dispatcher runtime and SONAME link must both be owned by neat-runtime." >&2
    return 1
  fi

  collect_stale_global_dispatcher_paths stale_paths
  if [[ "${#stale_paths[@]}" -gt 0 ]]; then
    echo "Stale global dispatcher paths remain in a loader directory:" >&2
    printf '  %s\n' "${stale_paths[@]}" >&2
    return 1
  fi
  log "Verified versioned package-owned dispatcher ${soname} resolves only from ${runtime_dir}."
}

sima_neat_global_lib_dir() {
  local versioned_lib
  versioned_lib="$(find_packaged_sima_neat_versioned_lib)" || return 1
  dirname "${versioned_lib}"
}

find_packaged_sima_neat_versioned_lib() {
  local package_files candidate basename
  local -a matches=()
  package_files="$(dpkg-query -L sima-neat 2>/dev/null)" || return 1

  # The installed package, not the host triplet or a guessed libdir, owns
  # this location (lib, lib64 and multiarch layouts are all supported).
  while IFS= read -r candidate; do
    basename="$(basename "${candidate}")"
    if [[ "${basename}" =~ ^libsima_neat\.so\.[0-9]+\.[0-9]+(\.[0-9]+)*$ ]]; then
      matches+=("${candidate}")
    fi
  done <<<"${package_files}"

  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one packaged versioned libsima_neat library, found ${#matches[@]}." >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

find_packaged_sima_neat_soname_link() {
  local lib_dir package_files candidate basename
  local -a matches=()
  lib_dir="$(sima_neat_global_lib_dir)" || return 1
  package_files="$(dpkg-query -L sima-neat 2>/dev/null)" || return 1

  while IFS= read -r candidate; do
    [[ "$(dirname "${candidate}")" == "${lib_dir}" ]] || continue
    basename="$(basename "${candidate}")"
    if [[ "${basename}" =~ ^libsima_neat\.so\.[1-9][0-9]*$ ]]; then
      matches+=("${candidate}")
    fi
  done <<<"${package_files}"

  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one packaged libsima_neat SONAME link, found ${#matches[@]}." >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

read_sima_neat_elf_soname() {
  local versioned_lib="$1"
  LC_ALL=C readelf -d "${versioned_lib}" 2>/dev/null |
    sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p' |
    head -n 1
}

sima_neat_path_is_package_owned() {
  dpkg-query -S "$1" >/dev/null 2>&1
}

quarantine_unowned_sima_neat_path() {
  local path="$1"
  [[ -e "${path}" || -L "${path}" ]] || return 0

  if sima_neat_path_is_package_owned "${path}"; then
    echo "Refusing to replace package-owned libsima_neat path: ${path}" >&2
    return 1
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local backup="${path}.bak-neat-installer-${timestamp}"
  local suffix=0
  while [[ -e "${backup}" || -L "${backup}" ]]; do
    suffix=$((suffix + 1))
    backup="${path}.bak-neat-installer-${timestamp}.${suffix}"
  done
  log "Quarantining stale unowned libsima_neat path ${path} -> ${backup}"
  run_sudo mv -f -- "${path}" "${backup}"
}

quarantine_stale_unowned_sima_neat_soname_links() {
  local expected_soname_link="$1"
  local lib_dir candidate basename nullglob_was_set=0
  lib_dir="$(sima_neat_global_lib_dir)" || return 1

  shopt -q nullglob && nullglob_was_set=1
  shopt -s nullglob
  for candidate in "${lib_dir}"/libsima_neat.so.*; do
    basename="$(basename "${candidate}")"
    [[ "${basename}" =~ ^libsima_neat\.so\.[1-9][0-9]*$ ]] || continue
    [[ "${candidate}" != "${expected_soname_link}" ]] || continue
    if sima_neat_path_is_package_owned "${candidate}"; then
      log "Preserving package-owned libsima_neat compatibility link ${candidate}"
      continue
    fi
    quarantine_unowned_sima_neat_path "${candidate}" || return 1
  done
  if [[ "${nullglob_was_set}" -eq 0 ]]; then
    shopt -u nullglob
  fi
}

ensure_sima_neat_symlink() {
  local link_path="$1"
  local link_target="$2"
  local expected_resolved="$3"
  local actual_target=""
  local actual_resolved=""

  if [[ -L "${link_path}" ]]; then
    actual_target="$(readlink "${link_path}")"
    actual_resolved="$(readlink -f "${link_path}" 2>/dev/null || true)"
    if [[ "${actual_target}" == "${link_target}" &&
          "${actual_resolved}" == "${expected_resolved}" ]]; then
      return 0
    fi
    if sima_neat_path_is_package_owned "${link_path}"; then
      log "Repairing package-owned symlink ${link_path} -> ${link_target}"
      run_sudo ln -sfn -- "${link_target}" "${link_path}"
      return 0
    fi
  fi

  quarantine_unowned_sima_neat_path "${link_path}" || return 1
  log "Repairing ${link_path} -> ${link_target}"
  run_sudo ln -s -- "${link_target}" "${link_path}"
}

repair_global_sima_neat_lib_links() {
  local versioned_lib soname_link soname_target devel_link elf_soname=""
  versioned_lib="$(find_packaged_sima_neat_versioned_lib)" || return 1
  soname_link="$(find_packaged_sima_neat_soname_link)" || return 1
  devel_link="$(sima_neat_global_lib_dir)/libsima_neat.so"

  if [[ ! -f "${versioned_lib}" || -L "${versioned_lib}" ]]; then
    echo "Packaged versioned libsima_neat library is missing: ${versioned_lib}" >&2
    return 1
  fi

  soname_target="$(basename "${versioned_lib}")"
  if command -v readelf >/dev/null 2>&1; then
    elf_soname="$(read_sima_neat_elf_soname "${versioned_lib}")"
    if [[ -z "${elf_soname}" || "${elf_soname}" != "$(basename "${soname_link}")" ]]; then
      echo "Packaged libsima_neat SONAME does not match its package manifest." >&2
      echo "  ELF SONAME:       ${elf_soname:-<missing>}" >&2
      echo "  packaged symlink: $(basename "${soname_link}")" >&2
      return 1
    fi
  fi

  quarantine_stale_unowned_sima_neat_soname_links "${soname_link}"
  ensure_sima_neat_symlink "${soname_link}" "${soname_target}" "${versioned_lib}"
  ensure_sima_neat_symlink "${devel_link}" "$(basename "${soname_link}")" "${versioned_lib}"
}

verify_global_sima_neat_lib_links() {
  local versioned_lib soname_link soname_resolved devel_link devel_resolved elf_soname=""
  versioned_lib="$(find_packaged_sima_neat_versioned_lib)" || exit 1
  soname_link="$(find_packaged_sima_neat_soname_link)" || exit 1
  devel_link="$(sima_neat_global_lib_dir)/libsima_neat.so"

  if [[ ! -f "${versioned_lib}" || -L "${versioned_lib}" ]]; then
    echo "Packaged versioned libsima_neat library is missing: ${versioned_lib}" >&2
    exit 1
  fi

  if command -v readelf >/dev/null 2>&1; then
    elf_soname="$(read_sima_neat_elf_soname "${versioned_lib}")"
    if [[ -z "${elf_soname}" || "${elf_soname}" != "$(basename "${soname_link}")" ]]; then
      echo "Packaged libsima_neat SONAME does not match its package manifest." >&2
      exit 1
    fi
  fi

  soname_resolved="$(readlink -f "${soname_link}" 2>/dev/null || true)"
  if [[ "${soname_resolved}" != "${versioned_lib}" || "${soname_resolved}" == *".bak"* ]]; then
    echo "${soname_link} does not resolve to the packaged library." >&2
    echo "  expected: ${versioned_lib}" >&2
    echo "  actual:   ${soname_resolved:-<missing>}" >&2
    exit 1
  fi

  devel_resolved="$(readlink -f "${devel_link}" 2>/dev/null || true)"
  if [[ "${devel_resolved}" != "${versioned_lib}" ]]; then
    echo "${devel_link} does not resolve to the packaged library." >&2
    echo "  expected: ${versioned_lib}" >&2
    echo "  actual:   ${devel_resolved:-<missing>}" >&2
    exit 1
  fi

  log "Verified $(basename "${soname_link}") and libsima_neat.so resolve to ${versioned_lib}"
}

complete_board_install_after_packages() {
  local direct_platform_before="${1:-}" direct_platform_after
  verify_board_platform_admission --installed || return 1
  if board_runtime_is_legacy; then
    migrate_stale_global_dispatcher_libs
    verify_private_dispatcher_runtime
  fi
  repair_global_sima_neat_lib_links
  verify_global_sima_neat_lib_links
  verify_canonical_palette_installation
  activate_board_runtime_after_install || return $?
  restart_board_codec_services
  verify_board_codec_services
  if [[ -n "${direct_platform_before}" ]]; then
    direct_platform_after="$(capture_direct_platform_identity)" || return 1
    if [[ "${direct_platform_after}" != "${direct_platform_before}" ]]; then
      echo "Native platform identity changed during direct-runtime installation." >&2
      return 1
    fi
  fi
}

validate_ros2_sdk_native_host() {
  local os_id os_version os_codename dpkg_arch machine command_name

  for command_name in apt-get dpkg dpkg-deb dpkg-query; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "${command_name} is required for native ROS2 SDK installation." >&2
      return 1
    fi
  done

  if [[ ! -f "${NEAT_OS_RELEASE_FILE}" ]]; then
    echo "Cannot verify the ROS2 SDK operating system: missing ${NEAT_OS_RELEASE_FILE}." >&2
    return 1
  fi
  os_id="$(read_metadata_field "${NEAT_OS_RELEASE_FILE}" "ID")"
  os_version="$(read_metadata_field "${NEAT_OS_RELEASE_FILE}" "VERSION_ID")"
  os_codename="$(read_metadata_field "${NEAT_OS_RELEASE_FILE}" "VERSION_CODENAME")"
  if [[ "${os_id}" != "debian" ||
        ( "${os_version}" != "12" && "${os_codename}" != "bookworm" ) ]]; then
    echo "Native ROS2 SDK installation requires Debian 12 (bookworm)." >&2
    echo "  Detected: ID=${os_id:-<missing>} VERSION_ID=${os_version:-<missing>} VERSION_CODENAME=${os_codename:-<missing>}" >&2
    return 1
  fi

  dpkg_arch="$(dpkg --print-architecture 2>/dev/null || true)"
  machine="$(uname -m 2>/dev/null || true)"
  if [[ "${dpkg_arch}" != "arm64" || "${machine}" != "aarch64" ]]; then
    echo "Native ROS2 SDK installation requires an ARM64 host environment." >&2
    echo "  Detected: dpkg=${dpkg_arch:-<missing>} uname=${machine:-<missing>}" >&2
    return 1
  fi
}

validate_ros2_sdk_deb_architectures() {
  local deb architecture
  for deb in "${DEBS[@]}"; do
    architecture="$(dpkg-deb -f "${deb}" Architecture 2>/dev/null || true)"
    case "${architecture}" in
      arm64|all) ;;
      *)
        echo "ROS2 SDK package architecture must be arm64 or all." >&2
        echo "  Package: ${deb}" >&2
        echo "  Architecture: ${architecture:-<missing>}" >&2
        return 1
        ;;
    esac
  done
}

verify_ros2_sdk_deb_packages_installed() {
  local deb package expected_version installed_status installed_version
  for deb in "${DEBS[@]}"; do
    package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    expected_version="$(dpkg-deb -f "${deb}" Version 2>/dev/null || true)"
    installed_status="$(dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null || true)"
    installed_version="$(dpkg-query -W -f='${Version}' "${package}" 2>/dev/null || true)"
    if [[ -z "${package}" || -z "${expected_version}" ||
          "${installed_status}" != "ii " ||
          "${installed_version}" != "${expected_version}" ]]; then
      echo "Native ROS2 SDK package verification failed." >&2
      echo "  Package: ${package:-<missing>} (${deb})" >&2
      echo "  Expected version: ${expected_version:-<missing>}" >&2
      echo "  Installed status/version: ${installed_status:-<missing>} / ${installed_version:-<missing>}" >&2
      return 1
    fi
  done
  log "Verified native ROS2 SDK packages are registered with dpkg at the bundled versions."
}

ros2_sdk_tvm_runtime_is_available() {
  [[ -x /sbin/ldconfig ]] &&
    /sbin/ldconfig -p 2>/dev/null | grep -qE 'libtvm_runtime\.so .* => /(usr/)?lib/'
}

validate_ros2_sdk_tvm_runtime() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  local source_lib="${sysroot}/usr/lib/libtvm_runtime.so"

  if ros2_sdk_tvm_runtime_is_available; then
    return 0
  fi

  if [[ ! -f "${source_lib}" ]]; then
    echo "The ROS2 SDK does not provide the native TVM runtime required by Neat." >&2
    echo "  Expected: ${source_lib}" >&2
    return 1
  fi

  if command -v readelf >/dev/null 2>&1 &&
     ! readelf -h "${source_lib}" 2>/dev/null | grep -q 'Machine:.*AArch64'; then
    echo "The ROS2 SDK TVM runtime is not an AArch64 library: ${source_lib}" >&2
    return 1
  fi
}

install_ros2_sdk_tvm_runtime() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  local source_lib="${sysroot}/usr/lib/libtvm_runtime.so"
  local target_dir="/usr/lib/aarch64-linux-gnu"
  local target_lib="${target_dir}/libtvm_runtime.so"

  validate_ros2_sdk_tvm_runtime || return 1
  if ros2_sdk_tvm_runtime_is_available; then
    log "Native TVM runtime is already available to the dynamic linker."
    return 0
  fi

  run_sudo install -d -m 0755 "${target_dir}"
  run_sudo install -m 0755 "${source_lib}" "${target_lib}"
  log "Installed the ROS2 SDK TVM runtime into ${target_lib}."
}

# Preserve the selected local archives and exact target versions for real reinstalls.
collect_exact_deb_install_specs() {
  local -n out_install_specs="$1"
  out_install_specs=()
  local -A seen_install_specs=()
  local spec
  for spec in "${DEBS[@]}"; do
    [[ -n "${spec}" ]] || continue
    if [[ -n "${seen_install_specs[${spec}]+x}" ]]; then
      continue
    fi
    seen_install_specs["${spec}"]=1
    out_install_specs+=("${spec}")
  done

  # APT's volatile local-file selection can ignore --reinstall when the
  # installed version is identical. Explicit exact-version selections make
  # reinstall apply while the accompanying DEBs supply the local archives.
  local deb package version architecture package_key exact_spec
  local -A selected_package_specs=()
  for deb in "${DEBS[@]}"; do
    package="$(dpkg-deb -f "${deb}" Package)" || return 1
    version="$(dpkg-deb -f "${deb}" Version)" || return 1
    architecture="$(dpkg-deb -f "${deb}" Architecture)" || return 1
    if [[ -z "${package}" || -z "${version}" || -z "${architecture}" ]]; then
      echo "Incomplete package identity in local DEB: ${deb}" >&2
      return 1
    fi
    package_key="${package}"
    [[ "${architecture}" == all ]] || package_key+=":${architecture}"
    exact_spec="${package_key}=${version}"
    if [[ -n "${selected_package_specs[${package_key}]+x}" &&
          "${selected_package_specs[${package_key}]}" != "${exact_spec}" ]]; then
      echo "Conflicting local DEB versions for ${package_key}." >&2
      return 1
    fi
    selected_package_specs["${package_key}"]="${exact_spec}"
    if [[ -z "${seen_install_specs[${exact_spec}]+x}" ]]; then
      seen_install_specs["${exact_spec}"]=1
      out_install_specs+=("${exact_spec}")
    fi
  done
}

install_debs_in_ros2_sdk() {
  local simulation_log
  local -a install_specs=()
  local -a apt_install_args=(
    apt-get install -y --reinstall --no-remove --allow-downgrades
  )

  validate_ros2_sdk_native_host || exit 1
  validate_single_sima_neat_package_pair || exit 1
  validate_ros2_sdk_deb_architectures || exit 1
  validate_ros2_sdk_tvm_runtime || exit 1

  log "Detected native ROS2 SDK environment; installing DEBs into system paths with apt."
  printf '[install_neat_framework] DEB install set:\n'
  printf '  %s\n' "${DEBS[@]}"
  refresh_apt_metadata_for_board_install

  if ! apt_package_database_is_healthy; then
    echo "APT package state is unhealthy; refusing the native ROS2 SDK installation." >&2
    echo "Repair the container package database first, then rerun this installer." >&2
    exit 1
  fi

  collect_exact_deb_install_specs install_specs || return 1
  simulation_log="$(mktemp /tmp/sima-neat-ros2-sdk-apt-simulation-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${simulation_log}")
  log "Simulating native ROS2 SDK package installation with package removal disabled."
  if ! run_sudo "${apt_install_args[@]}" --simulate "${install_specs[@]}" >"${simulation_log}" 2>&1; then
    cat "${simulation_log}" >&2
    echo "APT rejected the native ROS2 SDK package set; no packages were changed." >&2
    exit 1
  fi
  cat "${simulation_log}"
  if grep -q '^Remv[[:space:]]' "${simulation_log}"; then
    echo "APT simulation planned package removal in the ROS2 SDK; refusing to continue." >&2
    grep '^Remv[[:space:]]' "${simulation_log}" >&2
    exit 1
  fi
  verify_scheduler_transaction "${simulation_log}" || return 1

  run_sudo "${apt_install_args[@]}" "${install_specs[@]}"
  install_ros2_sdk_tvm_runtime || exit 1
  if [[ -x /sbin/ldconfig ]]; then
    run_sudo /sbin/ldconfig
  elif command -v ldconfig >/dev/null 2>&1; then
    run_sudo ldconfig
  fi
  verify_ros2_sdk_deb_packages_installed || exit 1
  repair_global_sima_neat_lib_links
  verify_global_sima_neat_lib_links
  if ! apt_package_database_is_healthy; then
    echo "APT dependency check failed after the native ROS2 SDK installation." >&2
    exit 1
  fi
}

# Scheduler owns provider classification and ELF compatibility. This thin
# delegate authenticates its incoming installed tooling before executing it.
prepare_board_platform_admission() {
  if [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]]; then
    verify_board_platform_admission
    return $?
  fi
  if [[ ! -f "${SCHEDULER_PLATFORM_ADMISSION_PATH}" || -L "${SCHEDULER_PLATFORM_ADMISSION_PATH}" ]]; then
    echo "Missing packaged Scheduler Platform admission delegate." >&2
    return 1
  fi
  BOARD_PLATFORM_ADMISSION_DIR="$(mktemp -d /tmp/sima-neat-platform-admission-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${BOARD_PLATFORM_ADMISSION_DIR}")
  python3 -B "${SCHEDULER_PLATFORM_ADMISSION_PATH}" preflight-if-declared --directory "$PWD" \
    --prepare "${BOARD_PLATFORM_ADMISSION_DIR}/prepared" \
    --output "${BOARD_PLATFORM_ADMISSION_DIR}/before.json"
}

board_platform_admission_is_active() {
  [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]] || return 1
  python3 - "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" <<'PYACTIVE'
import json, sys
with open(sys.argv[1]) as stream:
    raise SystemExit(0 if json.load(stream)['platform_packages'] else 1)
PYACTIVE
}

board_has_scheduler_cohort() {
  [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]] || return 1
  python3 - "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" <<'PYSPLIT'
import json, sys
with open(sys.argv[1]) as stream:
    raise SystemExit(0 if json.load(stream).get('split_cohort') is True else 1)
PYSPLIT
}

required_board_firmware_control() {
  [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]] || return 0
  # This declaration comes from the authenticated incoming Scheduler tooling.
  # Scheduler owns EV/platform selection and firmware image/cohort validation.
  python3 - "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" <<'PYFIRMWARE'
import json, sys
with open(sys.argv[1]) as stream:
    admission = json.load(stream)
if admission.get('split_cohort') is True:
    firmware = admission['firmware']
    if not isinstance(firmware.get('required'), bool):
        raise SystemExit('Invalid authenticated firmware requirement')
    if firmware['required']:
        control = firmware['debian_package']
        print('\t'.join(control[key] for key in ('Package', 'Version', 'Architecture', 'deb_sha256')))
    elif firmware['debian_package'] is not None:
        raise SystemExit('Unexpected firmware package for a non-EV cohort')
PYFIRMWARE
}

verify_board_platform_admission() {
  [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]] || return 0
  # APT supplies ordinary package dependencies. Scheduler admits protected
  # providers before the transaction; full installed closure precedes activation.
  local action=preflight-if-declared
  if [[ "${1:-}" == --installed ]]; then
    action=check-if-declared
  fi
  python3 -B "${SCHEDULER_PLATFORM_ADMISSION_PATH}" "${action}" --directory "$PWD" \
    --baseline "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" "$@" >/dev/null
}

verify_board_platform_plan() {
  [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]] || return 0
  python3 -B "${SCHEDULER_PLATFORM_ADMISSION_PATH}" verify-plan \
    --baseline "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" --simulation "$1"
}

verify_scheduler_transaction() {
  local simulation="$1"
  shift
  local -a prepared_args=()
  if [[ -n "${BOARD_PLATFORM_ADMISSION_DIR}" ]]; then
    prepared_args=(--prepared-from "${BOARD_PLATFORM_ADMISSION_DIR}/before.json")
  fi
  python3 -B "${SCHEDULER_PLATFORM_ADMISSION_PATH}" transaction-if-declared \
    --directory "$PWD" --simulation "${simulation}" "${prepared_args[@]}" "$@" >/dev/null
}

install_debs_on_board() {
  log "Detected Modalix board environment; installing DEBs with apt."
  printf '[install_neat_framework] DEB install set:\n'
  printf '  %s\n' "${DEBS[@]}"
  prepare_board_platform_admission || return 1
  local direct_platform_before=""
  if ! board_runtime_is_legacy; then
    direct_platform_before="$(capture_direct_platform_identity)" || return 1
  fi
  refresh_apt_metadata_for_board_install
  stop_board_runtime_before_install

  # Only the authenticated split coordinator may admit continuation of its own
  # incomplete product cohort. Scheduler distinguishes normal deferred triggers
  # from unrelated unfinished package installation/configuration.
  local repair_incomplete=0
  if ! apt_package_database_is_healthy; then
    repair_incomplete=1
  fi
  if board_has_scheduler_cohort; then
    local dpkg_state=0
    dpkg_package_database_is_configured || dpkg_state=$?
    case "${dpkg_state}" in
      0) ;;
      1) repair_incomplete=1 ;;
      *) echo "Cannot read dpkg package states; refusing split-cohort installation." >&2; return 1 ;;
    esac
  elif [[ "${repair_incomplete}" == 1 ]]; then
    echo "APT package state is unhealthy; refusing to install the Neat package set." >&2
    echo "Repair the board package database first, then rerun this installer." >&2
    exit 1
  fi
  if [[ "${repair_incomplete}" == 1 ]]; then
    log "Incomplete package state: requiring exact split-cohort recovery and declared retirement admission."
  fi

  local -a board_install_specs=()
  collect_exact_deb_install_specs board_install_specs || return 1

  local -a board_heal_specs=()
  mapfile -t board_heal_specs < <(collect_board_heal_specs)
  if [[ "${#board_heal_specs[@]}" -gt 0 ]]; then
    log "Pinning platform packages for the retirement transaction:"
    printf '  %s\n' "${board_heal_specs[@]}"
    board_install_specs+=("${board_heal_specs[@]}")
  fi

  # File ownership transitions must be declared by the packages' Breaks/Replaces
  # metadata. Removal simulation alone cannot detect undeclared file collisions;
  # keep dpkg's ownership rejection enabled for the real transaction.
  local -a apt_install_args=(
    apt-get install -y --fix-broken --allow-downgrades
  )
  local -a transaction_admission_args=()
  if [[ "${repair_incomplete}" == 1 ]]; then
    # APT may retain broken predecessor packages unless retirement is explicit.
    # Scheduler proposes current, configured identities; the existing Core and
    # Platform guards must approve them before any removal selector is added.
    local retirement_plan
    retirement_plan="$(mktemp /tmp/sima-neat-retirements-XXXXXX)"
    INSTALLER_TMP_DIRS+=("${retirement_plan}")
    python3 -B "${SCHEDULER_PLATFORM_ADMISSION_PATH}" retirements --directory "$PWD" \
      --prepared-from "${BOARD_PLATFORM_ADMISSION_DIR}/before.json" \
      >"${retirement_plan}" || return 1
    verify_board_platform_plan "${retirement_plan}" || return 1
    verify_simulated_package_removals "${retirement_plan}" "${board_install_specs[@]}" || return 1
    local -a retirement_names=()
    mapfile -t retirement_names < <(awk '$1 == "Remv" {print $2}' "${retirement_plan}")
    if [[ "${#retirement_names[@]}" == 0 ]]; then
      apt_install_args+=(--no-remove)
    else
      local retired
      for retired in "${retirement_names[@]}"; do
        board_install_specs+=("${retired}-")
      done
    fi
    # APT 3 cannot fetch a forced-reinstall archive for an already-unpacked
    # same-version package. First resume only the producer-admitted full cohort;
    # then perform a fresh strict reinstall before declaring installation done.
    transaction_admission_args+=(--repair-incomplete --resume-incomplete)
  else
    apt_install_args+=(--reinstall)
  fi

  local simulation_log
  simulation_log="$(mktemp /tmp/sima-neat-apt-simulate-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${simulation_log}")
  if ! run_sudo "${apt_install_args[@]}" --simulate \
      "${board_install_specs[@]}" >"${simulation_log}" 2>&1; then
    cat "${simulation_log}" >&2
    echo "APT cannot satisfy the bundled Neat package transaction." >&2
    exit 1
  fi
  if ! verify_board_platform_plan "${simulation_log}" ||
     ! verify_simulated_package_removals \
      "${simulation_log}" "${board_install_specs[@]}"; then
    exit 1
  fi

  verify_scheduler_transaction "${simulation_log}" "${transaction_admission_args[@]}" || return 1

  if run_sudo "${apt_install_args[@]}" "${board_install_specs[@]}"; then
    if [[ "${repair_incomplete}" == 1 ]]; then
      # Configuration is only preparation, not successful installation. Never
      # reuse retirement selectors or the permissive resume admission for the
      # final exact-byte reinstall of every selected package.
      apt_package_database_is_healthy || return 1
      dpkg_package_database_is_configured before-reinstall || return 1
      verify_board_platform_admission || return 1
      collect_exact_deb_install_specs board_install_specs || return 1
      apt_install_args=(apt-get install -y --fix-broken --allow-downgrades --reinstall --no-remove)
      log "Resumed package transaction; requiring a fresh strict full-cohort reinstall."
      if ! run_sudo "${apt_install_args[@]}" --simulate "${board_install_specs[@]}" \
          >"${simulation_log}" 2>&1; then
        cat "${simulation_log}" >&2
        echo "APT rejected the final recovery reinstall; installation is incomplete." >&2
        return 1
      fi
      verify_board_platform_plan "${simulation_log}" || return 1
      verify_simulated_package_removals "${simulation_log}" "${board_install_specs[@]}" || return 1
      verify_scheduler_transaction "${simulation_log}" --repair-incomplete || return 1
      if ! run_sudo "${apt_install_args[@]}" "${board_install_specs[@]}"; then
        echo "Final recovery reinstall failed; refusing package removal or direct dpkg fallback." >&2
        return 1
      fi
      dpkg_package_database_is_configured || return 1
    fi
    run_sudo apt-get check || return 1
    complete_board_install_after_packages "${direct_platform_before}"
    return $?
  fi

  if [[ "${repair_incomplete}" == 1 ]]; then
    echo "Exact split-cohort recovery failed; refusing further package removal or direct dpkg fallback." >&2
    return 1
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}" != "ON" ]]; then
    echo "apt-get rejected the local package set; refusing to remove the installed runtime because NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}." >&2
    echo "Fix the bundled package versions/dependencies, or explicitly set NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON during a recoverable maintenance operation." >&2
    exit 1
  fi

  log "apt-get install failed; NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON, removing installed NEAT packages represented by the local DEB set and retrying apt."
  remove_installed_local_deb_packages || return 1
  if board_platform_admission_is_active || board_has_scheduler_cohort; then
    verify_board_platform_admission || return 1
    # The first failed transaction/removal may have changed the package state.
    # Do not reuse an earlier solver plan for a maintenance retry.
    run_sudo "${apt_install_args[@]}" --simulate "${board_install_specs[@]}" \
      >"${simulation_log}" 2>&1 || return 1
    verify_board_platform_plan "${simulation_log}" || return 1
    verify_simulated_package_removals "${simulation_log}" "${board_install_specs[@]}" || return 1
    verify_scheduler_transaction "${simulation_log}" || return 1
  fi
  if run_sudo "${apt_install_args[@]}" "${board_install_specs[@]}"; then
    complete_board_install_after_packages "${direct_platform_before}"
    return $?
  fi

  if board_has_scheduler_cohort; then
    echo "Split-cohort APT retry failed; refusing direct dpkg installation without a valid transaction." >&2
    return 1
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}" != "ON" ]]; then
    echo "apt-get install failed and NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}; refusing direct dpkg fallback." >&2
    exit 1
  else
    log "apt-get install failed; retrying with direct dpkg install of the local NEAT DEB set."
  fi

  # Opting into maintenance fallback does not waive file ownership checks.
  run_sudo dpkg -i "${DEBS[@]}"
  run_sudo apt-get check
  complete_board_install_after_packages "${direct_platform_before}"
}

remove_stale_global_sima_lmm_pip_install() {
  if ! command -v pip3 >/dev/null 2>&1; then
    return 0
  fi

  if run_sudo pip3 show sima_lmm >/dev/null 2>&1; then
    log "Removing stale global sima_lmm pip package before installing LLiMa DEBs."
    run_sudo pip3 uninstall -y sima_lmm --break-system-packages
  fi
}

ensure_python_venv_support() {
  python3 -c 'import ensurepip' >/dev/null 2>&1 && return 0

  log "Python venv support (ensurepip) is missing; installing python3-venv with apt."
  refresh_apt_metadata_for_board_install
  if ! run_sudo apt-get install -y python3-venv; then
    echo "Failed to install python3-venv. Run 'sudo apt-get update && sudo apt-get install -y python3-venv', then rerun this installer." >&2
    exit 1
  fi
}

prepare_python_venv() {
  local venv_dir="$1"
  local interpreter
  # venv leaves existing interpreter symlinks untouched, even when their
  # original SDK/system Python has gone away. Repair only dangling aliases;
  # retain site-packages, configuration and working interpreter links.
  for interpreter in "${venv_dir}/bin/python" "${venv_dir}/bin/python3" "${venv_dir}"/bin/python3.[0-9]*; do
    if [[ -L "${interpreter}" && ! -e "${interpreter}" ]]; then
      log "Removing dangling virtualenv interpreter link: ${interpreter}"
      rm -- "${interpreter}"
    fi
  done
  python3 -m venv --system-site-packages "${venv_dir}"
  if ! "${venv_dir}/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix'; then
    echo "Virtualenv interpreter is unusable after preparation: ${venv_dir}/bin/python" >&2
    return 1
  fi
}

install_python_environment() {
  ensure_python_venv_support
  VENV_DIR="$(resolve_venv_dir)"
  ACTIVATE_PATH="$(activation_path_for_display "${VENV_DIR}")"
  remove_stale_global_sima_lmm_pip_install
  log_green "Preparing Python virtual environment at ${VENV_DIR}"
  mkdir -p "$(dirname "${VENV_DIR}")"
  prepare_python_venv "${VENV_DIR}"
  ensure_home_pyneat_symlink "${VENV_DIR}"
  print_green_banner "${VENV_DIR}" "${ACTIVATE_PATH}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip

  WHEEL_FILES=()
  collect_wheel_files "." WHEEL_FILES
  WHEEL_FILE="${WHEEL_FILES[0]:-}"
  if [[ -z "${WHEEL_FILE}" ]]; then
    echo "No wheel file found in current directory." >&2
    exit 1
  fi
  "${VENV_DIR}/bin/python" -m pip install --no-deps --force-reinstall "${WHEEL_FILE}"
}

validate_single_sima_neat_package_pair() {
  local deb package version
  local -a core_debs=()
  local -a core_versions=()
  local -a dev_debs=()
  local -a dev_versions=()

  for deb in "${DEBS[@]}"; do
    package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    case "${package}" in
      sima-neat)
        core_debs+=("${deb}")
        core_versions+=("$(dpkg-deb -f "${deb}" Version 2>/dev/null || true)")
        ;;
      sima-neat-dev)
        dev_debs+=("${deb}")
        dev_versions+=("$(dpkg-deb -f "${deb}" Version 2>/dev/null || true)")
        ;;
    esac
  done

  if [[ "${#core_debs[@]}" -ne 1 || "${#dev_debs[@]}" -ne 1 ]]; then
    echo "Neat installation requires exactly one sima-neat and one sima-neat-dev package." >&2
    echo "  sima-neat packages:     ${#core_debs[@]}" >&2
    echo "  sima-neat-dev packages: ${#dev_debs[@]}" >&2
    echo "Remove stale package versions or install from a generated metadata/manifest bundle." >&2
    return 1
  fi
  if [[ -z "${core_versions[0]}" || -z "${dev_versions[0]}" ||
        "${core_versions[0]}" != "${dev_versions[0]}" ]]; then
    echo "Neat installation sima-neat package versions do not match." >&2
    echo "  sima-neat:     ${core_versions[0]:-<missing>} (${core_debs[0]})" >&2
    echo "  sima-neat-dev: ${dev_versions[0]:-<missing>} (${dev_debs[0]})" >&2
    return 1
  fi
}

collect_current_bundle_sima_neat_lib_paths() {
  local sysroot="$1"
  local -n out_paths="$2"
  local deb package entry normalized basename
  out_paths=()

  for deb in "${DEBS[@]}"; do
    package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    case "${package}" in
      sima-neat | sima-neat-dev) ;;
      *) continue ;;
    esac

    while IFS= read -r entry; do
      normalized="/${entry#./}"
      [[ "$(dirname "${normalized}")" == "/usr/lib" ]] || continue
      basename="$(basename "${normalized}")"
      [[ "${basename}" == libsima_neat.so* ]] || continue
      out_paths+=("${sysroot}${normalized}")
    done < <(dpkg-deb --fsys-tarfile "${deb}" | tar -tf -)
  done

  if [[ "${#out_paths[@]}" -eq 0 ]]; then
    echo "Current install bundle does not declare any libsima_neat library paths." >&2
    return 1
  fi
}

sima_neat_path_is_in_array() {
  local needle="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ "${candidate}" == "${needle}" ]] && return 0
  done
  return 1
}

repair_sysroot_sima_neat_libs() {
  local sysroot="$1"
  local lib_dir="${sysroot}/usr/lib"
  local devel_link="${lib_dir}/libsima_neat.so"
  local soname_basename soname_link versioned_basename versioned_lib elf_soname=""
  local candidate basename timestamp backup suffix nullglob_was_set=0
  local -a bundle_paths=()

  collect_current_bundle_sima_neat_lib_paths "${sysroot}" bundle_paths

  if [[ ! -L "${devel_link}" ]]; then
    echo "SDK sysroot install is missing the packaged libsima_neat.so linker symlink." >&2
    return 1
  fi
  soname_basename="$(readlink "${devel_link}")"
  if [[ ! "${soname_basename}" =~ ^libsima_neat\.so\.[1-9][0-9]*$ ]]; then
    echo "SDK sysroot libsima_neat.so has an invalid SONAME target: ${soname_basename}" >&2
    return 1
  fi
  soname_link="${lib_dir}/${soname_basename}"
  if [[ ! -L "${soname_link}" ]]; then
    echo "SDK sysroot install is missing the packaged SONAME link: ${soname_link}" >&2
    return 1
  fi
  versioned_basename="$(readlink "${soname_link}")"
  if [[ ! "${versioned_basename}" =~ ^libsima_neat\.so\.[0-9]+\.[0-9]+(\.[0-9]+)*$ ]]; then
    echo "SDK sysroot SONAME link has an invalid target: ${versioned_basename}" >&2
    return 1
  fi
  versioned_lib="${lib_dir}/${versioned_basename}"
  if [[ ! -f "${versioned_lib}" || -L "${versioned_lib}" ]]; then
    echo "SDK sysroot packaged libsima_neat library is missing: ${versioned_lib}" >&2
    return 1
  fi

  for candidate in "${devel_link}" "${soname_link}" "${versioned_lib}"; do
    if ! sima_neat_path_is_in_array "${candidate}" "${bundle_paths[@]}"; then
      echo "SDK sysroot libsima_neat path is not owned by the current bundle: ${candidate}" >&2
      return 1
    fi
  done

  if command -v readelf >/dev/null 2>&1; then
    elf_soname="$(read_sima_neat_elf_soname "${versioned_lib}")"
    if [[ -z "${elf_soname}" || "${elf_soname}" != "${soname_basename}" ]]; then
      echo "SDK sysroot libsima_neat SONAME does not match the current bundle." >&2
      echo "  ELF SONAME:       ${elf_soname:-<missing>}" >&2
      echo "  packaged symlink: ${soname_basename}" >&2
      return 1
    fi
  fi

  shopt -q nullglob && nullglob_was_set=1
  shopt -s nullglob
  for candidate in "${lib_dir}"/libsima_neat.so.*; do
    basename="$(basename "${candidate}")"
    if [[ ! "${basename}" =~ ^libsima_neat\.so\.[1-9][0-9]*$ &&
          ! "${basename}" =~ ^libsima_neat\.so\.[0-9]+\.[0-9]+(\.[0-9]+)*$ ]]; then
      continue
    fi
    sima_neat_path_is_in_array "${candidate}" "${bundle_paths[@]}" && continue

    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup="${candidate}.bak-neat-installer-${timestamp}"
    suffix=0
    while [[ -e "${backup}" || -L "${backup}" ]]; do
      suffix=$((suffix + 1))
      backup="${candidate}.bak-neat-installer-${timestamp}.${suffix}"
    done
    log "Quarantining stale SDK sysroot libsima_neat path ${candidate} -> ${backup}"
    run_sudo mv -f -- "${candidate}" "${backup}"
  done
  if [[ "${nullglob_was_set}" -eq 0 ]]; then
    shopt -u nullglob
  fi

  if [[ "$(readlink -f "${devel_link}" 2>/dev/null || true)" != "${versioned_lib}" ||
        "$(readlink -f "${soname_link}" 2>/dev/null || true)" != "${versioned_lib}" ]]; then
    echo "SDK sysroot libsima_neat links do not resolve to the current bundle." >&2
    return 1
  fi
  log "Verified SDK sysroot ${soname_basename} and libsima_neat.so resolve to ${versioned_lib}"
}

install_debs_into_sysroot() {
  local sysroot
  sysroot="$(sysroot_path)"
  if [[ ! -d "${sysroot}" ]]; then
    echo "SYSROOT does not exist: ${sysroot}" >&2
    exit 1
  fi
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is required for eLxr SDK/sysroot installs." >&2
    exit 1
  fi

  log "Detected eLxr SDK environment; installing DEBs into sysroot: ${sysroot}"
  validate_single_sima_neat_package_pair
  ensure_sima_lmm_sysroot_deps "${sysroot}"
  local deb
  for deb in "${DEBS[@]}"; do
    log "Extracting $(basename "${deb}") into ${sysroot}"
    run_sudo dpkg-deb -x "${deb}" "${sysroot}"
  done

  repair_sysroot_sima_neat_libs "${sysroot}"
  cache_install_artifacts_in_sysroot
}

deploy_artifacts_to_paired_devkit_if_configured() {
  if [[ "${NEAT_INSTALLER_SKIP_DEVKIT_SYNC}" == "ON" ]]; then
    log "NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON; skipping paired DevKit sync."
    return 0
  fi

  local devkit_ip="${DEVKIT_SYNC_DEVKIT_IP:-}"
  [[ -n "${devkit_ip}" ]] || return 0

  local ssh_target="${DEVKIT_DEPLOY_USER}@${devkit_ip}"
  local remote_dir="/tmp/sima-neat-install-$(date +%Y%m%d-%H%M%S)"
  local -a CACHED_DEBS=()
  local -a CACHED_WHEELS=()
  local -a CACHED_DEPLOY_FILES=()
  collect_cached_devkit_deploy_files
  local installer_basename="install_neat_framework.sh"

  log_green "Paired DevKit detected; syncing install artifacts to ${ssh_target}"

  if ! run_ssh -o ConnectTimeout=5 "${ssh_target}" "true" >/dev/null 2>&1; then
    echo "Paired DevKit ${ssh_target} is not reachable over SSH." >&2
    if [[ "${DEVKIT_SYNC_REQUIRED}" == "ON" ]]; then
      exit 1
    fi
    log "DEVKIT_SYNC_REQUIRED=OFF; skipping DevKit sync."
    return 0
  fi

  run_ssh "${ssh_target}" "mkdir -p '${remote_dir}'"
  run_scp "${CACHED_DEPLOY_FILES[@]}" "${ssh_target}:${remote_dir}/"

  local remote_install_cmd
  remote_install_cmd="set -euo pipefail
remote_dir=$(printf '%q' "${remote_dir}")
installer_name=$(printf '%q' "${installer_basename}")
export FORCE_INSTALL=$(printf '%q' "${FORCE_INSTALL:-0}")
export NEAT_INSTALLER_SENTINEL_REQUIRED=$(printf '%q' "${NEAT_INSTALLER_SENTINEL_REQUIRED}")
cleanup_remote_artifacts() {
  rm -rf \"\${remote_dir}\"
}
trap cleanup_remote_artifacts EXIT
chmod +x \"\${remote_dir}/\${installer_name}\"
cd \"\${remote_dir}\"
NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON bash \"./\${installer_name}\" --local"

  run_ssh -t "${ssh_target}" "bash -lc $(printf '%q' "${remote_install_cmd}")"

  log_green "Paired DevKit sync completed: ${ssh_target}"
}

install_for_environment() {
  case "${ENV_MODE}" in
    elxr-sdk)
      install_debs_into_sysroot
      ensure_sdk_neat_cli_symlink
      install_agent_skills_for_current_user "${SYSROOT:-/opt/toolchain/aarch64/modalix}/usr/share/sima-neat/skills/sima-neat"
      deploy_artifacts_to_paired_devkit_if_configured
      ;;
    ros2-sdk)
      # The ROS2 SDK is a native C++ build environment. PyNeat remains a DevKit
      # runtime feature and is intentionally not installed here.
      install_debs_in_ros2_sdk
      install_agent_skills_for_current_user "/usr/share/sima-neat/skills/sima-neat"
      ;;
    modalix-board)
      # Sentinel comes first so a required Sentinel that fails to install
      # leaves the installed Neat packages untouched. Then preserve the
      # established board ordering: provision PyNeat before the board-specific
      # package recovery and runtime restart transaction.
      install_board_sentinel
      install_python_environment
      install_debs_on_board
      install_agent_skills_for_current_user "/usr/share/sima-neat/skills/sima-neat"
      ;;
    *)
      echo "Unsupported installation environment: ${ENV_MODE}" >&2
      return 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

parse_args "$@"

stage_bundle_in_fresh_dir
collect_debs_in_install_order "." DEBS
if [[ "${#DEBS[@]}" -lt 1 ]]; then
  echo "No required DEB files found in current directory." >&2
  exit 1
fi

ENV_MODE="$(detect_env_mode)"
log_green "Environment mode: ${ENV_MODE}"
ensure_platform_compatible
install_for_environment

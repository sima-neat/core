#!/usr/bin/env bash
set -euo pipefail

# install_neat_framework.sh
#
# Purpose:
# - Install SiMa NEAT wheel into a Python virtual environment (board mode only).
# - Install NEAT runtime .deb packages on Modalix board or into eLxr SDK sysroot.
#
# Behavior:
# - Auto-detects environment:
#   - eLxr SDK when /etc/sdk-release exists, or SYSROOT points to a valid directory.
#   - Modalix board when /etc/buildinfo reports MACHINE=modalix.
# - In Modalix board mode, installs .deb packages with apt (sudo).
# - In eLxr SDK mode, caches install artifacts under the sysroot
#   neat-install-packages folder for paired DevKit sync.
# - In eLxr SDK mode, exposes the sysroot-installed neat command at
#   /usr/local/bin/neat for convenient use inside the SDK container.
# - On devices with writable /media/nvme, creates the venv at /media/nvme/pyneat and
#   exposes it via $HOME/pyneat for consistent activation instructions.
# - On devices without /media/nvme, creates the venv directly at $HOME/pyneat.
#
# Expected working directory:
# - Directory containing:
#   - one .whl file
#   - sima-neat-*-Linux-core.deb
#   - sima-neat-*-Linux-dev.deb
#   - neat-*.deb / simaai-common*.deb runtime dependencies
#   - appcomplex_*.deb legacy runtime dependency packages when present
#   - sima-lmm-*-Linux-core.deb / sima-lmm-*-Linux-cli.deb / sima-lmm-*-Linux-dev.deb
#   - neat-install-manifest.txt when installed from a packaged release
#
# Environment overrides:
# - PYNEAT_VENV_DIR: Python virtualenv path
# - SUDO_PASSWORD / DEVKIT_PASSWORD: sudo password (preferred non-interactive override)
# - DEFAULT_SUDO_PASSWORD: fallback password (default: edgeai)
# - SYSROOT: SDK sysroot path override (default: /opt/toolchain/aarch64/modalix)
# - ELXR_SDK_RELEASE_FILE: SDK release metadata path (default: /etc/sdk-release)
# - NEAT_BUILDINFO_FILE: DevKit build metadata path (default: /etc/buildinfo)
# - NEAT_PACKAGE_MANIFEST: package manifest filename/path (default: manifest.json)
# - NEAT_INSTALLER_SKIP_PLATFORM_CHECK: ON/OFF (default: OFF) explicit escape hatch
#   for development installs on nonstandard images.
# - DEVKIT_SYNC_DEVKIT_IP: paired DevKit IP for SDK->DevKit artifact sync
# - DEVKIT_DEPLOY_USER: DevKit SSH user (default: sima)
# - DEVKIT_SYNC_REQUIRED: ON/OFF (default: ON) fail hard if paired DevKit sync fails
# - NEAT_INSTALLER_SKIP_DEVKIT_SYNC: ON/OFF (default: OFF) skip SDK->DevKit sync
# - NEAT_INSTALL_MANIFEST: install manifest filename (default: neat-install-manifest.txt)
# - CODEX_HOME: optional Codex home override for skill install target
# - CLAUDE_HOME: optional Claude home override for skill install target
# - NEAT_INSTALLER_INSTALL_CODEX_SKILL: ON/OFF (default: ON)
# - NEAT_INSTALLER_INSTALL_CLAUDE_SKILL: ON/OFF (default: ON)
# - NEAT_INSTALLER_RELAX_SIMA_LMM_DEP: ON/OFF (default: ON) relax sima-neat's
#   lower sima-lmm-core/dev dependency to the bundled local LMM DEB version when
#   both packages are from the same minor-version family.
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
#   staged EV74 firmware and reset runtime state after board package replacement.

SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"
DEVKIT_DEPLOY_USER="${DEVKIT_DEPLOY_USER:-sima}"
DEVKIT_SYNC_REQUIRED="${DEVKIT_SYNC_REQUIRED:-ON}"
NEAT_INSTALLER_SKIP_DEVKIT_SYNC="${NEAT_INSTALLER_SKIP_DEVKIT_SYNC:-OFF}"
NEAT_INSTALL_MANIFEST="${NEAT_INSTALL_MANIFEST:-neat-install-manifest.txt}"
NEAT_INSTALLER_INSTALL_CODEX_SKILL="${NEAT_INSTALLER_INSTALL_CODEX_SKILL:-ON}"
NEAT_INSTALLER_INSTALL_CLAUDE_SKILL="${NEAT_INSTALLER_INSTALL_CLAUDE_SKILL:-ON}"
NEAT_INSTALLER_RELAX_SIMA_LMM_DEP="${NEAT_INSTALLER_RELAX_SIMA_LMM_DEP:-ON}"
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK="${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK:-OFF}"
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL="${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL:-OFF}"
NEAT_INSTALLER_APT_UPDATE="${NEAT_INSTALLER_APT_UPDATE:-AUTO}"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD="${NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD:-ON}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
NEAT_BUILDINFO_FILE="${NEAT_BUILDINFO_FILE:-/etc/buildinfo}"
NEAT_PACKAGE_MANIFEST="${NEAT_PACKAGE_MANIFEST:-manifest.json}"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK="${NEAT_INSTALLER_SKIP_PLATFORM_CHECK:-OFF}"
INSTALLER_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
GREEN=$'\033[0;32m'
RESET=$'\033[0m'
INSTALLER_TMP_DIRS=()
SIMAAI_MEMORY_DEBS=()
SIMAAI_MEMORY_RUNTIME_DEB=""
SIMAAI_MEMORY_DEV_DEB=""
SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION=""
SIMAAI_MEMORY_ACTUAL_VERSION=""
SIMAAI_MEMORY_PAYLOAD_PATH=""
SIMAAI_MEMORY_PAYLOAD_SHA256=""
SIMAAI_MEMORY_PAYLOAD_BUILD_ID=""
SIMAAI_MEMORY_PREINSTALL_PACKAGES=""
SIMAAI_MEMORY_PREINSTALL_PALETTE_INSTALLED=0
SIMAAI_MEMORY_PREINSTALL_PALETTE_VERSION=""
SIMAAI_MEMORY_PREINSTALL_OTA_PATH=""
SIMAAI_MEMORY_TRANSACTION_COMPLETE=0

cleanup_installer_tmp_dirs() {
  local dir
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
      sima-neat | sima-neat-dev)
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
  if [[ -f "${ELXR_SDK_RELEASE_FILE}" ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -f "${NEAT_BUILDINFO_FILE}" ]] && grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' "${NEAT_BUILDINFO_FILE}"; then
    echo "modalix-board"
    return 0
  fi
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
    $1 ~ /^[[:space:]]*SDK Version[[:space:]]*$/ {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      sub(/_.*/, "", value)
      print value
      exit
    }
  ' "${release_file}" 2>/dev/null || true
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
      source_label="SDK Version"
      source_file="${ELXR_SDK_RELEASE_FILE}"
      if [[ ! -f "${source_file}" ]]; then
        echo "Cannot verify eLxr SDK compatibility: missing ${source_file}." >&2
        echo "Set ELXR_SDK_RELEASE_FILE or NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON for an explicit development override." >&2
        exit 1
      fi
      actual="$(read_sdk_platform_version "${source_file}")"
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
      echo "Unknown environment mode for platform compatibility check: ${ENV_MODE}" >&2
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

  # Install low-level runtime packages first, then LLiMa, then NEAT core/dev.
  append_matching_files "${out_array_name}" "${search_dir}" 'simaai-common*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'simaai-memory-lib_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'simaai-memory-lib-dev_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'libcamera_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'libcamera-dev_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'libcamera-tools_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-common_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-appcomplex_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'appcomplex_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-ev74-firmware_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-runtime_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-gst-plugins_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'neat-internals-dev_*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'sima-lmm-*.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'sima-neat-*-Linux-core.deb'
  append_matching_files "${out_array_name}" "${search_dir}" 'sima-neat-*-Linux-dev.deb'
}

sysroot_path() {
  printf '%s\n' "${SYSROOT:-/opt/toolchain/aarch64/modalix}"
}

sysroot_neat_install_packages_dir() {
  printf '%s\n' "$(sysroot_path)/neat-install-packages"
}

has_sima_lmm_sysroot_deps() {
  local sysroot="$1"
  [[ -f "${sysroot}/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor" &&
     -f "${sysroot}/usr/share/eigen3/cmake/Eigen3Config.cmake" &&
     -f "${sysroot}/usr/include/fmt/core.h" &&
     -f "${sysroot}/usr/lib/aarch64-linux-gnu/libfmt.so.9.1.0" &&
     -f "${sysroot}/usr/include/spdlog/spdlog.h" &&
     -f "${sysroot}/usr/lib/aarch64-linux-gnu/libspdlog.so.1.10.0" &&
     -f "${sysroot}/usr/include/nlohmann/json.hpp" &&
     -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlicommon.pc" &&
     -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlidec.pc" &&
     -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlienc.pc" &&
     -f "${sysroot}/usr/include/httplib.h" &&
     -e "${sysroot}/usr/lib/aarch64-linux-gnu/libcpp-httplib.so.0.11" ]]
}

ensure_sima_lmm_sysroot_deps() {
  local sysroot="$1"

  if ! compgen -G './sima-lmm-*.deb' >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is required to install SimaLMM SDK/sysroot dependencies." >&2
    exit 1
  fi

  local -a missing_packages=()
  if [[ ! -f "${sysroot}/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor" ||
        ! -f "${sysroot}/usr/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
    missing_packages+=("libeigen3-dev")
  fi
  if [[ ! -f "${sysroot}/usr/include/fmt/core.h" ]]; then
    missing_packages+=("libfmt-dev:arm64")
  fi
  if [[ ! -f "${sysroot}/usr/lib/aarch64-linux-gnu/libfmt.so.9.1.0" ]]; then
    missing_packages+=("libfmt9:arm64")
  fi
  if [[ ! -f "${sysroot}/usr/include/spdlog/spdlog.h" ]]; then
    missing_packages+=("libspdlog-dev:arm64")
  fi
  if [[ ! -f "${sysroot}/usr/lib/aarch64-linux-gnu/libspdlog.so.1.10.0" ]]; then
    missing_packages+=("libspdlog1.10:arm64")
  fi
  if [[ ! -f "${sysroot}/usr/include/nlohmann/json.hpp" ]]; then
    missing_packages+=("nlohmann-json3-dev")
  fi
  if [[ ! -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlicommon.pc" ||
        ! -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlidec.pc" ||
        ! -f "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlienc.pc" ]]; then
    missing_packages+=("libbrotli-dev:arm64")
  fi
  if [[ ! -f "${sysroot}/usr/include/httplib.h" ]]; then
    missing_packages+=("libcpp-httplib-dev:arm64")
  fi
  if [[ ! -e "${sysroot}/usr/lib/aarch64-linux-gnu/libcpp-httplib.so.0.11" ]]; then
    missing_packages+=("libcpp-httplib0.11:arm64")
  fi

  if [[ "${#missing_packages[@]}" -eq 0 ]]; then
    return 0
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sima-lmm-sysroot-deps-XXXXXX)"

  log "Installing SimaLMM SDK/sysroot dependencies:"
  printf '  %s\n' "${missing_packages[@]}"
  if ! (
    cd "${tmp_dir}"
    apt-get download "${missing_packages[@]}"
  ); then
    rm -rf "${tmp_dir}"
    echo "Failed to download SimaLMM SDK/sysroot dependencies." >&2
    exit 1
  fi

  local -a downloaded_debs=()
  mapfile -t downloaded_debs < <(find "${tmp_dir}" -maxdepth 1 -type f -name '*.deb' | sort)
  if [[ "${#downloaded_debs[@]}" -lt 1 ]]; then
    rm -rf "${tmp_dir}"
    echo "Failed to download SimaLMM SDK/sysroot dependencies." >&2
    exit 1
  fi

  local dep_deb
  for dep_deb in "${downloaded_debs[@]}"; do
    log "Extracting $(basename "${dep_deb}") into ${sysroot}"
    if ! dpkg-deb -x "${dep_deb}" "${sysroot}" 2>/dev/null; then
      run_sudo dpkg-deb -x "${dep_deb}" "${sysroot}"
    fi
  done
  rm -rf "${tmp_dir}"

  if ! has_sima_lmm_sysroot_deps "${sysroot}"; then
    echo "SimaLMM SDK/sysroot dependencies are still incomplete after install." >&2
    exit 1
  fi
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
    "${cache_dir}"/sima-neat-*-Linux-core.deb \
    "${cache_dir}"/sima-neat-*-Linux-dev.deb \
    "${cache_dir}"/neat-*.deb \
    "${cache_dir}"/simaai-common*.deb \
    "${cache_dir}"/simaai-memory-lib_*.deb \
    "${cache_dir}"/simaai-memory-lib-dev_*.deb \
    "${cache_dir}"/libcamera_*.deb \
    "${cache_dir}"/libcamera-dev_*.deb \
    "${cache_dir}"/libcamera-tools_*.deb \
    "${cache_dir}"/neat-common_*.deb \
    "${cache_dir}"/neat-appcomplex_*.deb \
    "${cache_dir}"/appcomplex_*.deb \
    "${cache_dir}"/sima-lmm-*.deb \
    "${cache_dir}"/*.whl \
    "${cache_dir}/${NEAT_INSTALL_MANIFEST}" \
    "${cache_dir}/${NEAT_PACKAGE_MANIFEST}" \
    "${cache_dir}"/install_neat_framework.sh

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
  if [[ ! -f "${cached_installer}" ]]; then
    echo "Cached installer script not found for paired DevKit sync: ${cached_installer}" >&2
    exit 1
  fi

  CACHED_DEPLOY_FILES=("${CACHED_DEBS[@]}" "${CACHED_WHEELS[@]}")
  if [[ -f "${cached_manifest}" ]]; then
    CACHED_DEPLOY_FILES+=("${cached_manifest}")
  fi
  if [[ -f "${cached_package_manifest}" ]]; then
    CACHED_DEPLOY_FILES+=("${cached_package_manifest}")
  fi
  CACHED_DEPLOY_FILES+=("${cached_installer}")
}

apt_package_database_is_healthy() {
  local apt_check_log
  apt_check_log="$(mktemp /tmp/sima-neat-apt-check-XXXXXX)"

  if run_sudo apt-get check >"${apt_check_log}" 2>&1; then
    rm -f "${apt_check_log}"
    return 0
  fi

  rm -f "${apt_check_log}"
  return 1
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

deb_package_is_present() {
  dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null | grep -q '^i'
}

deb_package_installed_version() {
  dpkg-query -W -f='${Version}' "$1" 2>/dev/null
}

apt_candidate_version() {
  local package="$1"
  LC_ALL=C apt-cache policy "${package}" 2>/dev/null |
    awk '/^[[:space:]]*Candidate:/ { print $2; exit }'
}

apt_exact_dependency_version() {
  local package="$1"
  local package_version="$2"
  local dependency="$3"
  local control
  control="$(LC_ALL=C apt-cache show "${package}=${package_version}" 2>/dev/null)" || return 1
  [[ -n "${control}" ]] || return 1

  python3 -c '
import re
import sys

dependency = sys.argv[1]
paragraph = sys.stdin.read().split("\n\n", 1)[0]
unfolded = re.sub(r"\n[ \t]+", " ", paragraph)
match = re.search(
    rf"(?:^|,)\s*{re.escape(dependency)}(?:\:[^\s(,|]+)?\s*"
    rf"\(\s*=\s*([^\s)]+)\s*\)",
    next((line[9:] for line in unfolded.splitlines() if line.startswith("Depends: ")), ""),
)
if match is None:
    raise SystemExit(1)
print(match.group(1))
' "${dependency}" <<<"${control}"
}

exact_dependency_version_from_relations() {
  local dependency="$1"
  python3 -c '
import re
import sys

dependency = sys.argv[1]
relations = sys.stdin.read().strip()
match = re.search(
    rf"(?:^|,)\s*{re.escape(dependency)}(?:\:[^\s(,|]+)?\s*"
    rf"\(\s*=\s*([^\s)]+)\s*\)",
    relations,
)
if match is None:
    raise SystemExit(1)
print(match.group(1))
' "${dependency}"
}

palette_required_simaai_memory_version() {
  local palette_version depends version
  if deb_package_is_installed simaai-palette-modalix; then
    palette_version="$(deb_package_installed_version simaai-palette-modalix)"
    depends="$(dpkg-query -W -f='${Depends}' simaai-palette-modalix 2>/dev/null || true)"
    version="$(exact_dependency_version_from_relations simaai-memory-lib <<<"${depends}" || true)"
    if [[ -z "${version}" ]]; then
      echo "Installed simaai-palette-modalix=${palette_version} has no exact simaai-memory-lib dependency." >&2
      return 1
    fi
    printf '%s\n' "${version}"
    return 0
  fi

  palette_version="$(apt_candidate_version simaai-palette-modalix)"
  if [[ -z "${palette_version}" || "${palette_version}" == "(none)" ]]; then
    echo "Cannot discover the platform's required simaai-memory-lib revision: simaai-palette-modalix is not installed and has no APT candidate." >&2
    return 1
  fi
  apt_exact_dependency_version simaai-palette-modalix "${palette_version}" simaai-memory-lib
}

board_debian_architecture() {
  dpkg --print-architecture
}

artifact_checksum_for_file() {
  local file="$1"
  python3 - "$(basename "${file}")" <<'PY'
import json
import re
import sys
from pathlib import Path

name = sys.argv[1]
checksums = set()
for path in sorted(Path.cwd().glob("metadata*.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid package metadata {path}: {exc}")
    value = data.get("resources-checksum", {}).get(name)
    if value:
        checksums.add(str(value).strip().lower())
if len(checksums) > 1:
    raise SystemExit(f"conflicting artifact checksums for {name}: {sorted(checksums)}")
if checksums:
    value = checksums.pop()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemExit(f"invalid SHA256 for {name}: {value}")
    print(value)
PY
}

collect_local_simaai_memory_debs() {
  local deb package arch depends provides provided_version dev_runtime_version board_arch
  local runtime_deb="" dev_deb="" runtime_version="" dev_version=""

  for deb in "${DEBS[@]}"; do
    [[ -f "${deb}" ]] || continue
    package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    case "${package}" in
      simaai-memory-lib)
        if [[ -n "${runtime_deb}" ]]; then
          echo "Artifact contains more than one simaai-memory-lib runtime DEB." >&2
          return 1
        fi
        runtime_deb="${deb}"
        ;;
      simaai-memory-lib-dev)
        if [[ -n "${dev_deb}" ]]; then
          echo "Artifact contains more than one simaai-memory-lib-dev DEB." >&2
          return 1
        fi
        dev_deb="${deb}"
        ;;
    esac
  done

  if [[ -z "${runtime_deb}" || -z "${dev_deb}" ]]; then
    echo "The board artifact must bundle exactly one local simaai-memory-lib runtime and dev DEB." >&2
    return 1
  fi

  SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION="$(palette_required_simaai_memory_version)" || return 1
  board_arch="$(board_debian_architecture)" || return 1
  runtime_version="$(dpkg-deb -f "${runtime_deb}" Version)"
  dev_version="$(dpkg-deb -f "${dev_deb}" Version)"
  if [[ -z "${runtime_version}" || "${dev_version}" != "${runtime_version}" ]]; then
    echo "Bundled memory runtime/dev package versions must match: runtime=${runtime_version:-<missing>} dev=${dev_version:-<missing>}." >&2
    return 1
  fi
  SIMAAI_MEMORY_ACTUAL_VERSION="${runtime_version}"

  for deb in "${runtime_deb}" "${dev_deb}"; do
    package="$(dpkg-deb -f "${deb}" Package)"
    arch="$(dpkg-deb -f "${deb}" Architecture)"
    if [[ "${arch}" != "${board_arch}" ]]; then
      echo "Bundled ${package} has architecture ${arch}; board architecture is ${board_arch}." >&2
      return 1
    fi
  done

  provides="$(dpkg-deb -f "${runtime_deb}" Provides 2>/dev/null || true)"
  provided_version="$(exact_dependency_version_from_relations simaai-memory-lib <<<"${provides}" || true)"
  if [[ "${provided_version}" != "${SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION}" ]]; then
    echo "Bundled simaai-memory-lib=${SIMAAI_MEMORY_ACTUAL_VERSION} must provide simaai-memory-lib (= ${SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION}); got ${provides:-<none>}." >&2
    return 1
  fi

  provides="$(dpkg-deb -f "${dev_deb}" Provides 2>/dev/null || true)"
  provided_version="$(exact_dependency_version_from_relations simaai-memory-lib-dev <<<"${provides}" || true)"
  if [[ "${provided_version}" != "${SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION}" ]]; then
    echo "Bundled simaai-memory-lib-dev=${SIMAAI_MEMORY_ACTUAL_VERSION} must provide simaai-memory-lib-dev (= ${SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION}); got ${provides:-<none>}." >&2
    return 1
  fi

  depends="$(dpkg-deb -f "${dev_deb}" Depends 2>/dev/null || true)"
  dev_runtime_version="$(exact_dependency_version_from_relations simaai-memory-lib <<<"${depends}" || true)"
  if [[ "${dev_runtime_version}" != "${SIMAAI_MEMORY_ACTUAL_VERSION}" ]]; then
    echo "Bundled simaai-memory-lib-dev must depend on simaai-memory-lib (= ${SIMAAI_MEMORY_ACTUAL_VERSION}); got ${depends:-<none>}." >&2
    return 1
  fi

  SIMAAI_MEMORY_RUNTIME_DEB="${runtime_deb}"
  SIMAAI_MEMORY_DEV_DEB="${dev_deb}"
  SIMAAI_MEMORY_DEBS=("${runtime_deb}" "${dev_deb}")
}

validate_local_simaai_memory_payload() {
  local extract_dir payload soname symbol build_id expected_deb_sha actual_deb_sha package_deb
  local -a payloads=()
  extract_dir="$(mktemp -d /tmp/sima-neat-memory-payload-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${extract_dir}")
  dpkg-deb -x "${SIMAAI_MEMORY_RUNTIME_DEB}" "${extract_dir}"
  mapfile -t payloads < <(find "${extract_dir}/usr/lib" -type f -name 'libsimaaimem.so.*' | sort)
  if [[ "${#payloads[@]}" -ne 1 ]]; then
    echo "Bundled simaai-memory-lib must contain exactly one versioned libsimaaimem payload; found ${#payloads[@]}." >&2
    return 1
  fi
  payload="${payloads[0]}"

  soname="$(LC_ALL=C readelf -d "${payload}" 2>/dev/null |
    sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p' | head -n1)"
  if [[ "${soname}" != "libsimaaimem.so.2" ]]; then
    echo "Bundled simaai-memory-lib has unexpected SONAME ${soname:-<missing>}; expected libsimaaimem.so.2." >&2
    return 1
  fi
  symbol="$(LC_ALL=C readelf -Ws "${payload}" 2>/dev/null |
    awk '$8 == "simaai_memory_export_dmabuf_fd" { print $8; exit }')"
  if [[ -z "${symbol}" ]]; then
    echo "Bundled simaai-memory-lib is missing simaai_memory_export_dmabuf_fd." >&2
    return 1
  fi
  build_id="$(LC_ALL=C readelf -n "${payload}" 2>/dev/null |
    sed -n 's/^[[:space:]]*Build ID: //p' | head -n1)"
  if [[ -z "${build_id}" ]]; then
    echo "Bundled simaai-memory-lib payload has no ELF build ID." >&2
    return 1
  fi

  for package_deb in "${SIMAAI_MEMORY_DEBS[@]}"; do
    expected_deb_sha="$(artifact_checksum_for_file "${package_deb}")" || return 1
    if [[ -n "${expected_deb_sha}" ]]; then
      actual_deb_sha="$(sha256sum "${package_deb}" | awk '{print $1}')"
      if [[ "${actual_deb_sha}" != "${expected_deb_sha}" ]]; then
        echo "Bundled $(basename "${package_deb}") checksum does not match package metadata." >&2
        return 1
      fi
    else
      log "No resources-checksum entry was supplied for $(basename "${package_deb}"); installed payload identity will still be verified against the bundled DEB."
    fi
  done

  SIMAAI_MEMORY_PAYLOAD_PATH="${payload#${extract_dir}}"
  SIMAAI_MEMORY_PAYLOAD_SHA256="$(sha256sum "${payload}" | awk '{print $1}')"
  SIMAAI_MEMORY_PAYLOAD_BUILD_ID="${build_id}"
  log "Validated bundled simaai-memory-lib=${SIMAAI_MEMORY_ACTUAL_VERSION} (provides ${SIMAAI_MEMORY_PLATFORM_COMPAT_VERSION}) payload sha256=${SIMAAI_MEMORY_PAYLOAD_SHA256} build-id=${SIMAAI_MEMORY_PAYLOAD_BUILD_ID}"
}

remove_local_simaai_memory_debs_from_general_transaction() {
  local deb memory_deb is_memory
  local -a remaining=()
  for deb in "${DEBS[@]}"; do
    is_memory=0
    for memory_deb in "${SIMAAI_MEMORY_DEBS[@]}"; do
      if [[ "${deb}" == "${memory_deb}" ]]; then
        is_memory=1
        break
      fi
    done
    [[ "${is_memory}" -eq 1 ]] || remaining+=("${deb}")
  done
  DEBS=("${remaining[@]}")
}

snapshot_memory_transaction_guard_state() {
  local ota_owner
  SIMAAI_MEMORY_PREINSTALL_PACKAGES="$(mktemp /tmp/sima-neat-memory-packages-before-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${SIMAAI_MEMORY_PREINSTALL_PACKAGES}")
  dpkg-query -W -f='${binary:Package}\t${db:Status-Abbrev}\n' 2>/dev/null |
    awk -F '\t' '$2 ~ /^ii / {print $1}' | sort -u >"${SIMAAI_MEMORY_PREINSTALL_PACKAGES}"

  SIMAAI_MEMORY_PREINSTALL_PALETTE_INSTALLED=0
  SIMAAI_MEMORY_PREINSTALL_PALETTE_VERSION=""
  SIMAAI_MEMORY_PREINSTALL_OTA_PATH=""
  if ! deb_package_is_installed simaai-palette-modalix; then
    log "simaai-palette-modalix is not installed; the isolated memory transaction has no pre-existing palette/OTA state to preserve."
    return 0
  fi
  SIMAAI_MEMORY_PREINSTALL_PALETTE_VERSION="$(deb_package_installed_version simaai-palette-modalix)"
  SIMAAI_MEMORY_PREINSTALL_OTA_PATH="$(command -v simaai-ota 2>/dev/null || true)"
  if [[ -z "${SIMAAI_MEMORY_PREINSTALL_OTA_PATH}" ]]; then
    echo "simaai-ota is missing before the memory replacement." >&2
    return 1
  fi
  ota_owner="$(dpkg-query -S "${SIMAAI_MEMORY_PREINSTALL_OTA_PATH}" 2>/dev/null || true)"
  if [[ ! "${ota_owner}" =~ ^simaai-palette-modalix(:[^:[:space:]]+)?:[[:space:]] ]]; then
    echo "${SIMAAI_MEMORY_PREINSTALL_OTA_PATH} is not owned by simaai-palette-modalix before the memory replacement: ${ota_owner:-<unowned>}" >&2
    return 1
  fi
  SIMAAI_MEMORY_PREINSTALL_PALETTE_INSTALLED=1
}

verify_memory_guard_palette_and_ota() {
  local current_palette ota_path ota_owner
  if [[ "${SIMAAI_MEMORY_PREINSTALL_PALETTE_INSTALLED:-0}" -ne 1 ]]; then
    return 0
  fi
  if ! deb_package_is_installed simaai-palette-modalix; then
    echo "simaai-palette-modalix was removed during NEAT installation." >&2
    return 1
  fi
  current_palette="$(deb_package_installed_version simaai-palette-modalix)"
  if [[ "${current_palette}" != "${SIMAAI_MEMORY_PREINSTALL_PALETTE_VERSION}" ]]; then
    echo "simaai-palette-modalix changed from ${SIMAAI_MEMORY_PREINSTALL_PALETTE_VERSION} to ${current_palette}." >&2
    return 1
  fi
  ota_path="$(command -v simaai-ota 2>/dev/null || true)"
  if [[ "${ota_path}" != "${SIMAAI_MEMORY_PREINSTALL_OTA_PATH}" ]]; then
    echo "simaai-ota path changed or disappeared during NEAT installation: ${ota_path:-<missing>}." >&2
    return 1
  fi
  ota_owner="$(dpkg-query -S "${ota_path}" 2>/dev/null || true)"
  if [[ ! "${ota_owner}" =~ ^simaai-palette-modalix(:[^:[:space:]]+)?:[[:space:]] ]]; then
    echo "simaai-ota is no longer owned by simaai-palette-modalix: ${ota_owner:-<unowned>}." >&2
    return 1
  fi
}

simaai_ota_command_path() {
  command -v simaai-ota 2>/dev/null || true
}

verify_canonical_palette_and_ota_installation() {
  local ota_path ota_owner
  if ! deb_package_is_installed simaai-palette-modalix; then
    echo "simaai-palette-modalix is not installed after the native Modalix transaction." >&2
    return 1
  fi
  ota_path="$(simaai_ota_command_path)"
  if [[ "${ota_path}" != "/usr/bin/simaai-ota" ]]; then
    echo "Canonical simaai-ota is missing after the native Modalix transaction: ${ota_path:-<missing>}." >&2
    return 1
  fi
  ota_owner="$(dpkg-query -S /usr/bin/simaai-ota 2>/dev/null || true)"
  if [[ ! "${ota_owner}" =~ ^simaai-palette-modalix(:[^:[:space:]]+)?:[[:space:]] ]]; then
    echo "/usr/bin/simaai-ota is not owned by simaai-palette-modalix: ${ota_owner:-<unowned>}." >&2
    return 1
  fi
}

verify_installed_simaai_memory_payload() {
  local installed_version installed_dev_version installed_sha installed_build_id owner
  if ! deb_package_is_installed simaai-memory-lib; then
    echo "simaai-memory-lib is not installed after the isolated replacement." >&2
    return 1
  fi
  installed_version="$(deb_package_installed_version simaai-memory-lib)"
  installed_dev_version="$(deb_package_installed_version simaai-memory-lib-dev 2>/dev/null || true)"
  if [[ "${installed_version}" != "${SIMAAI_MEMORY_ACTUAL_VERSION}" ||
        "${installed_dev_version}" != "${SIMAAI_MEMORY_ACTUAL_VERSION}" ]]; then
    echo "Installed memory runtime/dev versions do not match bundled ${SIMAAI_MEMORY_ACTUAL_VERSION}: runtime=${installed_version:-<missing>} dev=${installed_dev_version:-<missing>}." >&2
    return 1
  fi
  if [[ ! -f "${SIMAAI_MEMORY_PAYLOAD_PATH}" ]]; then
    echo "Installed simaai-memory-lib payload is missing: ${SIMAAI_MEMORY_PAYLOAD_PATH}" >&2
    return 1
  fi
  owner="$(dpkg-query -S "${SIMAAI_MEMORY_PAYLOAD_PATH}" 2>/dev/null || true)"
  if [[ ! "${owner}" =~ ^simaai-memory-lib(:[^:[:space:]]+)?:[[:space:]] ]]; then
    echo "Installed memory payload is not owned by simaai-memory-lib: ${owner:-<unowned>}." >&2
    return 1
  fi
  installed_sha="$(sha256sum "${SIMAAI_MEMORY_PAYLOAD_PATH}" | awk '{print $1}')"
  installed_build_id="$(LC_ALL=C readelf -n "${SIMAAI_MEMORY_PAYLOAD_PATH}" 2>/dev/null |
    sed -n 's/^[[:space:]]*Build ID: //p' | head -n1)"
  if [[ "${installed_sha}" != "${SIMAAI_MEMORY_PAYLOAD_SHA256}" ||
        "${installed_build_id}" != "${SIMAAI_MEMORY_PAYLOAD_BUILD_ID}" ]]; then
    echo "Installed simaai-memory-lib payload does not match the bundled artifact." >&2
    echo "  expected sha/build-id: ${SIMAAI_MEMORY_PAYLOAD_SHA256} / ${SIMAAI_MEMORY_PAYLOAD_BUILD_ID}" >&2
    echo "  installed sha/build-id: ${installed_sha:-<missing>} / ${installed_build_id:-<missing>}" >&2
    return 1
  fi
  log "Verified installed simaai-memory-lib payload matches the bundled artifact."
}

verify_memory_transaction_preservation() {
  local after missing audit_log
  after="$(mktemp /tmp/sima-neat-memory-packages-after-XXXXXX)"
  audit_log="$(mktemp /tmp/sima-neat-memory-dpkg-audit-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${after}" "${audit_log}")
  dpkg-query -W -f='${binary:Package}\t${db:Status-Abbrev}\n' 2>/dev/null |
    awk -F '\t' '$2 ~ /^ii / {print $1}' | sort -u >"${after}"
  missing="$(comm -23 "${SIMAAI_MEMORY_PREINSTALL_PACKAGES}" "${after}")"
  if [[ -n "${missing}" ]]; then
    echo "The isolated simaai-memory replacement removed preinstalled packages:" >&2
    while IFS= read -r package; do
      [[ -n "${package}" ]] && printf '  %s\n' "${package}" >&2
    done <<<"${missing}"
    return 1
  fi
  verify_memory_guard_palette_and_ota || return 1
  if ! run_sudo apt-get check; then
    echo "APT dependency check failed after the isolated simaai-memory replacement." >&2
    return 1
  fi
  if ! dpkg --audit >"${audit_log}" 2>&1 || [[ -s "${audit_log}" ]]; then
    echo "dpkg audit failed after the isolated simaai-memory replacement:" >&2
    cat "${audit_log}" >&2
    return 1
  fi
  if [[ "${SIMAAI_MEMORY_PREINSTALL_PALETTE_INSTALLED:-0}" -eq 1 ]]; then
    log "Verified the isolated simaai-memory replacement removed no preinstalled packages and preserved simaai-palette-modalix/simaai-ota."
  else
    log "Verified the isolated simaai-memory replacement removed no preinstalled packages; no pre-existing palette/OTA state required preservation."
  fi
}

install_local_simaai_memory_transaction() {
  local simulation_log
  local -a apt_args=(apt-get install -y --reinstall --no-remove)

  collect_local_simaai_memory_debs || return 1
  validate_local_simaai_memory_payload || return 1
  snapshot_memory_transaction_guard_state || return 1
  simulation_log="$(mktemp /tmp/sima-neat-memory-apt-simulation-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${simulation_log}")

  log "Simulating isolated local simaai-memory replacement with package removal disabled."
  if ! run_sudo "${apt_args[@]}" --simulate "${SIMAAI_MEMORY_DEBS[@]}" >"${simulation_log}" 2>&1; then
    cat "${simulation_log}" >&2
    echo "APT rejected the isolated local simaai-memory transaction; no packages were changed." >&2
    return 1
  fi
  cat "${simulation_log}"
  if grep -q '^Remv[[:space:]]' "${simulation_log}"; then
    echo "APT simulation planned package removal for the isolated simaai-memory transaction; refusing to continue." >&2
    grep '^Remv[[:space:]]' "${simulation_log}" >&2
    return 1
  fi

  log "Installing bundled simaai-memory runtime/dev packages in an isolated zero-removal transaction."
  run_sudo "${apt_args[@]}" "${SIMAAI_MEMORY_DEBS[@]}" || return 1
  verify_installed_simaai_memory_payload || return 1
  verify_memory_transaction_preservation || return 1
  SIMAAI_MEMORY_TRANSACTION_COMPLETE=1
  remove_local_simaai_memory_debs_from_general_transaction
}

local_deb_for_exact_package() {
  local package="$1"
  local version="$2"
  local deb deb_package deb_version
  for deb in "${DEBS[@]:-}"; do
    [[ -f "${deb}" ]] || continue
    deb_package="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    [[ "${deb_package}" == "${package}" ]] || continue
    deb_version="$(dpkg-deb -f "${deb}" Version 2>/dev/null || true)"
    if [[ "${deb_version}" == "${version}" ]]; then
      printf '%s\n' "${deb}"
      return 0
    fi
  done
  return 1
}

exact_package_install_spec() {
  local package="$1"
  local version="$2"
  local local_deb apt_control
  local_deb="$(local_deb_for_exact_package "${package}" "${version}" || true)"
  if [[ -n "${local_deb}" ]]; then
    printf '%s\n' "${local_deb}"
    return 0
  fi

  apt_control="$(LC_ALL=C apt-cache show "${package}=${version}" 2>/dev/null)" || true
  if [[ -n "${apt_control}" ]]; then
    printf '%s=%s\n' "${package}" "${version}"
    return 0
  fi

  echo "Required canonical Modalix package is unavailable locally and from apt: ${package}=${version}" >&2
  return 1
}

native_modalix_repair_is_required() {
  local package version
  for package in simaai-gst-plugins simaai-palette-modalix; do
    if ! deb_package_is_installed "${package}"; then
      return 0
    fi
  done

  # Old CI runs briefly published private same-name packages. They cannot
  # satisfy palette dependencies such as libcamera (= 2.1.1), even when they
  # self-Provide that version. Repair them before installing the local stack.
  for package in \
    libcamera libcamera-dev libcamera-tools \
    simaai-memory-lib simaai-memory-lib-dev; do
    version="$(deb_package_installed_version "${package}" || true)"
    if [[ "${version}" == *+neat* ]]; then
      return 0
    fi
  done
  return 1
}

native_modalix_restore_specs() {
  local out_array_name="$1"
  local -n out_array="${out_array_name}"
  local palette_version package version spec

  palette_version="$(apt_candidate_version simaai-palette-modalix)"
  if [[ -z "${palette_version}" || "${palette_version}" == "(none)" ]]; then
    echo "Required native Modalix package has no apt candidate: simaai-palette-modalix" >&2
    return 1
  fi

  out_array=()
  for package in libcamera libcamera-tools simaai-memory-lib; do
    # The bundled memory runtime/dev pair is installed and verified in its own
    # zero-removal transaction. Never let this broader native-repair resolver
    # substitute the repository's indistinguishable same-version payload.
    if [[ "${package}" == "simaai-memory-lib" &&
          "${SIMAAI_MEMORY_TRANSACTION_COMPLETE:-0}" -eq 1 ]]; then
      continue
    fi
    version="$(apt_exact_dependency_version \
      simaai-palette-modalix "${palette_version}" "${package}")" || {
      echo "simaai-palette-modalix=${palette_version} has no exact dependency on ${package}" >&2
      return 1
    }
    spec="$(exact_package_install_spec "${package}" "${version}")" || return 1
    out_array+=("${spec}")

    case "${package}" in
      libcamera)
        if deb_package_is_present libcamera-dev; then
          spec="$(exact_package_install_spec libcamera-dev "${version}")" || return 1
          out_array+=("${spec}")
        fi
        ;;
      simaai-memory-lib)
        if deb_package_is_present simaai-memory-lib-dev; then
          spec="$(exact_package_install_spec simaai-memory-lib-dev "${version}")" || return 1
          out_array+=("${spec}")
        fi
        ;;
    esac
  done

  out_array+=(simaai-gst-plugins "simaai-palette-modalix=${palette_version}")
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

  log "Removing installed NEAT packages before retrying apt downgrade/repair:"
  printf '  %s\n' "${packages[@]}"
  if run_sudo apt-get remove -y "${packages[@]}"; then
    return 0
  fi

  log "apt-get remove failed; falling back to forced dpkg removal before apt repair."
  run_sudo dpkg --remove --force-depends "${packages[@]}"
}

local_deb_version_for_package() {
  local package="$1"
  local deb pkg version
  for deb in "${DEBS[@]}"; do
    [[ -f "${deb}" ]] || continue
    pkg="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    [[ "${pkg}" == "${package}" ]] || continue
    version="$(dpkg-deb -f "${deb}" Version 2>/dev/null || true)"
    if [[ -n "${version}" ]]; then
      printf '%s\n' "${version}"
      return 0
    fi
  done
  return 1
}

maybe_relax_sima_lmm_dep() {
  local deb="$1"
  local out_array_name="$2"
  local -n out_array="${out_array_name}"

  if [[ "${NEAT_INSTALLER_RELAX_SIMA_LMM_DEP}" != "ON" ]]; then
    out_array+=("${deb}")
    return 0
  fi
  case "$(basename "${deb}")" in
    sima-neat-*-Linux-core.deb | sima-neat-*-Linux-dev.deb) ;;
    *)
      out_array+=("${deb}")
      return 0
      ;;
  esac
  if ! command -v dpkg-deb >/dev/null 2>&1 || ! command -v dpkg >/dev/null 2>&1 ||
      ! command -v python3 >/dev/null 2>&1; then
    out_array+=("${deb}")
    return 0
  fi

  local local_lmm_core_version local_lmm_dev_version
  local_lmm_core_version="$(local_deb_version_for_package sima-lmm-core || true)"
  local_lmm_dev_version="$(local_deb_version_for_package sima-lmm-dev || true)"
  if [[ -z "${local_lmm_core_version}" && -z "${local_lmm_dev_version}" ]]; then
    out_array+=("${deb}")
    return 0
  fi

  local tmp_dir unpack_dir out_deb changed_marker
  tmp_dir="$(mktemp -d /tmp/sima-neat-deb-normalize-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${tmp_dir}")
  unpack_dir="${tmp_dir}/unpack"
  out_deb="${tmp_dir}/$(basename "${deb}")"
  changed_marker="${tmp_dir}/changed"

  dpkg-deb -R "${deb}" "${unpack_dir}"
  if python3 - "${unpack_dir}/DEBIAN/control" "${local_lmm_core_version}" \
      "${local_lmm_dev_version}" "${changed_marker}" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

control = Path(sys.argv[1])
local_versions = {
    "sima-lmm-core": sys.argv[2],
    "sima-lmm-dev": sys.argv[3],
}
changed_marker = Path(sys.argv[4])
text = control.read_text()

def minor_family(version: str):
    match = re.match(r"(?:(\d+):)?(\d+)\.(\d+)(?:\.\d+)?(?:[.+~-].*)?$", version)
    if not match:
        return None
    return match.groups()

def relax_package(package: str, body: str) -> str:
    local = local_versions.get(package, "").strip()
    if not local:
        return body

    def repl(match: re.Match[str]) -> str:
        required = match.group(1).strip()
        if minor_family(local) != minor_family(required):
            return match.group(0)
        if subprocess.run(
            ["dpkg", "--compare-versions", local, "ge", required],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0:
            return match.group(0)
        changed_marker.write_text("1")
        return f"{package} (>= {local})"

    return re.sub(rf"{re.escape(package)}\s*\(>=\s*([^)]+)\)", repl, body)

new_text = relax_package("sima-lmm-core", text)
new_text = relax_package("sima-lmm-dev", new_text)
if new_text != text:
    control.write_text(new_text)
PY
  then
    if [[ -f "${changed_marker}" ]]; then
      log "Relaxed $(basename "${deb}") dependency on bundled sima-lmm package version(s): core=${local_lmm_core_version:-<none>} dev=${local_lmm_dev_version:-<none>}"
      dpkg-deb -b "${unpack_dir}" "${out_deb}" >/dev/null
      out_array+=("${out_deb}")
      return 0
    fi
  fi

  out_array+=("${deb}")
}

prepare_debs_for_board_install() {
  local -a prepared=()
  local deb
  for deb in "${DEBS[@]}"; do
    maybe_relax_sima_lmm_dep "${deb}" prepared
  done
  DEBS=("${prepared[@]}")
}

stop_board_runtime_before_install() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  log "Stopping NEAT runtime services before package replacement."
  local svc
  for svc in \
      simaai-pipeline-manager.service \
      simaai-appcomplex.service \
      rctd.service \
      encoder.service \
      decoder.service \
      simaai-log.service; do
    if systemctl cat "${svc}" >/dev/null 2>&1; then
      run_sudo systemctl stop "${svc}" >/dev/null 2>&1 || true
      run_sudo systemctl reset-failed "${svc}" >/dev/null 2>&1 || true
    fi
  done

  if [[ -x /usr/libexec/simaai-appcomplex/clean-stale-mlashmcomplex ]]; then
    run_sudo /usr/libexec/simaai-appcomplex/clean-stale-mlashmcomplex || true
  else
    run_sudo pkill -TERM -x mlashmcomplex >/dev/null 2>&1 || true
    sleep 0.5
    run_sudo pkill -KILL -x mlashmcomplex >/dev/null 2>&1 || true
  fi

  run_sudo rm -f /tmp/mlactrl /dev/shm/mlashmdata
}

activate_board_runtime_after_install() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  # These files are recreated by simaai-appcomplex.service.  Remove stale IPC
  # before the post-install MLA init/reset path so clients cannot observe an
  # old dispatcher lifetime after package replacement.
  run_sudo rm -f /tmp/mlactrl /dev/shm/mlashmdata
  # Package configuration intentionally does not restart services.  Reload
  # systemd here so the owned maintenance window starts services from the unit
  # files that were just unpacked.
  run_sudo systemctl daemon-reload || true

  if [[ "${NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD}" == "ON" &&
        -x /usr/libexec/sima-neat-firmware/install.sh ]]; then
    log "Activating staged EV74 firmware and resetting runtime state."
    run_sudo /usr/libexec/sima-neat-firmware/install.sh --activate
  else
    log "EV74 firmware activation skipped; starting simaai-appcomplex.service directly."
    if systemctl cat simaai-appcomplex.service >/dev/null 2>&1; then
      run_sudo systemctl restart simaai-appcomplex.service || true
    fi
  fi
}

verify_board_runtime_services() {
  local service="simaai-appcomplex.service"

  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  if ! systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
    return 0
  fi

  # The Debian maintainer script is intentionally generated through debhelper,
  # and deb-systemd-invoke treats service start failures as non-fatal so package
  # transactions can still complete.  For this installer the runtime is not
  # usable without the MLA shared-memory dispatcher, so make readiness explicit:
  # try one start/restart if the unit is inactive, then fail with the unit status
  # instead of leaving users with later "Connecting to server failed" errors.
  if ! systemctl is-active --quiet "${service}"; then
    log "${service} is not active after package install; attempting to start it once."
    run_sudo systemctl start "${service}" || true
    sleep 1
  fi

  if ! systemctl is-active --quiet "${service}"; then
    echo "${service} is not active after NEAT package installation." >&2
    run_sudo systemctl --no-pager --full status "${service}" >&2 || true
    run_sudo journalctl -u "${service}" --no-pager -n 80 >&2 || true
    run_sudo bash -c 'for f in /sys/class/remoteproc/remoteproc*/name /sys/class/remoteproc/remoteproc*/state; do [ -e "$f" ] && printf "%s: " "$f" && cat "$f"; done' >&2 || true
    exit 1
  fi

  log "Verified ${service} is active."
}


restart_board_codec_services() {
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
  if [[ ! "${soname}" =~ ^libneatdispatchercore\.so\.[1-9][0-9]*$ ]]; then
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
  printf '%s\n' "/usr/lib"
}

find_packaged_sima_neat_versioned_lib() {
  local lib_dir package_files candidate basename
  local -a matches=()
  lib_dir="$(sima_neat_global_lib_dir)"
  package_files="$(dpkg-query -L sima-neat 2>/dev/null)" || return 1

  while IFS= read -r candidate; do
    [[ "$(dirname "${candidate}")" == "${lib_dir}" ]] || continue
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
  lib_dir="$(sima_neat_global_lib_dir)"
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
  lib_dir="$(sima_neat_global_lib_dir)"

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
  migrate_stale_global_dispatcher_libs
  verify_private_dispatcher_runtime
  repair_global_sima_neat_lib_links
  verify_global_sima_neat_lib_links
  verify_installed_simaai_memory_payload
  verify_memory_guard_palette_and_ota
  verify_canonical_palette_and_ota_installation
  activate_board_runtime_after_install
  restart_board_codec_services
  verify_board_codec_services
  verify_board_runtime_services
}

install_debs_on_board() {
  prepare_debs_for_board_install
  log "Detected Modalix board environment; installing DEBs with apt."
  printf '[install_neat_framework] DEB install set:\n'
  printf '  %s\n' "${DEBS[@]}"
  refresh_apt_metadata_for_board_install
  stop_board_runtime_before_install

  # The memory replacement is deliberately isolated from the broader Neat
  # transaction. It must start from a coherent APT state because --fix-broken
  # could make an otherwise local reinstall remove unrelated platform packages.
  if ! apt_package_database_is_healthy; then
    echo "APT package state is unhealthy; refusing the isolated zero-removal simaai-memory replacement." >&2
    echo "Repair the board package database first, then rerun this installer." >&2
    exit 1
  fi
  if ! install_local_simaai_memory_transaction; then
    echo "Failed to install the bundled simaai-memory payload without package removals." >&2
    exit 1
  fi

  # Prefer apt-get for the remaining packages so normal system dependencies can
  # be resolved. The memory DEBs have been removed from this set; the repository
  # cannot silently substitute its indistinguishable same-version payload.

  local -a board_install_specs=()
  local -A seen_install_specs=()
  local spec
  for spec in "${DEBS[@]}"; do
    [[ -n "${spec}" ]] || continue
    if [[ -n "${seen_install_specs[${spec}]+x}" ]]; then
      continue
    fi
    seen_install_specs["${spec}"]=1
    board_install_specs+=("${spec}")
  done

  local -a apt_install_args=(
    apt-get install -y --fix-broken --allow-downgrades --reinstall
    -o Dpkg::Options::=--force-overwrite
  )

  local simulation_log
  simulation_log="$(mktemp /tmp/sima-neat-apt-simulate-XXXXXX)"
  INSTALLER_TMP_DIRS+=("${simulation_log}")
  if ! run_sudo "${apt_install_args[@]}" --simulate \
      "${board_install_specs[@]}" >"${simulation_log}" 2>&1; then
    cat "${simulation_log}" >&2
    echo "APT cannot satisfy the bundled Neat package transaction." >&2
    exit 1
  fi
  if ! verify_simulated_package_removals \
      "${simulation_log}" "${board_install_specs[@]}"; then
    exit 1
  fi

  if run_sudo "${apt_install_args[@]}" "${board_install_specs[@]}"; then
    run_sudo apt-get check
    complete_board_install_after_packages
    return 0
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}" != "ON" ]]; then
    echo "apt-get rejected the local package set; refusing to remove the installed runtime because NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}." >&2
    echo "Fix the bundled package versions/dependencies, or explicitly set NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON during a recoverable maintenance operation." >&2
    exit 1
  fi

  log "apt-get install failed; NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON, removing installed NEAT packages represented by the local DEB set and retrying apt."
  remove_installed_local_deb_packages
  if run_sudo "${apt_install_args[@]}" "${board_install_specs[@]}"; then
    complete_board_install_after_packages
    return 0
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}" != "ON" ]]; then
    echo "apt-get install failed and NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}; refusing direct dpkg fallback." >&2
    exit 1
  else
    log "apt-get install failed; retrying with direct dpkg install of the local NEAT DEB set."
  fi

  run_sudo dpkg -i --force-overwrite "${DEBS[@]}"
  run_sudo apt-get check
  complete_board_install_after_packages
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
    echo "SDK sysroot install requires exactly one sima-neat and one sima-neat-dev package." >&2
    echo "  sima-neat packages:     ${#core_debs[@]}" >&2
    echo "  sima-neat-dev packages: ${#dev_debs[@]}" >&2
    echo "Remove stale package versions or install from a generated metadata/manifest bundle." >&2
    return 1
  fi
  if [[ -z "${core_versions[0]}" || -z "${dev_versions[0]}" ||
        "${core_versions[0]}" != "${dev_versions[0]}" ]]; then
    echo "SDK sysroot sima-neat package versions do not match." >&2
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

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

parse_args "$@"

collect_debs_in_install_order "." DEBS
if [[ "${#DEBS[@]}" -lt 1 ]]; then
  echo "No required DEB files found in current directory." >&2
  exit 1
fi

ENV_MODE="$(detect_env_mode)"
log_green "Environment mode: ${ENV_MODE}"
ensure_platform_compatible
if [[ "${ENV_MODE}" == "elxr-sdk" ]]; then
  install_debs_into_sysroot
  ensure_sdk_neat_cli_symlink
  install_agent_skills_for_current_user "${SYSROOT:-/opt/toolchain/aarch64/modalix}/usr/share/sima-neat/skills/sima-neat"
  deploy_artifacts_to_paired_devkit_if_configured
else
  VENV_DIR="$(resolve_venv_dir)"
  ACTIVATE_PATH="$(activation_path_for_display "${VENV_DIR}")"
  remove_stale_global_sima_lmm_pip_install
  log_green "Preparing Python virtual environment at ${VENV_DIR}"
  mkdir -p "$(dirname "${VENV_DIR}")"
  python3 -m venv --system-site-packages "${VENV_DIR}"
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
  install_debs_on_board
  install_agent_skills_for_current_user "/usr/share/sima-neat/skills/sima-neat"
fi

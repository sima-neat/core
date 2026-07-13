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
# - NEAT_INSTALLER_RELAX_SIMAAI_MEMORY_DEP: ON/OFF (default: ON) relax selected
#   exact simaai-memory-lib dependencies to the matching minor-version family
#   when the board carries the SDK's git-suffixed compatible memory package.
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
NEAT_INSTALLER_RELAX_SIMAAI_MEMORY_DEP="${NEAT_INSTALLER_RELAX_SIMAAI_MEMORY_DEP:-ON}"
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

cleanup_installer_tmp_dirs() {
  local dir
  for dir in "${INSTALLER_TMP_DIRS[@]}"; do
    [[ -n "${dir}" && -d "${dir}" ]] && rm -rf "${dir}"
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

maybe_relax_simaai_memory_dep() {
  local deb="$1"
  local out_array_name="$2"
  local -n out_array="${out_array_name}"

  if [[ "${NEAT_INSTALLER_RELAX_SIMAAI_MEMORY_DEP}" != "ON" ]]; then
    out_array+=("${deb}")
    return 0
  fi
  case "$(basename "${deb}")" in
    sima-lmm-*-Linux-core.deb | neat-appcomplex_*.deb | appcomplex_*.deb) ;;
    *)
      out_array+=("${deb}")
      return 0
      ;;
  esac
  if ! command -v dpkg-deb >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    out_array+=("${deb}")
    return 0
  fi

  local installed_memory_version
  installed_memory_version="$(dpkg-query -W -f='${Version}' simaai-memory-lib 2>/dev/null || true)"
  if [[ -z "${installed_memory_version}" ]]; then
    installed_memory_version="$(local_deb_version_for_package simaai-memory-lib || true)"
  fi
  if [[ -z "${installed_memory_version}" ]]; then
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
  if python3 - "${unpack_dir}/DEBIAN/control" "${installed_memory_version}" "${changed_marker}" <<'PY'
import re
import sys
from pathlib import Path

control = Path(sys.argv[1])
installed = sys.argv[2]
changed_marker = Path(sys.argv[3])
text = control.read_text()

# Some SDK/DevKit images ship simaai-memory-lib as a same-upstream local build
# such as 2.0.0~git..., 2.1.1+neat1, or 2.1.1-1 while selected runtime DEBs can
# depend on the unsuffixed exact version 2.0.0/2.1.1. Debian exact-version
# dependencies make otherwise compatible boards uninstallable. Only relax this
# dependency when the installed version is the exact required version or the same
# upstream version plus a Debian suffix, and cap the relaxed range to the same
# minor-version family.
def same_upstream_family(installed: str, required: str) -> bool:
    if installed == required:
        return True
    return any(installed.startswith(required + suffix) for suffix in ("~", "+", "-"))

def next_minor_bound(version: str):
    match = re.match(r"(?:(\d+):)?(\d+)\.(\d+)(?:[.+~-].*)?$", version)
    if not match:
        match = re.match(r"(?:(\d+):)?(\d+)\.(\d+)\.\d+(?:[.+~-].*)?$", version)
    if not match:
        return None
    epoch, major, minor = match.groups()
    prefix = f"{epoch}:" if epoch else ""
    return f"{prefix}{major}.{int(minor) + 1}~"

def repl(match: re.Match[str]) -> str:
    required = match.group(1).strip()
    if same_upstream_family(installed, required):
        upper = next_minor_bound(required)
        if upper is None:
            return match.group(0)
        changed_marker.write_text("1")
        return f"simaai-memory-lib (>= {required}~), simaai-memory-lib (<< {upper})"
    return match.group(0)

new_text = re.sub(r"simaai-memory-lib\s*\(=\s*([^)]+)\)", repl, text)
if new_text != text:
    control.write_text(new_text)
PY
  then
    if [[ -f "${changed_marker}" ]]; then
      log "Relaxed $(basename "${deb}") dependency on simaai-memory-lib for installed version ${installed_memory_version}"
      dpkg-deb -b "${unpack_dir}" "${out_deb}" >/dev/null
      out_array+=("${out_deb}")
      return 0
    fi
  fi

  out_array+=("${deb}")
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
  local -a memory_prepared=()
  local deb
  for deb in "${DEBS[@]}"; do
    maybe_relax_simaai_memory_dep "${deb}" memory_prepared
  done
  DEBS=("${memory_prepared[@]}")
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

repair_stale_global_dispatcher_lib() {
  local global_lib="/usr/lib/aarch64-linux-gnu/libneatdispatchercore.so"
  local runtime_lib="/usr/lib/aarch64-linux-gnu/neat/runtime/libneatdispatchercore.so"

  if [[ ! -e "${runtime_lib}" ]]; then
    log "Dispatcher lib repair skipped; runtime lib not found at ${runtime_lib}"
    return 0
  fi

  if [[ -L "${global_lib}" && "$(readlink -f "${global_lib}")" == "${runtime_lib}" ]]; then
    log "Dispatcher lib repair skipped; ${global_lib} already points at the packaged runtime lib."
    return 0
  fi

  if [[ -e "${global_lib}" ]]; then
    if dpkg-query -S "${global_lib}" >/dev/null 2>&1; then
      log "Dispatcher lib repair skipped; ${global_lib} is package-owned."
      return 0
    fi

    local backup="${global_lib}.bak-neat-installer-$(date -u +%Y%m%dT%H%M%SZ)"
    log "Quarantining stale unowned dispatcher lib ${global_lib} -> ${backup}"
    run_sudo mv -f "${global_lib}" "${backup}"
  fi

  if [[ ! -e "${global_lib}" ]]; then
    log "Linking ${global_lib} to packaged runtime lib ${runtime_lib}"
    run_sudo ln -s "${runtime_lib}" "${global_lib}"
  fi
}

find_packaged_sima_neat_versioned_lib() {
  local candidate
  local selected=""
  while IFS= read -r candidate; do
    case "${candidate}" in
      /usr/lib/libsima_neat.so.2.*)
        [[ "${candidate}" != *".bak"* && -e "${candidate}" ]] || continue
        selected="${candidate}"
        ;;
    esac
  done < <(dpkg-query -L sima-neat 2>/dev/null | sort -V)
  if [[ -n "${selected}" ]]; then
    printf '%s\n' "${selected}"
    return 0
  fi

  find /usr/lib -maxdepth 1 -type f -name 'libsima_neat.so.2.*' ! -name '*.bak*' 2>/dev/null \
    | sort -V \
    | tail -n 1
}

repair_global_sima_neat_lib_links() {
  local versioned_lib
  versioned_lib="$(find_packaged_sima_neat_versioned_lib || true)"
  if [[ -z "${versioned_lib}" || ! -e "${versioned_lib}" ]]; then
    log "libsima_neat link repair skipped; packaged versioned library not found."
    return 0
  fi

  local soname_link="/usr/lib/libsima_neat.so.2"
  local devel_link="/usr/lib/libsima_neat.so"
  local soname_target
  soname_target="$(basename "${versioned_lib}")"

  if [[ ! -L "${soname_link}" || "$(readlink "${soname_link}")" != "${soname_target}" ]]; then
    log "Repairing ${soname_link} -> ${soname_target}"
    run_sudo ln -sfn "${soname_target}" "${soname_link}"
  fi

  if [[ ! -L "${devel_link}" || "$(readlink "${devel_link}")" != "libsima_neat.so.2" ]]; then
    log "Repairing ${devel_link} -> libsima_neat.so.2"
    run_sudo ln -sfn "libsima_neat.so.2" "${devel_link}"
  fi
}

verify_global_sima_neat_lib_links() {
  local versioned_lib
  versioned_lib="$(find_packaged_sima_neat_versioned_lib || true)"
  if [[ -z "${versioned_lib}" || ! -e "${versioned_lib}" ]]; then
    echo "Could not find packaged /usr/lib/libsima_neat.so.2.* after install." >&2
    exit 1
  fi

  local resolved
  resolved="$(readlink -f /usr/lib/libsima_neat.so.2 2>/dev/null || true)"
  if [[ -z "${resolved}" || "${resolved}" != "${versioned_lib}" || "${resolved}" == *".bak"* ]]; then
    echo "/usr/lib/libsima_neat.so.2 does not resolve to the packaged library." >&2
    echo "  expected: ${versioned_lib}" >&2
    echo "  actual:   ${resolved:-<missing>}" >&2
    exit 1
  fi

  local devel_resolved
  devel_resolved="$(readlink -f /usr/lib/libsima_neat.so 2>/dev/null || true)"
  if [[ "${devel_resolved}" != "${versioned_lib}" ]]; then
    echo "/usr/lib/libsima_neat.so does not resolve to the packaged library." >&2
    echo "  expected: ${versioned_lib}" >&2
    echo "  actual:   ${devel_resolved:-<missing>}" >&2
    exit 1
  fi

  log "Verified libsima_neat links resolve to ${versioned_lib}"
}

install_debs_on_board() {
  prepare_debs_for_board_install
  log "Detected Modalix board environment; installing DEBs with apt."
  printf '[install_neat_framework] DEB install set:\n'
  printf '  %s\n' "${DEBS[@]}"
  refresh_apt_metadata_for_board_install
  stop_board_runtime_before_install

  # Prefer apt-get for normal installs so system dependencies can be resolved.
  # Some CI/self-hosted DevKit runners can be left in a transiently broken
  # exact-version state after a previous partial NEAT install, e.g.
  # neat-gst-plugins(main) depending on neat-runtime(main) while
  # neat-runtime(beta) is already unpacked.  Still try apt first even when
  # `apt-get check` is already unhappy: passing the local DEB set gives apt the
  # replacement packages it needs to repair the transaction and fetch normal
  # Debian dependencies.  Direct dpkg is only a last resort because it cannot
  # fetch missing dependencies and can leave CI boards half-configured.
  if ! apt_package_database_is_healthy; then
    log "apt package database has unresolved dependencies; attempting apt repair with the local NEAT DEB set."
  fi

  # Restore native palette packages before the local transaction. Otherwise,
  # apt may "repair" a board left by an older partial install by removing
  # simaai-palette-modalix instead of restoring its exact simaai-gst-plugins
  # dependency. This repair remains --no-remove; the following NEAT install is
  # allowed to perform its declared Conflicts/Replaces handoff.
  local -a preserved_native_packages=(simaai-gst-plugins simaai-palette-modalix)
  local repair_native_packages=0
  local package
  for package in "${preserved_native_packages[@]}"; do
    if ! deb_package_is_installed "${package}"; then
      repair_native_packages=1
    fi
  done
  if [[ "${repair_native_packages}" -eq 1 ]]; then
    for package in "${preserved_native_packages[@]}"; do
      if ! apt-cache show "${package}" >/dev/null 2>&1; then
        echo "Required native Modalix package is unavailable from apt: ${package}" >&2
        exit 1
      fi
    done
    log "Restoring native Modalix packages before the NEAT transaction: ${preserved_native_packages[*]}"
    if ! run_sudo apt-get install -y --fix-broken --no-remove "${preserved_native_packages[@]}"; then
      echo "Failed to restore native Modalix packages before the NEAT install." >&2
      exit 1
    fi
  fi

  local -a apt_install_args=(
    apt-get install -y --fix-broken --allow-downgrades --reinstall
    -o Dpkg::Options::=--force-overwrite
  )
  if run_sudo "${apt_install_args[@]}" "${DEBS[@]}"; then
    repair_stale_global_dispatcher_lib
    repair_global_sima_neat_lib_links
    verify_global_sima_neat_lib_links
    activate_board_runtime_after_install
    restart_board_codec_services
    verify_board_codec_services
    verify_board_runtime_services
    return 0
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}" != "ON" ]]; then
    echo "apt-get rejected the local package set; refusing to remove the installed runtime because NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=${NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL}." >&2
    echo "Fix the bundled package versions/dependencies, or explicitly set NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON during a recoverable maintenance operation." >&2
    exit 1
  fi

  log "apt-get install failed; NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON, removing installed NEAT packages represented by the local DEB set and retrying apt."
  remove_installed_local_deb_packages
  if run_sudo "${apt_install_args[@]}" "${DEBS[@]}"; then
    repair_stale_global_dispatcher_lib
    repair_global_sima_neat_lib_links
    verify_global_sima_neat_lib_links
    activate_board_runtime_after_install
    restart_board_codec_services
    verify_board_codec_services
    verify_board_runtime_services
    return 0
  fi

  if [[ "${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}" != "ON" ]]; then
    echo "apt-get install failed and NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=${NEAT_INSTALLER_ALLOW_DPKG_FALLBACK}; refusing direct dpkg fallback." >&2
    exit 1
  else
    log "apt-get install failed; retrying with direct dpkg install of the local NEAT DEB set."
  fi

  run_sudo dpkg -i --force-overwrite "${DEBS[@]}"
  repair_stale_global_dispatcher_lib
  repair_global_sima_neat_lib_links
  verify_global_sima_neat_lib_links
  activate_board_runtime_after_install
  restart_board_codec_services
  verify_board_codec_services
  verify_board_runtime_services
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
  ensure_sima_lmm_sysroot_deps "${sysroot}"
  local deb
  for deb in "${DEBS[@]}"; do
    log "Extracting $(basename "${deb}") into ${sysroot}"
    run_sudo dpkg-deb -x "${deb}" "${sysroot}"
  done

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

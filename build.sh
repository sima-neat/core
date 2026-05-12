#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BUILD_DIR=build
BUILD_TYPE=Release
OS_NAME="$(uname -s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIGINAL_SOURCE_DIR="${SIMANEAT_ORIGINAL_SOURCE_DIR:-${SCRIPT_DIR}}"
ORIGINAL_WORKSPACE_ROOT="${SIMANEAT_ORIGINAL_WORKSPACE_ROOT:-${WORKSPACE_ROOT}}"
SHADOW_BUILD_MODE="${SIMANEAT_SHADOW_BUILD:-auto}"
SHADOW_BUILD_ACTIVE="${SIMANEAT_SHADOW_BUILD_ACTIVE:-OFF}"
SOURCE_FS_TYPE=""

# Defaults
BUILD_SAMPLES=OFF
BUILD_TESTS=OFF
BUILD_DOCS=OFF
BUILD_ALL=OFF
DO_CLEAN=OFF
DOCS_ONLY=OFF
INSTALL_NODE=ON
SKIP_DOCS=OFF
INSTALL_DEPS_ONLY=OFF
EXAMPLES_ONLY=OFF
SKIP_DIST=OFF
BUILD_PYTHON=OFF
INSTALL_NEAT_INTERNALS=OFF
STRICT_WARNINGS="${SIMANEAT_STRICT_WARNINGS:-OFF}"
NEAT_INTERNALS_MANIFEST="${NEAT_INTERNALS_MANIFEST:-deps/manifest.json}"
NEAT_INTERNALS_BASE_URL="${NEAT_INTERNALS_BASE_URL:-https://neat-artifacts.modalix.info/neat-internals}"
NEAT_INTERNALS_DIR="${NEAT_INTERNALS_DIR:-neat-internals}"
NEAT_INTERNALS_PLUGIN_DIR="${NEAT_INTERNALS_DIR}/gst-plugins"
NEAT_INTERNALS_BASIC_AUTH="${NEAT_INTERNALS_BASIC_AUTH:-}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
ELXR_INIT_SCRIPT="${ELXR_INIT_SCRIPT:-/opt/bin/simaai-init-build-env}"
ELXR_MACHINE="${ELXR_MACHINE:-modalix}"
ELXR_SDK=OFF
ELXR_SDK_VERSION=""
ELXR_VERSION=""
ELXR_WHEEL_HOST_PLATFORM="${ELXR_WHEEL_HOST_PLATFORM:-}"

# ------------------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------------------
SYSTEM_DEPS_LINUX=(
  build-essential
  cmake
  pkg-config
  git
  curl
  wget
  doxygen
  graphviz
  ffmpeg
  libgstreamer1.0-dev
  libgstreamer-plugins-base1.0-dev
  libgstreamer-plugins-bad1.0-dev
  gstreamer1.0-plugins-base
  gstreamer1.0-plugins-good
  gstreamer1.0-plugins-bad
  gstreamer1.0-plugins-ugly
  gstreamer1.0-libav
  gstreamer1.0-tools
  libopenh264-dev
  libgstrtspserver-1.0-dev
  libglib2.0-dev
  libopencv-dev
  nlohmann-json3-dev
  clang
  clang-format
  ripgrep
)

SYSTEM_DEPS_MAC=(
  cmake
  pkg-config
  git
  curl
  wget
  doxygen
  graphviz
  gstreamer
  gst-plugins-base
  gst-plugins-bad
  gst-rtsp-server
  glib
  opencv
  nlohmann-json
)

SELECTED_SYSTEM_DEPS_LINUX=()
SELECTED_SYSTEM_DEPS_MAC=()
BUILD_JOBS=""

detect_fs_type() {
  local path="$1"
  local fs_type=""

  if command -v findmnt >/dev/null 2>&1; then
    fs_type="$(findmnt -T "${path}" -n -o FSTYPE 2>/dev/null | head -n1 || true)"
  fi

  if [[ -z "${fs_type}" ]]; then
    if [[ "${OS_NAME}" == "Darwin" ]]; then
      fs_type="$(stat -f %T "${path}" 2>/dev/null || true)"
    else
      fs_type="$(stat -f -c %T "${path}" 2>/dev/null || true)"
    fi
  fi

  printf '%s\n' "${fs_type}"
}

is_remote_fs_type() {
  local fs_type="${1,,}"
  case "${fs_type}" in
    cifs|nfs|nfs4|smb2|smb3|smbfs|fuse.sshfs)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

shadow_workspace_path() {
  local key
  key="$(printf '%s\n' "${ORIGINAL_WORKSPACE_ROOT}" | cksum | awk '{print $1}')"
  printf '%s\n' "${SIMANEAT_SHADOW_ROOT:-/tmp/sima-neat-shadow-${key}}"
}

prepare_shadow_workspace() {
  local shadow_root="$1"

  rm -rf "${shadow_root}"
  mkdir -p "${shadow_root}"

  tar -C "${ORIGINAL_WORKSPACE_ROOT}" \
    --exclude="core/.git" \
    --exclude="core/build" \
    --exclude="core/build-*" \
    --exclude="core/Testing" \
    --exclude="core/dist" \
    --exclude="core/tmp" \
    --exclude="core/.pytest_cache" \
    --exclude="core/.build-py" \
    --exclude="core/__pycache__" \
    --exclude="core/website/node_modules" \
    --exclude="core/neat-internals/gst-plugins" \
    --exclude="internals/.git" \
    --exclude="internals/build" \
    --exclude="internals/build-*" \
    --exclude="internals/dist" \
    --exclude="internals/tmp" \
    -cf - \
    core \
    internals/core \
    internals/gst_plugins \
    internals/sima-ai-cvu-sw/graphs \
    | tar -C "${shadow_root}" -xf -
}

maybe_reexec_from_shadow() {
  if [[ "${SHADOW_BUILD_ACTIVE}" == "ON" || "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
    return 0
  fi

  SOURCE_FS_TYPE="$(detect_fs_type "${ORIGINAL_SOURCE_DIR}")"

  case "${SHADOW_BUILD_MODE,,}" in
    off|0|false|no)
      return 0
      ;;
    force|on|1|true|yes)
      ;;
    auto|"")
      if ! is_remote_fs_type "${SOURCE_FS_TYPE}"; then
        return 0
      fi
      ;;
    *)
      echo "ERROR: Unsupported SIMANEAT_SHADOW_BUILD mode: ${SHADOW_BUILD_MODE}" >&2
      echo "Use one of: auto, off, force" >&2
      exit 1
      ;;
  esac

  local shadow_root
  shadow_root="$(shadow_workspace_path)"

  echo "Detected remote source filesystem: ${SOURCE_FS_TYPE:-unknown}"
  echo "Preparing local shadow workspace at ${shadow_root}"
  prepare_shadow_workspace "${shadow_root}"

  echo "Re-running build from shadow workspace..."
  (
    cd "${shadow_root}/core"
    SIMANEAT_SHADOW_BUILD_ACTIVE=ON \
    SIMANEAT_ORIGINAL_SOURCE_DIR="${ORIGINAL_SOURCE_DIR}" \
    SIMANEAT_ORIGINAL_WORKSPACE_ROOT="${ORIGINAL_WORKSPACE_ROOT}" \
    bash "${shadow_root}/core/build.sh" "$@"
  )
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "Shadow build failed. Workspace retained at ${shadow_root}" >&2
    exit "${rc}"
  fi

  echo
  echo "Shadow build completed successfully."
  echo "Shadow source   : ${shadow_root}/core"
  echo "Shadow build dir: ${shadow_root}/core/${BUILD_DIR}"
  exit 0
}

usage() {
  cat <<USAGE
Usage: ./build.sh [options]

Options:
  --dev-only     Build only the core library + headers (DEFAULT)
  --all          Build library + samples + tests + Python wheel
  --example      Build only examples (and core library)
  --python       Build Python bindings (pyneat) in addition to selected targets
  --install-neat-internals
                 Download/install neat-internals artifacts before build
  --internals-manifest <path>
                 Manifest that selects neat-internals artifact tag (default: deps/manifest.json)
  --doc          Build only docs
  --no-dist      Skip DEB packaging
  --clean        Remove build directory before building
  --no-doc       Skip documentation build (even with --all)
  --no-node      Skip Node.js install (docs build will not work)
  --install-deps-only
                 Install system dependencies only, then exit
  -h, --help     Show this help

Environment:
  SIMANEAT_SHADOW_BUILD=auto|off|force
                 Auto-stage a local shadow workspace for builds from remote filesystems.
  SIMANEAT_SHADOW_ROOT=/tmp/sima-neat-shadow-<id>
                 Override where the shadow workspace is created.

Examples:
  ./build.sh
  ./build.sh --dev-only
  ./build.sh --all
  ./build.sh --doc
  ./build.sh --all --clean
USAGE
}

parse_args() {
  # Parse CLI flags and map them directly to build-mode toggles.
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dev-only)
        BUILD_SAMPLES=OFF
        BUILD_TESTS=OFF
        shift
        ;;
      --all)
        BUILD_SAMPLES=ON
        BUILD_TESTS=ON
        BUILD_DOCS=ON
        BUILD_PYTHON=ON
        INSTALL_NEAT_INTERNALS=ON
        BUILD_ALL=ON
        shift
        ;;
      --doc)
        BUILD_SAMPLES=OFF
        BUILD_TESTS=OFF
        BUILD_DOCS=ON
        DOCS_ONLY=ON
        shift
        ;;
      --python)
        BUILD_PYTHON=ON
        shift
        ;;
      --install-neat-internals)
        INSTALL_NEAT_INTERNALS=ON
        shift
        ;;
      --internals-manifest)
        NEAT_INTERNALS_MANIFEST="${2:-}"
        if [[ -z "${NEAT_INTERNALS_MANIFEST}" ]]; then
          echo "ERROR: --internals-manifest requires a path" >&2
          exit 1
        fi
        shift 2
        ;;
      --example)
        BUILD_SAMPLES=ON
        BUILD_TESTS=OFF
        BUILD_DOCS=OFF
        EXAMPLES_ONLY=ON
        shift
        ;;
      --clean)
        DO_CLEAN=ON
        shift
        ;;
      --no-node)
        INSTALL_NODE=OFF
        shift
        ;;
      --no-doc)
        SKIP_DOCS=ON
        BUILD_DOCS=OFF
        shift
        ;;
      --no-dist)
        SKIP_DIST=ON
        shift
        ;;
      --install-deps-only)
        INSTALL_DEPS_ONLY=ON
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

detect_elxr_sdk() {
  # Detect eLxr SDK environments from /etc/sdk-release style metadata.
  ELXR_SDK=OFF
  ELXR_SDK_VERSION=""
  ELXR_VERSION=""
  if [[ ! -f "${ELXR_SDK_RELEASE_FILE}" ]]; then
    return 0
  fi

  ELXR_SDK_VERSION="$(sed -n 's/^SDK Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
  ELXR_VERSION="$(sed -n 's/^eLXr Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
  if [[ -n "${ELXR_SDK_VERSION}" && -n "${ELXR_VERSION}" ]]; then
    ELXR_SDK=ON
  fi
}

detect_elxr_wheel_platform() {
  # Prefer explicit override; otherwise derive from target-arch hints set by SDK env.
  if [[ -n "${ELXR_WHEEL_HOST_PLATFORM}" ]]; then
    return 0
  fi

  case "${OECORE_TARGET_ARCH:-}" in
    aarch64|arm64)
      ELXR_WHEEL_HOST_PLATFORM="linux-aarch64"
      ;;
    x86_64|amd64)
      ELXR_WHEEL_HOST_PLATFORM="linux-x86_64"
      ;;
    armv7*|armhf)
      ELXR_WHEEL_HOST_PLATFORM="linux-armv7l"
      ;;
    *)
      ELXR_WHEEL_HOST_PLATFORM="linux-aarch64"
      ;;
  esac
}

elxr_ext_platform_triplet() {
  case "${ELXR_WHEEL_HOST_PLATFORM}" in
    linux-aarch64)
      printf '%s\n' "aarch64-linux-gnu"
      ;;
    linux-x86_64)
      printf '%s\n' "x86_64-linux-gnu"
      ;;
    linux-armv7l)
      printf '%s\n' "arm-linux-gnueabihf"
      ;;
    *)
      printf '%s\n' "aarch64-linux-gnu"
      ;;
  esac
}

activate_elxr_build_env_if_needed() {
  # eLxr SDK builds must source SiMa's init script so CMake/builds run in SDK sysroot context.
  if [[ "${ELXR_SDK}" != "ON" ]]; then
    return 0
  fi
  if [[ ! -f "${ELXR_INIT_SCRIPT}" ]]; then
    echo "ERROR: eLxr SDK detected, but init script is missing: ${ELXR_INIT_SCRIPT}" >&2
    exit 1
  fi

  echo "Detected eLxr SDK:"
  echo "  SDK Version  : ${ELXR_SDK_VERSION}"
  echo "  eLXr Version : ${ELXR_VERSION}"
  echo "Activating eLxr build environment: source ${ELXR_INIT_SCRIPT} ${ELXR_MACHINE}"
  # shellcheck disable=SC1090
  source "${ELXR_INIT_SCRIPT}" "${ELXR_MACHINE}"
  detect_elxr_wheel_platform
}

select_system_deps() {
  # Start from the baseline dependency lists for each OS.
  SELECTED_SYSTEM_DEPS_LINUX=("${SYSTEM_DEPS_LINUX[@]}")
  SELECTED_SYSTEM_DEPS_MAC=("${SYSTEM_DEPS_MAC[@]}")

  # Python runtime is required for docs helper scripts and wheel builds.
  if [[ "${BUILD_DOCS}" == "ON" || "${BUILD_PYTHON}" == "ON" ]]; then
    SELECTED_SYSTEM_DEPS_LINUX+=(python3 python3-venv)
    SELECTED_SYSTEM_DEPS_MAC+=(python)
  fi
}

run_privileged() {
  # Run command as root when possible:
  # 1) direct if already root
  # 2) non-interactive sudo if available
  # 3) interactive sudo only in TTY sessions
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
      sudo -n "$@"
      return 0
    fi

    # Fallback for local/dev environments where sudo requires a password.
    if [[ -t 0 && -t 1 ]]; then
      sudo "$@"
      return $?
    fi
  fi

  return 1
}

is_dep_satisfied_linux() {
  local pkg="$1"
  # Some tools are version-suffixed on devkits, so check common aliases first.
  case "${pkg}" in
    clang)
      command -v clang >/dev/null 2>&1 || command -v clang-18 >/dev/null 2>&1 || command -v clang-17 >/dev/null 2>&1
      return $?
      ;;
    clang-format)
      command -v clang-format >/dev/null 2>&1 || command -v clang-format-18 >/dev/null 2>&1 || command -v clang-format-17 >/dev/null 2>&1
      return $?
      ;;
    ripgrep)
      command -v rg >/dev/null 2>&1
      return $?
      ;;
    cmake|pkg-config|git|curl|wget|doxygen|ffmpeg|python3)
      command -v "${pkg}" >/dev/null 2>&1
      return $?
      ;;
  esac

  # Default path for package-style dependencies.
  dpkg -s "${pkg}" >/dev/null 2>&1
}

install_system_deps_mac() {
  # Homebrew is required for dependency bootstrap on macOS.
  if ! command -v brew >/dev/null 2>&1; then
    echo "Error: Homebrew not found. Please install Homebrew, then re-run this script."
    exit 1
  fi

  local missing_deps=()
  local pkg
  # Build a minimal install set by checking only missing formulas.
  for pkg in "${SELECTED_SYSTEM_DEPS_MAC[@]}"; do
    if ! brew list --formula "${pkg}" >/dev/null 2>&1; then
      missing_deps+=("${pkg}")
    fi
  done

  if (( ${#missing_deps[@]} > 0 )); then
    echo "Missing system dependencies: ${missing_deps[*]}"
    echo "Installing system dependencies via Homebrew..."
    brew install "${missing_deps[@]}"
  fi
}

install_system_deps_linux() {
  # apt-get is the supported package manager path on Linux.
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: apt-get not found. This script currently supports Debian/Ubuntu or macOS."
    echo "Please install build dependencies manually on your OS."
    exit 1
  fi

  local missing_deps=()
  local pkg
  # Only install packages that are genuinely missing.
  for pkg in "${SELECTED_SYSTEM_DEPS_LINUX[@]}"; do
    if ! is_dep_satisfied_linux "${pkg}"; then
      missing_deps+=("${pkg}")
    fi
  done

  if (( ${#missing_deps[@]} > 0 )); then
    echo "Missing system dependencies: ${missing_deps[*]}"
    echo "Updating package index..."
    if ! run_privileged apt-get update; then
      echo "Cannot run apt-get update without root or passwordless sudo."
      echo "Please preinstall: ${missing_deps[*]}"
      exit 1
    fi

    echo "Installing system dependencies..."
    if ! run_privileged apt-get install -y "${missing_deps[@]}"; then
      echo "Cannot install dependencies without root or passwordless sudo."
      echo "Please preinstall: ${missing_deps[*]}"
      exit 1
    fi
  fi
}

install_system_deps() {
  # Dispatch dependency installation by host OS.
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    install_system_deps_mac
  else
    install_system_deps_linux
  fi
}

extract_json_string() {
  # Lightweight JSON field extraction for simple manifest keys.
  local key="$1"
  local file="$2"
  sed -n "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" "${file}" | head -n1
}

download_file() {
  local url="$1"
  local out="$2"
  local basic_auth="${3:-}"
  # Prefer curl; fall back to wget for minimal environments.
  if command -v curl >/dev/null 2>&1; then
    if [[ -n "${basic_auth}" ]]; then
      curl -fL -u "${basic_auth}" "${url}" -o "${out}"
    else
      curl -fL "${url}" -o "${out}"
    fi
    return $?
  fi
  if command -v wget >/dev/null 2>&1; then
    if [[ -n "${basic_auth}" ]]; then
      local wget_user="${basic_auth%%:*}"
      local wget_pass="${basic_auth#*:}"
      if [[ "${wget_user}" == "${wget_pass}" ]]; then
        echo "ERROR: NEAT_INTERNALS_BASIC_AUTH must be in 'user:password' format for wget." >&2
        return 1
      fi
      wget --user="${wget_user}" --password="${wget_pass}" -O "${out}" "${url}"
    else
      wget -O "${out}" "${url}"
    fi
    return $?
  fi
  return 1
}

compute_sha256() {
  local path="$1"
  # Prefer GNU sha256sum; fall back to shasum for macOS compatibility.
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
    return 0
  fi
  return 1
}

find_legacy_plugin_dir() {
  # Support legacy tarball layouts where plugins were shipped directly as files.
  local extract_dir="$1"
  local extracted_plugin_dir="${extract_dir}"

  if [[ -d "${extract_dir}/gst-plugins" ]]; then
    printf '%s\n' "${extract_dir}/gst-plugins"
    return 0
  fi

  local nested_plugin_dir
  # Handle nested archives that tuck gst-plugins below one or more directories.
  nested_plugin_dir="$(find "${extract_dir}" -mindepth 1 -maxdepth 4 -type d -name gst-plugins | head -n1 || true)"
  if [[ -n "${nested_plugin_dir}" ]]; then
    printf '%s\n' "${nested_plugin_dir}"
    return 0
  fi

  # Support archives that contain a single top-level directory with the plugin files.
  mapfile -t top_dirs < <(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d)
  if [[ "${#top_dirs[@]}" -eq 1 ]]; then
    extracted_plugin_dir="${top_dirs[0]}"
  fi

  printf '%s\n' "${extracted_plugin_dir}"
}

collect_plugin_files_from_debs() {
  local extract_dir="$1"
  local output_file="$2"
  local install_root="${SYSROOT:-}"

  # New artifact model: tarball contains one or more .deb files.
  mapfile -t deb_files < <(find "${extract_dir}" -type f -name '*.deb' | sort)
  if (( ${#deb_files[@]} == 0 )); then
    return 1
  fi

  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "ERROR: dpkg-deb is required to extract .so files from neat-internals .deb packages." >&2
    exit 1
  fi

  local -a host_install_debs=()
  declare -A host_install_by_deb=()
  local skipped_debug_debs=0
  local deb_filter_path
  for deb_filter_path in "${deb_files[@]}"; do
    local filter_pkg_name=""
    filter_pkg_name="$(dpkg-deb -f "${deb_filter_path}" Package 2>/dev/null || true)"
    # Debug-symbol packages are optional and add noise to host installs.
    if [[ "${filter_pkg_name}" == *-dbgsym ]]; then
      ((skipped_debug_debs += 1))
      continue
    fi
    host_install_debs+=("${deb_filter_path}")
    host_install_by_deb["${deb_filter_path}"]=1
  done

  if [[ "${ELXR_SDK}" == "ON" ]]; then
    if [[ -z "${install_root}" ]]; then
      echo "ERROR: eLxr SDK mode requires SYSROOT to be set before installing neat-internals .deb packages." >&2
      exit 1
    fi
    if ! mkdir -p "${install_root}" 2>/dev/null; then
      if ! run_privileged mkdir -p "${install_root}"; then
        echo "ERROR: Unable to create SYSROOT directory: ${install_root}" >&2
        exit 1
      fi
    fi
    echo "Installing neat-internals .deb payloads into eLxr sysroot: ${install_root}"
  else
    if (( ${#host_install_debs[@]} == 0 )); then
      echo "ERROR: No installable neat-internals .deb packages were found (all packages were debug symbols)." >&2
      exit 1
    fi

    if (( skipped_debug_debs > 0 )); then
      echo "Skipping ${skipped_debug_debs} debug-symbol package(s) for host install."
    fi

    # Use apt-get for dependency-aware local package installation.
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing neat-internals .deb packages into host system via apt-get..."
      if ! run_privileged apt-get install -y "${host_install_debs[@]}"; then
        echo "WARNING: apt-get install failed; falling back to dpkg -i + apt-get -f." >&2
        if ! command -v dpkg >/dev/null 2>&1; then
          echo "ERROR: dpkg is required to install neat-internals .deb packages." >&2
          exit 1
        fi
        if ! run_privileged dpkg -i "${host_install_debs[@]}"; then
          if ! run_privileged apt-get install -f -y; then
            echo "ERROR: Failed to install neat-internals .deb packages." >&2
            exit 1
          fi
        fi
      fi
    else
      if ! command -v dpkg >/dev/null 2>&1; then
        echo "ERROR: dpkg is required to install neat-internals .deb packages." >&2
        exit 1
      fi
      echo "Installing neat-internals .deb packages into host system via dpkg..."
      if ! run_privileged dpkg -i "${host_install_debs[@]}"; then
        echo "ERROR: Failed to install neat-internals .deb packages." >&2
        exit 1
      fi
    fi
  fi

  local deb_extract_root
  # Also unpack each .deb into temp space to collect .so files for local mirror.
  deb_extract_root="$(dirname "${output_file}")/deb-extract"
  mkdir -p "${deb_extract_root}"

  : > "${output_file}"

  local deb_path
  for deb_path in "${deb_files[@]}"; do
    local deb_name
    deb_name="$(basename "${deb_path}" .deb)"
    local pkg_name
    local pkg_version
    pkg_name="$(dpkg-deb -f "${deb_path}" Package 2>/dev/null || true)"
    pkg_version="$(dpkg-deb -f "${deb_path}" Version 2>/dev/null || true)"

    local deb_extract_dir="${deb_extract_root}/${deb_name}"
    mkdir -p "${deb_extract_dir}"
    dpkg-deb -x "${deb_path}" "${deb_extract_dir}"

    local installed_on_host=0
    if [[ "${ELXR_SDK}" != "ON" && -n "${host_install_by_deb["${deb_path}"]:-}" ]]; then
      installed_on_host=1
    fi

    if [[ "${ELXR_SDK}" == "ON" ]]; then
      if ! cp -a "${deb_extract_dir}/." "${install_root}/" 2>/dev/null; then
        if ! run_privileged cp -a "${deb_extract_dir}/." "${install_root}/"; then
          echo "ERROR: Failed to install ${deb_name}.deb into SYSROOT=${install_root}" >&2
          exit 1
        fi
      fi
    fi

    if [[ -n "${pkg_name}" ]]; then
      if [[ -n "${pkg_version}" ]]; then
        echo "Processed package: ${pkg_name} (${pkg_version})"
      else
        echo "Processed package: ${pkg_name}"
      fi
    else
      echo "Processed package from ${deb_name}.deb (unable to resolve package metadata)."
    fi

    if [[ "${ELXR_SDK}" == "ON" ]]; then
      echo "SYSROOT library paths:"
      local found_sysroot_paths=0
      local rel_path
      for rel_path in usr/lib usr/libexec; do
        if [[ -d "${deb_extract_dir}/${rel_path}" ]]; then
          found_sysroot_paths=1
          find "${deb_extract_dir}/${rel_path}" -mindepth 1 -print | \
            sed "s|^${deb_extract_dir}|${install_root}|"
        fi
      done | awk '{ print "  " $0 }'
      if [[ "${found_sysroot_paths}" -eq 0 ]]; then
        echo "  (no /usr/lib* or /usr/libexec paths reported)"
      fi
    elif [[ "${installed_on_host}" -eq 1 ]]; then
      echo "System library paths:"
      # Print user-visible install destinations for quick verification/debugging.
      if [[ -n "${pkg_name}" ]]; then
        dpkg -L "${pkg_name}" 2>/dev/null | awk '
          /^\/usr\/lib/ || /^\/usr\/libexec/ { print "  " $0; found=1 }
          END { if (!found) print "  (no /usr/lib* or /usr/libexec paths reported)" }
        '
      else
        echo "  (package name unavailable; unable to query dpkg -L)"
      fi
    fi

    find "${deb_extract_dir}" \
      \( -type f -o -type l \) \
      \( -name '*.so' -o -name '*.so.*' \) \
      -print >> "${output_file}"
  done

  return 0
}

collect_plugin_files_from_legacy_tar() {
  # Backward compatibility: handle old tarballs containing plugin .so files.
  local extract_dir="$1"
  local output_file="$2"

  local extracted_plugin_dir
  extracted_plugin_dir="$(find_legacy_plugin_dir "${extract_dir}")"

  find "${extracted_plugin_dir}" -maxdepth 1 -type f \
    \( -name '*.so' -o -name '*.so.*' -o -name 'dispatcher_watchdog' \) \
    -print > "${output_file}"
}

copy_plugins_to_neat_internals() {
  # Mirror plugin binaries into repo-local neat-internals for source builds/tests.
  local plugins_list_file="$1"

  if [[ ! -s "${plugins_list_file}" ]]; then
    echo "ERROR: Downloaded artifact does not contain extractable .so files." >&2
    exit 1
  fi

  mkdir -p "${NEAT_INTERNALS_DIR}"
  rm -rf "${NEAT_INTERNALS_PLUGIN_DIR}"
  mkdir -p "${NEAT_INTERNALS_PLUGIN_DIR}"

  declare -A plugin_by_name=()
  local plugin_path
  # Deduplicate by basename to avoid cp collisions from overlapping package payloads.
  while IFS= read -r plugin_path; do
    [[ -z "${plugin_path}" ]] && continue
    plugin_by_name["$(basename "${plugin_path}")"]="${plugin_path}"
  done < "${plugins_list_file}"

  mapfile -t plugin_names < <(printf '%s\n' "${!plugin_by_name[@]}" | sort)
  local plugin_name
  for plugin_name in "${plugin_names[@]}"; do
    cp -a "${plugin_by_name[${plugin_name}]}" "${NEAT_INTERNALS_PLUGIN_DIR}/${plugin_name}"
  done
}

ensure_neat_internals() {
  # Sync neat-internals from remote artifact, validate integrity, then materialize plugins.
  if [[ ! -f "${NEAT_INTERNALS_MANIFEST}" ]]; then
    echo "ERROR: Missing manifest: ${NEAT_INTERNALS_MANIFEST}" >&2
    exit 1
  fi

  local artifact_tag
  # Manifest drives which artifact tag to fetch.
  artifact_tag="$(extract_json_string "internals" "${NEAT_INTERNALS_MANIFEST}")"
  if [[ -z "${artifact_tag}" ]]; then
    artifact_tag="$(extract_json_string "artifact_tag" "${NEAT_INTERNALS_MANIFEST}")"
  fi
  if [[ -z "${artifact_tag}" ]]; then
    echo "ERROR: ${NEAT_INTERNALS_MANIFEST} must define a non-empty internals string." >&2
    exit 1
  fi

  local marker_file="${NEAT_INTERNALS_DIR}/.artifact_tag"
  local checksum_file="${NEAT_INTERNALS_DIR}/.artifact_sha256"
  local archive_name="sima-neat-internals-${artifact_tag}.tar.gz"
  local archive_url="${NEAT_INTERNALS_BASE_URL}/${archive_name}"
  local checksum_url="${archive_url}.sha256"

  local tmp_dir
  # Use an isolated temp workspace so partial failures do not pollute tree state.
  tmp_dir="$(mktemp -d /tmp/sima-neat-internals-XXXXXX)"

  local checksum_path="${tmp_dir}/${archive_name}.sha256"
  local archive_path="${tmp_dir}/${archive_name}"
  local extract_dir="${tmp_dir}/extract"
  local plugins_list_file="${tmp_dir}/plugin-files.list"

  if ! download_file "${checksum_url}" "${checksum_path}" "${NEAT_INTERNALS_BASIC_AUTH}"; then
    echo "ERROR: Unable to download checksum sidecar: ${checksum_url}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  local server_sha
  server_sha="$(awk '{print $1}' "${checksum_path}" | tr -d '[:space:]' | head -n1)"
  if [[ -z "${server_sha}" || ! "${server_sha}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: Invalid sha256 content in ${checksum_url}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  if [[ -f "${marker_file}" ]] && [[ -d "${NEAT_INTERNALS_PLUGIN_DIR}" ]]; then
    local current_tag
    current_tag="$(tr -d '[:space:]' < "${marker_file}")"
    local cached_sha=""
    if [[ -f "${checksum_file}" ]]; then
      cached_sha="$(tr -d '[:space:]' < "${checksum_file}")"
    fi
    # Cache hit requires matching tag, checksum, and a known plugin sentinel.
    if [[ "${current_tag}" == "${artifact_tag}" ]] &&
       [[ "${cached_sha}" == "${server_sha}" ]] &&
       [[ -f "${NEAT_INTERNALS_PLUGIN_DIR}/libgstneatdecoder.so" ]]; then
      echo "Using cached neat-internals plugins (${artifact_tag}, sha256=${server_sha})."
      rm -rf "${tmp_dir}"
      return 0
    fi
  fi

  echo "Fetching neat-internals plugins: ${archive_url} (sha256=${server_sha})"

  mkdir -p "${extract_dir}"

  if ! download_file "${archive_url}" "${archive_path}" "${NEAT_INTERNALS_BASIC_AUTH}"; then
    echo "ERROR: curl or wget is required to download neat-internals artifacts." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  local actual_sha
  if ! actual_sha="$(compute_sha256 "${archive_path}")"; then
    echo "ERROR: Unable to compute sha256 checksum for ${archive_path}." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi
  # Hard fail on checksum mismatch; never use unverified internals artifacts.
  if [[ "${actual_sha}" != "${server_sha}" ]]; then
    echo "ERROR: sha256 mismatch for ${archive_name}" >&2
    echo "  expected: ${server_sha}" >&2
    echo "  actual  : ${actual_sha}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  tar -xzf "${archive_path}" -C "${extract_dir}"

  # Prefer .deb flow; fall back to legacy flat-plugin tarballs for compatibility.
  if ! collect_plugin_files_from_debs "${extract_dir}" "${plugins_list_file}"; then
    collect_plugin_files_from_legacy_tar "${extract_dir}" "${plugins_list_file}"
  fi

  copy_plugins_to_neat_internals "${plugins_list_file}"

  printf '%s\n' "${artifact_tag}" > "${marker_file}"
  printf '%s\n' "${server_sha}" > "${checksum_file}"
  rm -rf "${tmp_dir}"
}

ensure_node20_for_docs() {
  # Docs toolchain expects Node.js 20.x.
  if [[ "${INSTALL_NODE}" != "ON" || "${BUILD_DOCS}" != "ON" ]]; then
    return 0
  fi

  local node_major=""
  if command -v node >/dev/null 2>&1; then
    node_major="$(node -v | sed 's/^v//' | cut -d. -f1 || true)"
  fi

  # Auto-install Node 20 only when current version is missing/too old.
  if [[ -z "${node_major}" || "${node_major}" -lt 20 ]]; then
    if [[ "${OS_NAME}" == "Darwin" ]]; then
      echo "Installing Node.js 20.x via Homebrew..."
      brew install node@20
      brew link --overwrite --force node@20
    else
      echo "Installing Node.js 20.x..."
      if ! run_privileged bash -c "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"; then
        echo "Cannot configure NodeSource without root or passwordless sudo."
        echo "Please preinstall Node.js 20.x or run with --no-node."
        exit 1
      fi
      if ! run_privileged apt-get install -y nodejs; then
        echo "Cannot install Node.js without root or passwordless sudo."
        echo "Please preinstall Node.js 20.x or run with --no-node."
        exit 1
      fi
    fi
  else
    echo "Node.js ${node_major}.x already installed."
  fi
}

detect_build_jobs() {
  # Derive parallelism from env override or machine CPU count.
  if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL}"
  elif command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
  elif [[ "${OS_NAME}" == "Darwin" ]]; then
    BUILD_JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  else
    BUILD_JOBS=""
  fi

  if [[ -z "${BUILD_JOBS}" || "${BUILD_JOBS}" -lt 1 ]]; then
    BUILD_JOBS=8
  fi
}

print_build_config() {
  # Emit resolved configuration so build-mode decisions are explicit.
  echo "========================================"
  echo " SiMa NEAT build configuration"
  echo "========================================"
  echo "Build type     : ${BUILD_TYPE}"
  echo "Build samples  : ${BUILD_SAMPLES}"
  echo "Build tests    : ${BUILD_TESTS}"
  echo "Build docs     : ${BUILD_DOCS}"
  echo "Build python   : ${BUILD_PYTHON}"
  echo "Build all      : ${BUILD_ALL}"
  echo "Neat internals : ${INSTALL_NEAT_INTERNALS}"
  echo "Examples only  : ${EXAMPLES_ONLY}"
  echo "Skip dist      : ${SKIP_DIST}"
  echo "Skip docs      : ${SKIP_DOCS}"
  echo "Clean build    : ${DO_CLEAN}"
  echo "Strict warns   : ${STRICT_WARNINGS}"
  echo "Source dir     : ${SCRIPT_DIR}"
  echo "Source fs      : ${SOURCE_FS_TYPE:-$(detect_fs_type "${SCRIPT_DIR}")}"
  echo "Shadow build   : ${SHADOW_BUILD_ACTIVE}"
  if [[ "${SHADOW_BUILD_ACTIVE}" == "ON" ]]; then
    echo "Original source: ${ORIGINAL_SOURCE_DIR}"
  fi
  echo "Build dir      : ${BUILD_DIR}"
  echo "Build jobs     : ${BUILD_JOBS}"
  echo "eLxr SDK       : ${ELXR_SDK}"
  if [[ "${ELXR_SDK}" == "ON" ]]; then
    echo "eLxr SDK ver   : ${ELXR_SDK_VERSION}"
    echo "eLXr ver       : ${ELXR_VERSION}"
    echo "Wheel platform : ${ELXR_WHEEL_HOST_PLATFORM}"
  fi
  echo "========================================"
}

clean_build_dir_if_requested() {
  if [[ "${DO_CLEAN}" == "ON" ]]; then
    echo
    echo "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
}

configure_cmake() {
  # Configure once; subsequent steps reuse this build tree.
  local neat_internals_root=""
  if [[ "${ELXR_SDK}" == "ON" && -n "${SYSROOT:-}" ]]; then
    neat_internals_root="${SYSROOT}/usr"
  elif [[ -d /usr/lib/aarch64-linux-gnu/cmake/NeatInternals || -d /usr/lib/cmake/NeatInternals ]]; then
    neat_internals_root="/usr"
  fi

  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DSIMANEAT_BUILD_SAMPLES="${BUILD_SAMPLES}" \
    -DSIMANEAT_BUILD_TESTS="${BUILD_TESTS}" \
    -DSIMANEAT_BUILD_TUTORIALS="${BUILD_TESTS}" \
    -DSIMANEAT_BUILD_PYTHON="${BUILD_PYTHON}" \
    -DSIMANEAT_STRICT_WARNINGS="${STRICT_WARNINGS}" \
    ${neat_internals_root:+-DNeatInternals_ROOT="${neat_internals_root}"}
}

build_docs_site() {
  # Shared docs pipeline used by both --doc and --all/--no-doc flows.
  echo
  echo "Building docs..."
  cmake --build "${BUILD_DIR}" --target docs -j"${BUILD_JOBS}"
  echo
  echo "Generating API docs from Doxygen XML..."
  bash tools/generate_api_docs.sh
  echo
  echo "Generating Python API reference placeholders..."
  python3 tools/generate_python_api_docs.py --repo-root .
  echo
  echo "Generating tutorial docs from C++ sources..."
  python3 tools/generate_tutorial_docs.py --repo-root .
  echo
  echo "Expanding code tabs..."
  local expanded_docs_dir
  expanded_docs_dir="$(pwd)/${BUILD_DIR}/docs-expanded"
  python3 tools/expand_code_tabs.py --src docs --dst "${expanded_docs_dir}"
  if [[ "${INSTALL_NODE}" == "ON" ]]; then
    echo
    echo "Installing website dependencies..."
    npm --prefix website install
  fi
  echo
  echo "Building Docusaurus site..."
  DOCS_PATH="${expanded_docs_dir}" npm --prefix website run build
}

build_docs_only_if_requested() {
  if [[ "${DOCS_ONLY}" != "ON" ]]; then
    return 0
  fi

  echo
  echo "Building docs only..."
  build_docs_site
  echo
  echo "========================================"
  echo " Docs build completed successfully"
  echo "========================================"
  exit 0
}

build_targets() {
  # For dev-only builds, avoid building tests/tutorials/samples by targeting core lib.
  if [[ "${EXAMPLES_ONLY}" == "ON" ]]; then
    # Build only example executables (and their dependencies).
    mapfile -t EXAMPLE_BLACKLIST < <(
      # Parse SIMANEAT_EXAMPLE_BLACKLIST from CMake to stay aligned with source of truth.
      awk '
        /set\(SIMANEAT_EXAMPLE_BLACKLIST/ {in_list=1; next}
        in_list {
          if ($0 ~ /\)/) {exit}
          sub(/#.*/, "", $0)
          gsub(/[[:space:]]+/, "", $0)
          if (length($0) > 0) print $0
        }
      ' examples/CMakeLists.txt
    )

    EXAMPLE_TARGETS=()
    local src fname
    # Build all examples except helper translation units and blacklisted files.
    for src in examples/*.cpp; do
      fname="$(basename "${src}")"
      if [[ "${fname}" == "example_utils.cpp" ]]; then
        continue
      fi
      if [[ " ${EXAMPLE_BLACKLIST[*]} " == *" ${fname} "* ]]; then
        continue
      fi
      EXAMPLE_TARGETS+=("${fname%.cpp}")
    done

    if (( ${#EXAMPLE_TARGETS[@]} == 0 )); then
      echo "ERROR: No example targets found to build."
      exit 1
    fi

    cmake --build "${BUILD_DIR}" --target "${EXAMPLE_TARGETS[@]}" -j"${BUILD_JOBS}"
  elif [[ "${BUILD_SAMPLES}" == "OFF" && "${BUILD_TESTS}" == "OFF" && "${BUILD_DOCS}" == "OFF" ]]; then
    cmake --build "${BUILD_DIR}" --target sima_neat -j"${BUILD_JOBS}"
  else
    cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"
  fi
}

copy_test_images() {
  # Ensure runtime tests/examples can find expected image assets in build tree.
  if [[ -d "tests/images" ]]; then
    mkdir -p "${BUILD_DIR}/tests/images"
    cp -a tests/images/. "${BUILD_DIR}/tests/images/"
  fi
}

build_docs_if_requested() {
  if [[ "${BUILD_DOCS}" == "ON" && "${SKIP_DOCS}" == "OFF" ]]; then
    build_docs_site
  fi
}

build_python_wheel_if_requested() {
  if [[ "${BUILD_PYTHON}" != "ON" ]]; then
    return 0
  fi

  echo
  echo "Building Python wheel (pyneat)..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found; cannot build wheel."
    exit 1
  fi

  local wheel_python="python3"
  # Use isolated venv if python-build module is missing on the host.
  if ! python3 -m build --version >/dev/null 2>&1; then
    local venv_dir="${BUILD_DIR}/.wheel-venv"
    echo "Python 'build' module not found; creating isolated venv at ${venv_dir}..."
    if ! python3 -m venv "${venv_dir}"; then
      echo "ERROR: failed to create venv. Install python3-venv and retry."
      exit 1
    fi
    wheel_python="${venv_dir}/bin/python"
    "${wheel_python}" -m pip install --upgrade pip build
  fi

  rm -rf dist
  if [[ "${ELXR_SDK}" == "ON" ]]; then
    echo "Using eLxr wheel target platform: ${ELXR_WHEEL_HOST_PLATFORM}"
    echo "Preparing non-isolated wheel backend environment for cross-build..."
    "${wheel_python}" -m pip install --upgrade pip build scikit-build-core nanobind ninja
    local py_abi
    local py_triplet
    local pyneat_ext_suffix
    py_abi="$("${wheel_python}" - <<'PY'
import sys
print(f"{sys.version_info.major}{sys.version_info.minor}")
PY
)"
    py_triplet="$(elxr_ext_platform_triplet)"
    pyneat_ext_suffix=".cpython-${py_abi}-${py_triplet}.so"
    echo "Using eLxr extension suffix override: ${pyneat_ext_suffix}"
    # In eLxr cross-builds, PEP517 isolation may pull target-arch build tools
    # (notably ninja), which are not executable on the host container.
    # Build without isolation and force Makefiles to keep host tools executable.
    _PYTHON_HOST_PLATFORM="${ELXR_WHEEL_HOST_PLATFORM}" \
      CMAKE_ARGS="-DPYNEAT_EXT_SUFFIX=${pyneat_ext_suffix}" \
      CMAKE_GENERATOR="Unix Makefiles" \
      CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" SIMANEAT_BUILD_PYTHON=ON \
      "${wheel_python}" -m build --wheel --outdir dist --no-isolation
  else
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" SIMANEAT_BUILD_PYTHON=ON \
      "${wheel_python}" -m build --wheel --outdir dist
  fi
  echo "Built wheel(s):"
  ls -lh dist/*.whl
}

run_install_sanity_check() {
  echo
  echo "Running install sanity check..."
  local install_test_dir="/tmp/sima-neat-install-test"
  rm -rf "${install_test_dir}"

  cmake --install "${BUILD_DIR}" --prefix "${install_test_dir}"

  echo "Installed files:"
  find "${install_test_dir}" | sed 's|^|  |'

  # Ensure core archive is present in the install tree
  if [[ ! -f "${install_test_dir}/lib/libsima_neat.a" ]]; then
    echo
    echo "ERROR: libsima_neat.a missing from install tree."
    echo "Refusing to package an incomplete core .deb."
    exit 1
  fi
}

build_deb_if_requested() {
  # Package only in Linux full-build mode unless explicitly disabled.
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo
    echo "Skipping DEB packaging on macOS."
  elif [[ "${SKIP_DIST}" == "ON" ]]; then
    echo
    echo "Skipping DEB packaging (--no-dist)."
  elif [[ "${BUILD_ALL}" != "ON" ]]; then
    echo
    echo "Skipping DEB packaging (requires --all)."
  else
    echo
    echo "Building core DEB package..."
    # Remove stale debs so it's obvious what was generated
    rm -f ./*.deb
    cpack --config "${BUILD_DIR}/CPackConfig.cmake" -D CPACK_COMPONENTS_ALL=core
  fi
}

build_extras_archive_if_requested() {
  # Package extras as a relocatable tarball for user-chosen install prefixes.
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo
    echo "Skipping extras tarball packaging on macOS."
  elif [[ "${SKIP_DIST}" == "ON" ]]; then
    echo
    echo "Skipping extras tarball packaging (--no-dist)."
  elif [[ "${BUILD_ALL}" != "ON" ]]; then
    echo
    echo "Skipping extras tarball packaging (requires --all)."
  else
    echo
    echo "Building extras tarball..."
    local stage_root="./_work/extras-stage"
    local install_prefix="${stage_root}/prefix"
    local package_version="unknown"
    local core_deb
    local archive_name

    rm -rf "${stage_root}"
    mkdir -p "${install_prefix}"
    cmake --install "${BUILD_DIR}" --component extras --prefix "${install_prefix}"

    if [[ ! -d "${install_prefix}/lib/sima-neat" ]] && [[ ! -d "${install_prefix}/share/sima-neat" ]]; then
      echo "ERROR: extras install tree is empty under ${install_prefix}." >&2
      exit 1
    fi

    # Keep extras version aligned with the generated core .deb version.
    core_deb="$(ls -1 ./*-Linux-core.deb 2>/dev/null | head -n1 || true)"
    if [[ -n "${core_deb}" ]]; then
      local core_base
      core_base="$(basename "${core_deb}")"
      if [[ "${core_base}" =~ ^sima-neat-(.+)-Linux-core\.deb$ ]]; then
        package_version="${BASH_REMATCH[1]}"
      fi
    fi

    if [[ "${package_version}" == "unknown" ]] && [[ -f "${BUILD_DIR}/CPackConfig.cmake" ]]; then
      package_version="$(sed -n 's/^[[:space:]]*set(CPACK_PACKAGE_VERSION[[:space:]]*\"\\([^\"]*\\)\").*$/\\1/p' "${BUILD_DIR}/CPackConfig.cmake" | head -n1)"
      [[ -z "${package_version}" ]] && package_version="unknown"
    fi

    archive_name="sima-neat-${package_version}-Linux-extras.tar.gz"
    rm -f "./${archive_name}"
    tar -C "${install_prefix}" -czf "./${archive_name}" .
    echo "Built extras archive: ${archive_name}"
  fi
}

print_artifact_summary() {
  # Final artifact listing for CI logs and local developer visibility.
  echo
  echo "========================================"
  echo " Build completed successfully"
  echo "========================================"
  if compgen -G "./*.deb" >/dev/null 2>&1; then
    echo "DEB artifacts:"
    ls -lh ./*.deb
  fi
  if compgen -G "./*extras.tar.gz" >/dev/null 2>&1; then
    echo "Extras tarball artifacts:"
    ls -lh ./*extras.tar.gz
  fi
  if [[ "${BUILD_PYTHON}" == "ON" ]] && compgen -G "dist/*.whl" >/dev/null 2>&1; then
    echo "Python wheel artifacts:"
    ls -lh dist/*.whl
  fi
}

main() {
  # High-level pipeline:
  # parse -> bootstrap deps -> sync internals -> configure/build -> package -> summary
  parse_args "$@"
  maybe_reexec_from_shadow "$@"
  cd "${SCRIPT_DIR}"
  detect_elxr_sdk
  select_system_deps
  install_system_deps
  ensure_node20_for_docs
  activate_elxr_build_env_if_needed

  if [[ "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
    echo
    echo "System dependencies installed. Exiting due to --install-deps-only."
    exit 0
  fi

  if [[ "${OS_NAME}" != "Darwin" && "${INSTALL_NEAT_INTERNALS}" == "ON" ]]; then
    ensure_neat_internals
  fi

  detect_build_jobs
  print_build_config
  clean_build_dir_if_requested
  configure_cmake
  build_docs_only_if_requested
  build_targets
  copy_test_images
  build_docs_if_requested
  build_python_wheel_if_requested
  run_install_sanity_check
  build_deb_if_requested
  build_extras_archive_if_requested
  print_artifact_summary
}

main "$@"

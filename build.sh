#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BUILD_DIR=build
BUILD_TYPE=Release
OS_NAME="$(uname -s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIGINAL_SOURCE_DIR="${SIMANEAT_ORIGINAL_SOURCE_DIR:-${SCRIPT_DIR}}"
ORIGINAL_WORKSPACE_ROOT="${SIMANEAT_ORIGINAL_WORKSPACE_ROOT:-${WORKSPACE_ROOT}}"
SHADOW_BUILD_MODE="${SIMANEAT_SHADOW_BUILD:-auto}"
SHADOW_BUILD_ACTIVE="${SIMANEAT_SHADOW_BUILD_ACTIVE:-OFF}"
SOURCE_FS_TYPE=""
cd "${REPO_ROOT}"

# Defaults
BUILD_SAMPLES=OFF
BUILD_TESTS=OFF
BUILD_TUTORIALS=OFF
BUILD_DOCS=OFF
BUILD_ALL=OFF
DO_CLEAN=OFF
DOCS_ONLY=OFF
INSTALL_NODE=ON
SKIP_DOCS=OFF
INSTALL_DEPS_ONLY=OFF
INSTALL_AFTER_BUILD=OFF
SKIP_DIST=OFF
BUILD_PYTHON=OFF
BUILD_FUZZ=OFF
BUILD_SANITIZER_MODE=""
SIMA_ENABLE_ASAN=OFF
SIMA_ENABLE_UBSAN=OFF
SIMA_ENABLE_TSAN=OFF
SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS=OFF
INSTALL_NEAT_INTERNALS=OFF
STRICT_WARNINGS="${SIMANEAT_STRICT_WARNINGS:-OFF}"
NEAT_INTERNALS_MANIFEST="${NEAT_INTERNALS_MANIFEST:-deps/manifest.json}"
NEAT_INTERNALS_BASE_URL="${NEAT_INTERNALS_BASE_URL:-https://artifacts.sima-neat.com/internals}"
NEAT_INTERNALS_DIR="${NEAT_INTERNALS_DIR:-deps}"
NEAT_INTERNALS_PLUGIN_DIR="${NEAT_INTERNALS_DIR}/gst-plugins"
NEAT_INTERNALS_DEB_DIR="${NEAT_INTERNALS_DEB_DIR:-${NEAT_INTERNALS_DIR}/debs}"
NEAT_INTERNALS_BASIC_AUTH="${NEAT_INTERNALS_BASIC_AUTH:-}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
ELXR_INIT_SCRIPT="${ELXR_INIT_SCRIPT:-/opt/bin/simaai-init-build-env}"
ELXR_MACHINE="${ELXR_MACHINE:-modalix}"
ELXR_SDK=OFF
ELXR_SDK_VERSION=""
ELXR_VERSION=""
ELXR_WHEEL_HOST_PLATFORM="${ELXR_WHEEL_HOST_PLATFORM:-}"
DEVKIT_DEPLOY_USER="${DEVKIT_DEPLOY_USER:-sima}"

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
  --all          Build library + tests + tutorials + Python wheel
  --python       Build Python bindings (pyneat) in addition to selected targets
  --fuzz         Build fuzz-enabled package artifacts (core + extras + wheel)
  --asan-ubsan   Enable ASan+UBSan instrumentation for this build
  --tsan         Enable TSan instrumentation for this build
  --install-neat-internals, --install-deps
                 Download/install deps artifacts before build
  --doc          Build only docs
  --install      After build/package, install artifacts into the current environment.
                 In paired eLxr SDK mode, also deploy/install on the paired DevKit.
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
  ./build.sh --fuzz
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
        BUILD_TESTS=ON
        BUILD_TUTORIALS=ON
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
      --install)
        INSTALL_AFTER_BUILD=ON
        BUILD_ALL=ON
        BUILD_TESTS=ON
        BUILD_TUTORIALS=ON
        BUILD_DOCS=ON
        BUILD_PYTHON=ON
        INSTALL_NEAT_INTERNALS=ON
        shift
        ;;
      --python)
        BUILD_PYTHON=ON
        shift
        ;;
      --fuzz)
        BUILD_SAMPLES=OFF
        BUILD_TESTS=ON
        BUILD_TUTORIALS=ON
        BUILD_DOCS=OFF
        BUILD_PYTHON=ON
        BUILD_ALL=ON
        BUILD_FUZZ=ON
        INSTALL_NEAT_INTERNALS=ON
        shift
        ;;
      --asan-ubsan)
        if [[ "${BUILD_SANITIZER_MODE}" == "tsan" ]]; then
          echo "ERROR: --asan-ubsan and --tsan are mutually exclusive." >&2
          exit 1
        fi
        BUILD_SANITIZER_MODE="asan-ubsan"
        SIMA_ENABLE_ASAN=ON
        SIMA_ENABLE_UBSAN=ON
        SIMA_ENABLE_TSAN=OFF
        BUILD_SAMPLES=OFF
        BUILD_TUTORIALS=OFF
        shift
        ;;
      --tsan)
        if [[ "${BUILD_SANITIZER_MODE}" == "asan-ubsan" ]]; then
          echo "ERROR: --asan-ubsan and --tsan are mutually exclusive." >&2
          exit 1
        fi
        BUILD_SANITIZER_MODE="tsan"
        SIMA_ENABLE_ASAN=OFF
        SIMA_ENABLE_UBSAN=OFF
        SIMA_ENABLE_TSAN=ON
        BUILD_SAMPLES=OFF
        BUILD_TUTORIALS=OFF
        shift
        ;;
      --install-neat-internals|--install-deps)
        INSTALL_NEAT_INTERNALS=ON
        shift
        ;;
      --example)
        echo "ERROR: Core no longer builds examples. Use the separate apps repository for curated examples." >&2
        exit 1
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
        if [[ "${INSTALL_AFTER_BUILD}" == "ON" ]]; then
          echo "WARNING: --no-dist ignored because --install requires distribution packages." >&2
          SKIP_DIST=OFF
        else
          SKIP_DIST=ON
        fi
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

validate_build_mode_combinations() {
  if [[ "${INSTALL_AFTER_BUILD}" == "ON" ]]; then
    SKIP_DIST=OFF
  fi

  if [[ "${BUILD_FUZZ}" == "ON" && -n "${BUILD_SANITIZER_MODE}" ]]; then
    echo "ERROR: --fuzz cannot be combined with sanitizer modes (--asan-ubsan/--tsan)." >&2
    exit 1
  fi
}

apply_sanitizer_build_profile() {
  if [[ -z "${BUILD_SANITIZER_MODE}" ]]; then
    return 0
  fi

  # Sanitizer lanes are test-focused; skip samples/examples to avoid
  # optional UI/OpenGL dependencies in cross-build environments.
  BUILD_SAMPLES=OFF
  BUILD_TUTORIALS=OFF
  # Keep sanitizer extras payloads small by shipping only gate test binaries.
  SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS=ON
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
  # 3) password-based sudo via env in non-interactive sessions
  # 4) interactive sudo only in TTY sessions
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
      sudo -n "$@"
      return 0
    fi

    local sudo_pw="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
    if [[ -n "${sudo_pw}" ]]; then
      if printf '%s\n' "${sudo_pw}" | sudo -S -v >/dev/null 2>&1; then
        printf '%s\n' "${sudo_pw}" | sudo -S "$@"
        return $?
      fi
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
  local deb_cache_dir="$3"
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

  mkdir -p "${deb_cache_dir}"
  rm -f "${deb_cache_dir}"/*.deb
  local cached_deb
  for cached_deb in "${deb_files[@]}"; do
    cp -f "${cached_deb}" "${deb_cache_dir}/$(basename "${cached_deb}")"
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
    if ! command -v dpkg >/dev/null 2>&1; then
      echo "ERROR: dpkg is required to install neat-internals .deb packages." >&2
      exit 1
    fi
    # Install packages into host system locations so runtime can resolve them globally.
    echo "Installing neat-internals .deb packages into host system..."
    if command -v apt >/dev/null 2>&1; then
      mapfile -t deb_abs_files < <(for deb in "${deb_files[@]}"; do realpath "${deb}"; done)
      # Use apt for local .deb install so dependency resolution happens automatically in CI.
      if ! run_privileged apt install -y --allow-downgrades -o Dpkg::Options::=--force-overwrite "${deb_abs_files[@]}"; then
        echo "ERROR: Failed to install neat-internals .deb packages via apt." >&2
        exit 1
      fi
    else
      if ! run_privileged dpkg -i "${deb_files[@]}"; then
        echo "ERROR: Failed to install deps .deb packages." >&2
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
        echo "Installed package: ${pkg_name} (${pkg_version})"
      else
        echo "Installed package: ${pkg_name}"
      fi
    else
      echo "Installed package from ${deb_name}.deb (unable to resolve package metadata)."
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
    else
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

  local internals_ref
  # Manifest drives which internals artifact to fetch.
  internals_ref="$(extract_json_string "internals" "${NEAT_INTERNALS_MANIFEST}")"
  if [[ -z "${internals_ref}" ]]; then
    echo "ERROR: ${NEAT_INTERNALS_MANIFEST} must define a non-empty internals string." >&2
    exit 1
  fi

  local marker_file="${NEAT_INTERNALS_DIR}/.internals"
  local checksum_file="${NEAT_INTERNALS_DIR}/.artifact_sha256"
  local deb_cache_dir="${NEAT_INTERNALS_DEB_DIR}"
  local archive_name="sima-neat-internals-${internals_ref}.tar.gz"
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
    if [[ "${current_tag}" == "${internals_ref}" ]] &&
       [[ "${cached_sha}" == "${server_sha}" ]] &&
       [[ -f "${NEAT_INTERNALS_PLUGIN_DIR}/libgstneatdecoder.so" ]] &&
       compgen -G "${deb_cache_dir}/neat-*.deb" >/dev/null 2>&1; then
      echo "Using cached neat-internals plugins/debs (${internals_ref}, sha256=${server_sha})."
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
  if ! collect_plugin_files_from_debs "${extract_dir}" "${plugins_list_file}" "${deb_cache_dir}"; then
    collect_plugin_files_from_legacy_tar "${extract_dir}" "${plugins_list_file}"
  fi

  copy_plugins_to_neat_internals "${plugins_list_file}"

  printf '%s\n' "${internals_ref}" > "${marker_file}"
  printf '%s\n' "${server_sha}" > "${checksum_file}"
  rm -rf "${tmp_dir}"
}

collect_install_artifact_files() {
  local -n out_files_ref="$1"
  out_files_ref=()

  local -A seen_basenames=()
  local file
  local basename_file

  for file in ./*.deb; do
    [[ -e "${file}" ]] || continue
    basename_file="$(basename "${file}")"
    [[ -n "${seen_basenames[${basename_file}]:-}" ]] && continue
    seen_basenames["${basename_file}"]=1
    out_files_ref+=("${file}")
  done

  for file in "${NEAT_INTERNALS_DEB_DIR}"/neat-*.deb; do
    [[ -e "${file}" ]] || continue
    basename_file="$(basename "${file}")"
    [[ -n "${seen_basenames[${basename_file}]:-}" ]] && continue
    seen_basenames["${basename_file}"]=1
    out_files_ref+=("${file}")
  done

  for file in dist/*.whl; do
    [[ -e "${file}" ]] || continue
    basename_file="$(basename "${file}")"
    [[ -n "${seen_basenames[${basename_file}]:-}" ]] && continue
    seen_basenames["${basename_file}"]=1
    out_files_ref+=("${file}")
  done
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

configure_fuzz_toolchain_if_needed() {
  if [[ "${BUILD_FUZZ}" != "ON" ]]; then
    return 0
  fi

  if [[ "${ELXR_SDK}" == "ON" ]]; then
    echo "ERROR: --fuzz is not supported in eLxr SDK environment." >&2
    echo "Run fuzz builds on Modalix ARM64 runners/devkits where libFuzzer targets execute natively." >&2
    exit 1
  fi

  if [[ -n "${CC:-}" && -n "${CXX:-}" ]]; then
    echo "Using user-provided fuzz toolchain: CC=${CC} CXX=${CXX}"
    return 0
  fi

  local clang_bin=""
  local clangxx_bin=""

  for c in clang clang-18 clang-17 clang-16; do
    if command -v "${c}" >/dev/null 2>&1; then
      clang_bin="${c}"
      break
    fi
  done
  for cxx in clang++ clang++-18 clang++-17 clang++-16; do
    if command -v "${cxx}" >/dev/null 2>&1; then
      clangxx_bin="${cxx}"
      break
    fi
  done

  if [[ -z "${clang_bin}" || -z "${clangxx_bin}" ]]; then
    echo "ERROR: --fuzz requires Clang/libFuzzer, but clang/clang++ were not found." >&2
    echo "Set CC/CXX explicitly or install clang and clang++ on this runner." >&2
    exit 1
  fi

  export CC="${CC:-${clang_bin}}"
  export CXX="${CXX:-${clangxx_bin}}"
  echo "Auto-selected fuzz toolchain: CC=${CC} CXX=${CXX}"
}

print_build_config() {
  # Emit resolved configuration so build-mode decisions are explicit.
  echo "========================================"
  echo " SiMa Neat build configuration"
  echo "========================================"
  echo "Build type     : ${BUILD_TYPE}"
  echo "Build samples  : ${BUILD_SAMPLES}"
  echo "Build tests    : ${BUILD_TESTS}"
  echo "Build tutorials: ${BUILD_TUTORIALS}"
  echo "Build docs     : ${BUILD_DOCS}"
  echo "Build python   : ${BUILD_PYTHON}"
  echo "Build all      : ${BUILD_ALL}"
  echo "Build fuzz     : ${BUILD_FUZZ}"
  echo "Sanitizer mode : ${BUILD_SANITIZER_MODE:-none}"
  echo "Gate-only extras: ${SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS}"
  echo "Neat internals : ${INSTALL_NEAT_INTERNALS}"
  echo "Skip dist      : ${SKIP_DIST}"
  echo "Install after  : ${INSTALL_AFTER_BUILD}"
  echo "Skip docs      : ${SKIP_DOCS}"
  echo "Clean build    : ${DO_CLEAN}"
  echo "Strict warns   : ${STRICT_WARNINGS}"
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
  local -a cmake_args=(
    -S . -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DSIMANEAT_BUILD_SAMPLES="${BUILD_SAMPLES}"
    -DSIMANEAT_BUILD_TESTS="${BUILD_TESTS}"
    -DSIMANEAT_BUILD_TUTORIALS="${BUILD_TUTORIALS}"
    -DSIMANEAT_BUILD_PYTHON="${BUILD_PYTHON}"
    -DSIMANEAT_STRICT_WARNINGS="${STRICT_WARNINGS}"
    -DSIMA_ENABLE_ASAN="${SIMA_ENABLE_ASAN}"
    -DSIMA_ENABLE_UBSAN="${SIMA_ENABLE_UBSAN}"
    -DSIMA_ENABLE_TSAN="${SIMA_ENABLE_TSAN}"
    -DSIMANEAT_SANITIZER_GATE_ONLY_EXTRAS="${SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS}"
    -DFUZZING="${BUILD_FUZZ}"
  )

  if [[ "${ELXR_SDK}" == "ON" && -n "${SYSROOT:-}" ]]; then
    local -a pkgconfig_dirs=(
      "${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig"
      "${SYSROOT}/usr/lib/pkgconfig"
      "${SYSROOT}/usr/share/pkgconfig"
    )
    local pkg_config_executable
    pkg_config_executable="${PKG_CONFIG_EXECUTABLE:-$(command -v pkg-config)}"
    export PKG_CONFIG_SYSROOT_DIR="${PKG_CONFIG_SYSROOT_DIR:-${SYSROOT}}"
    export PKG_CONFIG_LIBDIR="${PKG_CONFIG_LIBDIR:-$(IFS=:; echo "${pkgconfig_dirs[*]}")}"
    export PKG_CONFIG_EXECUTABLE="${pkg_config_executable}"

    cmake_args+=(
      -DCMAKE_SYSROOT="${SYSROOT}"
      -DCMAKE_FIND_ROOT_PATH="${SYSROOT}"
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
      -DCMAKE_PREFIX_PATH="${SYSROOT}/usr;${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/lib/cmake"
      -DPKG_CONFIG_EXECUTABLE="${pkg_config_executable}"
    )
  fi

  cmake "${cmake_args[@]}"
}

build_docs_site() {
  # Shared docs pipeline used by both --doc and --all/--no-doc flows.
  cd "${REPO_ROOT}"
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
  expanded_docs_dir="${REPO_ROOT}/${BUILD_DIR}/docs-expanded"
  python3 tools/expand_code_tabs.py --src "${REPO_ROOT}/docs" --dst "${expanded_docs_dir}"
  echo
  echo "Resetting website build caches..."
  rm -rf "${REPO_ROOT}/website/node_modules/.cache"
  rm -rf "${REPO_ROOT}/website/.docusaurus"
  rm -rf "${REPO_ROOT}/website/build"
  if [[ "${INSTALL_NODE}" == "ON" ]]; then
    echo
    echo "Installing website dependencies..."
    npm --prefix "${REPO_ROOT}/website" ci --no-audit --no-fund
  fi
  echo
  echo "Building Docusaurus site..."
  DOCS_PATH="${expanded_docs_dir}" npm --prefix "${REPO_ROOT}/website" run build
  if [[ "${DOCS_STRICT_LINKS:-0}" == "1" ]]; then
    echo
    echo "Checking rendered docs links..."
    bash "${REPO_ROOT}/scripts/ci/check_docs_links.sh"
  fi
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
  if [[ "${BUILD_SAMPLES}" == "OFF" && "${BUILD_TESTS}" == "OFF" && "${BUILD_DOCS}" == "OFF" ]]; then
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

generate_platform_version_artifacts() {
  # Keep C++ and Python consumers aligned with deps/manifest.json.
  python3 tools/generate_platform_version_artifacts.py --repo-root "${REPO_ROOT}"
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
    echo "Building DEB packages (core + extras)..."
    # Remove stale debs so it's obvious what was generated
    rm -f ./*.deb
    cpack --config "${BUILD_DIR}/CPackConfig.cmake"
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
    local stage_root
    local keep_stage_root=OFF
    if [[ -n "${SIMANEAT_EXTRAS_STAGE_ROOT:-}" ]]; then
      stage_root="${SIMANEAT_EXTRAS_STAGE_ROOT%/}/extras-stage"
      keep_stage_root=ON
    else
      stage_root="$(mktemp -d /tmp/sima-neat-extras-stage.XXXXXX)"
    fi
    local install_prefix="${stage_root}/prefix"
    local package_version="unknown"
    local core_deb
    local archive_name

    (
      trap 'if [[ "${keep_stage_root}" != "ON" ]]; then rm -rf "${stage_root}"; fi' EXIT

      rm -rf "${stage_root}"
      mkdir -p "${install_prefix}"
      cmake --install "${BUILD_DIR}" --component extras --prefix "${install_prefix}"

      # Include source-side CMake manifests so downstream jobs can inspect and
      # reason about test/tutorial layout from extras alone.
      mkdir -p \
        "${install_prefix}/share/sima-neat/tests" \
        "${install_prefix}/share/sima-neat/tutorials"
      cp -f "tests/CMakeLists.txt" "${install_prefix}/share/sima-neat/tests/CMakeLists.txt"
      cp -f "tutorials/CMakeLists.txt" "${install_prefix}/share/sima-neat/tutorials/CMakeLists.txt"

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
    )

    if [[ "${keep_stage_root}" == "ON" ]]; then
      echo "Preserved extras staging directory: ${stage_root}"
    fi
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

should_deploy_to_devkit() {
  [[ "${INSTALL_AFTER_BUILD}" == "ON" ]] || return 1
  [[ "${ELXR_SDK}" == "ON" ]] || return 1
  [[ -n "${DEVKIT_SYNC_DEVKIT_IP:-}" ]] || return 1
  [[ "${OS_NAME}" != "Darwin" ]] || return 1

  compgen -G "./*.deb" >/dev/null 2>&1 || [[ "${BUILD_PYTHON}" == "ON" && -n "$(compgen -G 'dist/*.whl' || true)" ]] || return 1
}

run_devkit_ssh() {
  if command -v ssh >/dev/null 2>&1; then
    ssh "$@"
    return 0
  fi
  if command -v sima-ssh >/dev/null 2>&1; then
    sima-ssh "$@"
    return 0
  fi
  echo "ERROR: ssh/sima-ssh is required for DevKit deployment in paired SDK mode." >&2
  exit 1
}

run_devkit_scp() {
  if command -v scp >/dev/null 2>&1; then
    scp "$@"
    return 0
  fi
  if command -v sima-scp >/dev/null 2>&1; then
    sima-scp "$@"
    return 0
  fi
  echo "ERROR: scp/sima-scp is required for DevKit deployment in paired SDK mode." >&2
  exit 1
}

install_artifacts_into_current_environment_if_requested() {
  [[ "${INSTALL_AFTER_BUILD}" == "ON" ]] || return 0

  if [[ "${SKIP_DIST}" == "ON" ]]; then
    echo "ERROR: --install cannot be used with --no-dist." >&2
    exit 1
  fi

  local -a staged_files=()
  collect_install_artifact_files staged_files
  local artifact_stage
  artifact_stage="$(mktemp -d "/tmp/sima-neat-install.XXXXXX")"

  local file
  for file in "${staged_files[@]}"; do
    cp "${file}" "${artifact_stage}/"
  done

  if [[ "${#staged_files[@]}" -eq 0 ]]; then
    rm -rf "${artifact_stage}"
    echo "ERROR: --install requested, but no .deb/.whl artifacts were generated." >&2
    exit 1
  fi

  echo
  echo "Installing generated artifacts into the current environment..."
  (
    cd "${artifact_stage}"
    NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON bash "${REPO_ROOT}/tools/install_neat_framework.sh"
  )
  rm -rf "${artifact_stage}"
}

deploy_artifacts_to_devkit_if_requested() {
  if ! should_deploy_to_devkit; then
    return 0
  fi

  echo
  echo "SDK/DevKit deploy context detected for --install:"
  echo "  ELXR_SDK=${ELXR_SDK}"
  echo "  SYSROOT=${SYSROOT:-<unset>}"
  echo "  DEVKIT_SYNC_DEVKIT_IP=${DEVKIT_SYNC_DEVKIT_IP}"

  local ssh_target="${DEVKIT_DEPLOY_USER}@${DEVKIT_SYNC_DEVKIT_IP}"
  if ! run_devkit_ssh -o ConnectTimeout=5 "${ssh_target}" "true" >/dev/null 2>&1; then
    echo "ERROR: Paired DevKit ${ssh_target} is not reachable over SSH." >&2
    echo "Fix network/pairing, or unset DEVKIT_SYNC_DEVKIT_IP to run SDK-only installs." >&2
    exit 1
  fi

  local remote_dir="/tmp/sima-neat-build-$(date +%Y%m%d-%H%M%S)"
  local -a deploy_files=()
  collect_install_artifact_files deploy_files
  local installer_path="${REPO_ROOT}/tools/install_neat_framework.sh"
  local installer_basename
  installer_basename="$(basename "${installer_path}")"

  if [[ "${#deploy_files[@]}" -eq 0 ]]; then
    echo "No deployable .deb or .whl artifacts found."
    return 0
  fi
  deploy_files+=("${installer_path}")

  echo
  echo "Deploying artifacts to ${ssh_target}:${remote_dir}"
  run_devkit_ssh "${ssh_target}" "mkdir -p '${remote_dir}'"
  run_devkit_scp "${deploy_files[@]}" "${ssh_target}:${remote_dir}/"

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

  run_devkit_ssh -t "${ssh_target}" "bash -lc $(printf '%q' "${remote_install_cmd}")"

  echo
  echo "DevKit deployment completed successfully."
  echo "Remote artifact directory cleaned: ${remote_dir}"
}

main() {
  # High-level pipeline:
  # parse -> bootstrap deps -> sync internals -> configure/build -> package -> summary
  parse_args "$@"
  maybe_reexec_from_shadow "$@"
  validate_build_mode_combinations
  apply_sanitizer_build_profile
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
  configure_fuzz_toolchain_if_needed
  generate_platform_version_artifacts
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
  install_artifacts_into_current_environment_if_requested
  deploy_artifacts_to_devkit_if_requested
}

main "$@"

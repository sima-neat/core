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
ALLOW_FUZZ_IN_NEAT_SDK="${SIMANEAT_ALLOW_FUZZ_IN_NEAT_SDK:-OFF}"
BUILD_SANITIZER_MODE=""
SIMA_ENABLE_ASAN=OFF
SIMA_ENABLE_UBSAN=OFF
SIMA_ENABLE_TSAN=OFF
SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS=OFF
INSTALL_NEAT_INTERNALS=OFF
INSTALL_NEAT_LLIMA=OFF
STRICT_WARNINGS="${SIMANEAT_STRICT_WARNINGS:-OFF}"
NEAT_DEPS_MANIFEST="${NEAT_DEPS_MANIFEST:-${NEAT_INTERNALS_MANIFEST:-deps/manifest.json}}"
NEAT_VULCAN_ENV="${NEAT_VULCAN_ENV:-production}"
NEAT_VULCAN_BASE_URL="${NEAT_VULCAN_BASE_URL:-}"
NEAT_INTERNALS_VULCAN_REPOSITORY="${NEAT_INTERNALS_VULCAN_REPOSITORY:-internals}"
NEAT_LLIMA_VULCAN_REPOSITORY="${NEAT_LLIMA_VULCAN_REPOSITORY:-llima}"
NEAT_INTERNALS_RESOLVED_MANIFEST="${NEAT_INTERNALS_RESOLVED_MANIFEST:-${BUILD_DIR}/resolved_manifest.json}"
NEAT_PACKAGE_BUILDINFO_JSON="${NEAT_PACKAGE_BUILDINFO_JSON:-${BUILD_DIR}/buildinfo.json}"
NEAT_INTERNALS_DIR="${NEAT_INTERNALS_DIR:-deps}"
NEAT_DEP_HEADERS_DIR="${REPO_ROOT}/deps/headers"
NEAT_INTERNALS_PLUGIN_DIR="${NEAT_INTERNALS_DIR}/gst-plugins"
NEAT_INTERNALS_DEB_DIR="${NEAT_INTERNALS_DEB_DIR:-${NEAT_INTERNALS_DIR}/debs}"
NEAT_INTERNALS_RESOLVED_REF=""
NEAT_INTERNALS_REQUESTED_REF=""
NEAT_INTERNALS_SNAP_POLICY=OFF
NEAT_INTERNALS_SNAP_TAG_POLICY=OFF
NEAT_LLIMA_DEB_DIR="${NEAT_LLIMA_DEB_DIR:-${NEAT_INTERNALS_DIR}/llima-debs}"
NEAT_LLIMA_RESOLVED_REF=""
NEAT_LLIMA_REQUESTED_REF=""
NEAT_LLIMA_SNAP_POLICY=OFF
NEAT_LLIMA_SNAP_TAG_POLICY=OFF
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
ELXR_INIT_SCRIPT="${ELXR_INIT_SCRIPT:-/opt/bin/simaai-init-build-env}"
ELXR_MACHINE="${ELXR_MACHINE:-modalix}"
ELXR_SDK=OFF
ELXR_SDK_VERSION=""
ELXR_VERSION=""
ELXR_WHEEL_HOST_PLATFORM="${ELXR_WHEEL_HOST_PLATFORM:-}"
ELXR_HOST_PYTHON_EXECUTABLE=""
ELXR_TARGET_PYTHON_VERSION=""
ELXR_TARGET_PYTHON_INCLUDE_DIR=""
DEVKIT_DEPLOY_USER="${DEVKIT_DEPLOY_USER:-sima}"
NEAT_PACKAGE_NAME="${NEAT_PACKAGE_NAME:-sima-neat}"
NEAT_PACKAGE_DESCRIPTION="${NEAT_PACKAGE_DESCRIPTION:-SiMa.ai Neural Edge Acceleration Toolkit}"
NEAT_PACKAGE_INSTALL_SCRIPT="${NEAT_PACKAGE_INSTALL_SCRIPT:-install_neat_framework.sh}"
NEAT_INSTALL_MANIFEST="${NEAT_INSTALL_MANIFEST:-neat-install-manifest.txt}"
NEAT_EXTRAS_SELECTABLE_NAME="${NEAT_EXTRAS_SELECTABLE_NAME:-SiMa NEAT extras (tutorials/tests)}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
SIMANEAT_BOOTSTRAP_SIMA_CLI="${SIMANEAT_BOOTSTRAP_SIMA_CLI:-auto}"

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

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_remote_fs_type() {
  local fs_type
  fs_type="$(lowercase "$1")"
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

  case "$(lowercase "${SHADOW_BUILD_MODE}")" in
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
  --fuzz         Build fuzz-enabled package artifacts (core + dev + extras + wheel)
  --asan-ubsan   Enable ASan+UBSan instrumentation for this build
  --tsan         Enable TSan instrumentation for this build
  --install-neat-internals, --install-deps
                 Download/install internals + LLiMa deps artifacts before build
  --doc          Build only docs
  --install      After build/package, install artifacts into the current environment.
                 In paired eLxr SDK mode, also deploy/install on the paired DevKit.
  --no-dist      Skip DEB packaging
  --clean        Remove build directory before building
  --no-doc       Skip documentation build (even with --all)
  --no-node      Skip Node.js install (docs build will not work)
  --install-deps-only
                 Install system dependencies and dependency headers, then exit
  -h, --help     Show this help

Environment:
  SIMANEAT_SHADOW_BUILD=auto|off|force
                 Auto-stage a local shadow workspace for builds from remote filesystems.
  SIMANEAT_SHADOW_ROOT=/tmp/sima-neat-shadow-<id>
                 Override where the shadow workspace is created.
  NEAT_DEPS_MANIFEST=deps/manifest.json
                 Manifest used to resolve Vulcan dependency artifacts.
  NEAT_VULCAN_ENV=production
                 Vulcan environment used for dependency artifact installs.
  SIMA_CLI_BIN=sima-cli
                 sima-cli executable used for Vulcan installs and metadata generation.

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
        BUILD_TESTS=OFF
        shift
        ;;
      --all)
        BUILD_TESTS=ON
        BUILD_TUTORIALS=ON
        BUILD_DOCS=ON
        BUILD_PYTHON=ON
        INSTALL_NEAT_INTERNALS=ON
        INSTALL_NEAT_LLIMA=ON
        BUILD_ALL=ON
        shift
        ;;
      --doc)
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
        INSTALL_NEAT_LLIMA=ON
        shift
        ;;
      --python)
        BUILD_PYTHON=ON
        shift
        ;;
      --fuzz)
        BUILD_TESTS=ON
        BUILD_TUTORIALS=ON
        BUILD_DOCS=OFF
        BUILD_PYTHON=ON
        BUILD_ALL=ON
        BUILD_FUZZ=ON
        INSTALL_NEAT_INTERNALS=ON
        INSTALL_NEAT_LLIMA=ON
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
        BUILD_TUTORIALS=OFF
        shift
        ;;
      --install-neat-internals|--install-deps)
        INSTALL_NEAT_INTERNALS=ON
        INSTALL_NEAT_LLIMA=ON
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

  # Sanitizer lanes are test-focused; skip tutorials to keep payloads small.
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

detect_elxr_host_python() {
  if [[ "${ELXR_SDK}" != "ON" ]]; then
    return 0
  fi

  local candidate="${SIMANEAT_HOST_PYTHON:-${Python3_EXECUTABLE:-${PYTHON3_EXECUTABLE:-}}}"
  if [[ -z "${candidate}" ]]; then
    candidate="$(command -v python3 || command -v python || true)"
  fi

  if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
    echo "ERROR: eLxr SDK cross-build requires an executable host python3." >&2
    echo "Set SIMANEAT_HOST_PYTHON to the host Python interpreter path." >&2
    exit 1
  fi

  if ! "${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' >/dev/null 2>&1; then
    echo "ERROR: selected Python cannot run in this SDK container: ${candidate}" >&2
    echo "Set SIMANEAT_HOST_PYTHON to a host-runnable Python interpreter." >&2
    exit 1
  fi

  ELXR_HOST_PYTHON_EXECUTABLE="${candidate}"
}

detect_elxr_target_python() {
  if [[ "${ELXR_SDK}" != "ON" ]]; then
    return 0
  fi

  local include_dir
  for include_dir in "${SYSROOT:-}/usr/include"/python3.*; do
    if [[ -f "${include_dir}/Python.h" ]]; then
      ELXR_TARGET_PYTHON_INCLUDE_DIR="${include_dir}"
      ELXR_TARGET_PYTHON_VERSION="${include_dir##*/python}"
      return 0
    fi
  done

  echo "ERROR: eLxr SDK cross-build requires target Python headers in the SDK sysroot." >&2
  echo "Expected ${SYSROOT:-<unset>}/usr/include/python3.x/Python.h." >&2
  exit 1
}

select_system_deps() {
  # Start from the baseline dependency lists for each OS.
  if [[ "${DOCS_ONLY}" == "ON" ]]; then
    SELECTED_SYSTEM_DEPS_LINUX=(
      build-essential
      cmake
      pkg-config
      git
      curl
      doxygen
      graphviz
    )
    SELECTED_SYSTEM_DEPS_MAC=(
      cmake
      pkg-config
      git
      doxygen
      graphviz
    )
  else
    SELECTED_SYSTEM_DEPS_LINUX=("${SYSTEM_DEPS_LINUX[@]}")
    SELECTED_SYSTEM_DEPS_MAC=("${SYSTEM_DEPS_MAC[@]}")
  fi

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

ensure_llima_sdk_sysroot_deps() {
  if [[ "${ELXR_SDK}" != "ON" ]]; then
    return
  fi

  local install_root="${SYSROOT:-}"
  if [[ -z "${install_root}" ]]; then
    echo "ERROR: eLxr SDK mode requires SYSROOT to be set before installing LLiMa dependencies." >&2
    exit 1
  fi

  if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "ERROR: apt-get and dpkg-deb are required to install LLiMa dependencies into the SDK sysroot." >&2
    exit 1
  fi

  local -a missing_packages=()
  if [[ ! -f "${install_root}/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor" ||
        ! -f "${install_root}/usr/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
    missing_packages+=("libeigen3-dev")
  fi
  if [[ ! -f "${install_root}/usr/include/fmt/core.h" ]]; then
    missing_packages+=("libfmt-dev:arm64")
  fi
  if [[ ! -f "${install_root}/usr/lib/aarch64-linux-gnu/libfmt.so.9.1.0" ]]; then
    missing_packages+=("libfmt9:arm64")
  fi
  if [[ ! -f "${install_root}/usr/include/spdlog/spdlog.h" ]]; then
    missing_packages+=("libspdlog-dev:arm64")
  fi
  if [[ ! -f "${install_root}/usr/lib/aarch64-linux-gnu/libspdlog.so.1.10.0" ]]; then
    missing_packages+=("libspdlog1.10:arm64")
  fi
  if [[ ! -f "${install_root}/usr/include/nlohmann/json.hpp" ]]; then
    missing_packages+=("nlohmann-json3-dev")
  fi
  if [[ ! -f "${install_root}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlicommon.pc" ||
        ! -f "${install_root}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlidec.pc" ||
        ! -f "${install_root}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlienc.pc" ]]; then
    missing_packages+=("libbrotli-dev:arm64")
  fi
  if [[ ! -f "${install_root}/usr/include/httplib.h" ]]; then
    missing_packages+=("libcpp-httplib-dev:arm64")
  fi
  if [[ ! -e "${install_root}/usr/lib/aarch64-linux-gnu/libcpp-httplib.so.0.11" ]]; then
    missing_packages+=("libcpp-httplib0.11:arm64")
  fi

  if (( ${#missing_packages[@]} == 0 )); then
    return
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sima-neat-llima-deps-XXXXXX)"

  echo "Installing LLiMa SDK sysroot dependencies:"
  printf '  %s\n' "${missing_packages[@]}"
  (
    cd "${tmp_dir}"
    apt-get download "${missing_packages[@]}"
  )

  mapfile -t downloaded_debs < <(find "${tmp_dir}" -maxdepth 1 -type f -name '*.deb' | sort)
  if (( ${#downloaded_debs[@]} == 0 )); then
    echo "ERROR: Failed to download LLiMa SDK sysroot dependencies." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  local dep_deb
  for dep_deb in "${downloaded_debs[@]}"; do
    echo "  $(basename "${dep_deb}")"
    if ! dpkg-deb -x "${dep_deb}" "${install_root}" 2>/dev/null; then
      if ! run_privileged dpkg-deb -x "${dep_deb}" "${install_root}"; then
        echo "ERROR: Failed to install $(basename "${dep_deb}") into SYSROOT=${install_root}" >&2
        rm -rf "${tmp_dir}"
        exit 1
      fi
    fi
  done

  if [[ ! -f "${install_root}/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor" ||
        ! -f "${install_root}/usr/include/fmt/core.h" ||
        ! -f "${install_root}/usr/lib/aarch64-linux-gnu/libfmt.so.9.1.0" ||
        ! -f "${install_root}/usr/include/spdlog/spdlog.h" ||
        ! -f "${install_root}/usr/lib/aarch64-linux-gnu/libspdlog.so.1.10.0" ||
        ! -f "${install_root}/usr/include/nlohmann/json.hpp" ]]; then
      echo "ERROR: LLiMa SDK sysroot dependencies are still incomplete after install." >&2
      rm -rf "${tmp_dir}"
      exit 1
  fi

  rm -rf "${tmp_dir}"
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

manifest_dependency_spec() {
  local key="$1"
  local file="$2"
  python3 - "${key}" "${file}" <<'PY'
import json
import sys
from pathlib import Path

key = sys.argv[1]
manifest_path = Path(sys.argv[2])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if key not in data:
    raise SystemExit(f"ERROR: {manifest_path} must define '{key}'.")

value = data[key]
if isinstance(value, str):
    print("__SNAP__" if not value.strip() else value.strip())
    raise SystemExit(0)

if isinstance(value, dict):
    policy = str(value.get("policy", "")).strip().lower()
    if policy == "snap":
        print("__SNAP__")
        raise SystemExit(0)
    if policy:
        raise SystemExit(f"ERROR: unsupported {key}.policy in {manifest_path}: {policy!r}")

    spec = str(value.get("spec", "")).strip()
    branch = str(value.get("branch", value.get("ref", ""))).strip()
    if branch:
        print(f"{branch}:{spec or 'latest'}")
        raise SystemExit(0)

raise SystemExit(
    f"ERROR: {manifest_path} field '{key}' must be a string, "
    "or an object with {'policy':'snap'} or {'branch':'...', 'spec':'...'}."
)
PY
}

current_core_branch() {
  if [[ -n "${GITHUB_HEAD_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_HEAD_REF}"
    return 0
  fi
  if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 &&
     git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null
    return 0
  fi
  printf '\n'
}

current_core_tag() {
  if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 &&
     git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" describe --tags --exact-match HEAD 2>/dev/null || true
    return 0
  fi
  printf '\n'
}

resolve_neat_internals_ref() {
  NEAT_INTERNALS_SNAP_TAG_POLICY=OFF
  if [[ ! -f "${NEAT_DEPS_MANIFEST}" ]]; then
    echo "ERROR: Missing manifest: ${NEAT_DEPS_MANIFEST}" >&2
    return 1
  fi

  local manifest_spec
  if ! manifest_spec="$(manifest_dependency_spec "internals" "${NEAT_DEPS_MANIFEST}")"; then
    return 1
  fi

  local branch spec tag
  if [[ "${manifest_spec}" == "__SNAP__" ]]; then
    NEAT_INTERNALS_SNAP_POLICY=ON
    tag="$(current_core_tag)"
    if [[ -n "${tag}" ]]; then
      NEAT_INTERNALS_SNAP_TAG_POLICY=ON
      branch="${tag}"
    else
      branch="$(current_core_branch)"
      if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        echo "Could not determine current branch for internals snap; using develop." >&2
        branch="develop"
      fi
    fi
  elif [[ "${manifest_spec}" == *":"* ]]; then
    branch="${manifest_spec%%:*}"
    spec="${manifest_spec#*:}"
  else
    branch="${manifest_spec}"
  fi
  spec="${spec:-latest}"

  NEAT_INTERNALS_REQUESTED_REF="${branch}:${spec}"
}

resolve_neat_llima_ref() {
  NEAT_LLIMA_SNAP_TAG_POLICY=OFF
  if [[ ! -f "${NEAT_DEPS_MANIFEST}" ]]; then
    echo "ERROR: Missing manifest: ${NEAT_DEPS_MANIFEST}" >&2
    return 1
  fi

  local manifest_spec
  if ! manifest_spec="$(manifest_dependency_spec "llima" "${NEAT_DEPS_MANIFEST}")"; then
    return 1
  fi

  local branch spec tag
  if [[ "${manifest_spec}" == "__SNAP__" ]]; then
    NEAT_LLIMA_SNAP_POLICY=ON
    tag="$(current_core_tag)"
    if [[ -n "${tag}" ]]; then
      NEAT_LLIMA_SNAP_TAG_POLICY=ON
      branch="${tag}"
    else
      branch="$(current_core_branch)"
      if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        echo "Could not determine current branch for LLiMa snap; using develop." >&2
        branch="develop"
      fi
    fi
  elif [[ "${manifest_spec}" == *":"* ]]; then
    branch="${manifest_spec%%:*}"
    spec="${manifest_spec#*:}"
  else
    branch="${manifest_spec}"
  fi
  spec="${spec:-latest}"

  NEAT_LLIMA_REQUESTED_REF="${branch}:${spec}"
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
      local rel_path
      local sysroot_path
      local -a sysroot_paths=()
      for rel_path in usr/lib usr/libexec; do
        if [[ -d "${deb_extract_dir}/${rel_path}" ]]; then
          while IFS= read -r sysroot_path; do
            sysroot_paths+=("${sysroot_path}")
          done < <(
            find "${deb_extract_dir}/${rel_path}" -mindepth 1 -print |
              sed "s|^${deb_extract_dir}|${install_root}|"
          )
        fi
      done
      if [[ "${#sysroot_paths[@]}" -eq 0 ]]; then
        echo "  (no /usr/lib* or /usr/libexec paths reported)"
      else
        printf '  %s\n' "${sysroot_paths[@]}"
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

bootstrap_sima_cli_for_ci_if_needed() {
  case "${SIMANEAT_BOOTSTRAP_SIMA_CLI}" in
    OFF|off|0|false|FALSE|False)
      return 1
      ;;
  esac

  if [[ "${SIMANEAT_BOOTSTRAP_SIMA_CLI}" != "ON" &&
        "${SIMANEAT_BOOTSTRAP_SIMA_CLI}" != "on" &&
        "${SIMANEAT_BOOTSTRAP_SIMA_CLI}" != "1" &&
        -z "${GITHUB_ACTIONS:-}" &&
        -z "${CI:-}" ]]; then
    return 1
  fi

  echo "Installing sima-cli with Neat artifact support for CI..."
  "${SCRIPT_DIR}/scripts/ci/install_sima_cli_main.sh"

  export PATH="${HOME}/.sima-cli/.venv/bin:${PATH}"
  if [[ -x "${HOME}/.sima-cli/.venv/bin/sima-cli" ]]; then
    SIMA_CLI_BIN="${HOME}/.sima-cli/.venv/bin/sima-cli"
  else
    SIMA_CLI_BIN="$(command -v sima-cli || true)"
  fi
}

require_sima_cli_neat_install() {
  if ! command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1 ||
     ! "${SIMA_CLI_BIN}" neat install --help >/dev/null 2>&1; then
    bootstrap_sima_cli_for_ci_if_needed || true
  fi

  if ! command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required for Neat artifact access." >&2
    echo "Set SIMA_CLI_BIN to a sima-cli executable with Neat artifact install support if needed." >&2
    exit 1
  fi

  if ! "${SIMA_CLI_BIN}" neat install --help >/dev/null 2>&1; then
    echo "ERROR: sima-cli with Neat artifact install support is required." >&2
    echo "Set SIMA_CLI_BIN to a sima-cli executable with 'neat install' support if needed." >&2
    exit 1
  fi
}

fetch_neat_internals_vulcan_artifacts() {
  local internals_ref="$1"
  local output_dir="$2"

  require_sima_cli_neat_install

  local -a base_args=(neat install --env "${NEAT_VULCAN_ENV}")
  if [[ -n "${NEAT_VULCAN_BASE_URL}" ]]; then
    base_args+=(--base-url "${NEAT_VULCAN_BASE_URL}")
  fi

  local resolve_output resolved_ref
  if ! resolve_output="$("${SIMA_CLI_BIN}" "${base_args[@]}" "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
    if [[ "${NEAT_INTERNALS_SNAP_TAG_POLICY}" == "ON" ]]; then
      echo "ERROR: Failed to resolve exact tag-snap internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi
    if [[ "${NEAT_INTERNALS_SNAP_POLICY}" != "ON" || "${internals_ref}" == "develop:latest" ]]; then
      echo "ERROR: Failed to resolve internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi

    echo "No internals Vulcan artifact found for '${internals_ref}'; retrying develop:latest." >&2
    internals_ref="develop:latest"
    if ! resolve_output="$("${SIMA_CLI_BIN}" "${base_args[@]}" "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
      echo "ERROR: Failed to resolve fallback internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi
  fi

  resolved_ref="$(python3 - <<'PY' "${resolve_output}"
import json
import sys

text = sys.argv[1]
start = text.find("{")
if start < 0:
    raise SystemExit("missing JSON object in sima-cli neat install --json output")
payload = json.loads(text[start:])
ref = str(payload.get("ref", "")).strip()
spec = str(payload.get("resolved_spec", "")).strip()
if not ref or not spec:
    raise SystemExit("sima-cli neat install --json did not return ref and resolved_spec")
print(f"{ref}:{spec}")
PY
  )"
  NEAT_INTERNALS_RESOLVED_REF="${resolved_ref}"

  local -a install_args=(
    "${base_args[@]}"
    -d "${output_dir}"
    "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  )

  echo "Fetching neat-internals packages from Vulcan: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  if ! "${SIMA_CLI_BIN}" "${install_args[@]}"; then
    echo "ERROR: Failed to fetch internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}" >&2
    exit 1
  fi
}

fetch_neat_llima_vulcan_artifacts() {
  local llima_ref="$1"
  local output_dir="$2"

  require_sima_cli_neat_install

  local -a base_args=(neat install --env "${NEAT_VULCAN_ENV}")
  if [[ -n "${NEAT_VULCAN_BASE_URL}" ]]; then
    base_args+=(--base-url "${NEAT_VULCAN_BASE_URL}")
  fi

  local resolve_output resolved_ref
  if ! resolve_output="$("${SIMA_CLI_BIN}" "${base_args[@]}" "${NEAT_LLIMA_VULCAN_REPOSITORY}@${llima_ref}" --json)"; then
    if [[ "${NEAT_LLIMA_SNAP_TAG_POLICY}" == "ON" ]]; then
      echo "ERROR: Failed to resolve exact tag-snap LLiMa Vulcan artifact: ${NEAT_LLIMA_VULCAN_REPOSITORY}@${llima_ref}" >&2
      exit 1
    fi
    if [[ "${NEAT_LLIMA_SNAP_POLICY}" != "ON" || "${llima_ref}" == "develop:latest" ]]; then
      echo "ERROR: Failed to resolve LLiMa Vulcan artifact: ${NEAT_LLIMA_VULCAN_REPOSITORY}@${llima_ref}" >&2
      exit 1
    fi

    echo "No LLiMa Vulcan artifact found for '${llima_ref}'; retrying develop:latest." >&2
    llima_ref="develop:latest"
    if ! resolve_output="$("${SIMA_CLI_BIN}" "${base_args[@]}" "${NEAT_LLIMA_VULCAN_REPOSITORY}@${llima_ref}" --json)"; then
      echo "ERROR: Failed to resolve fallback LLiMa Vulcan artifact: ${NEAT_LLIMA_VULCAN_REPOSITORY}@${llima_ref}" >&2
      exit 1
    fi
  fi

  resolved_ref="$(python3 - <<'PY' "${resolve_output}"
import json
import sys

text = sys.argv[1]
start = text.find("{")
if start < 0:
    raise SystemExit("missing JSON object in sima-cli neat install --json output")
payload = json.loads(text[start:])
ref = str(payload.get("ref", "")).strip()
spec = str(payload.get("resolved_spec", "")).strip()
if not ref or not spec:
    raise SystemExit("sima-cli neat install --json did not return ref and resolved_spec")
print(f"{ref}:{spec}")
PY
  )"
  NEAT_LLIMA_RESOLVED_REF="${resolved_ref}"

  local -a install_args=(
    "${base_args[@]}"
    -d "${output_dir}"
    "${NEAT_LLIMA_VULCAN_REPOSITORY}@${resolved_ref}"
  )

  echo "Fetching LLiMa packages from Vulcan: ${NEAT_LLIMA_VULCAN_REPOSITORY}@${resolved_ref}"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  if ! "${SIMA_CLI_BIN}" "${install_args[@]}"; then
    echo "ERROR: Failed to fetch LLiMa Vulcan artifact: ${NEAT_LLIMA_VULCAN_REPOSITORY}@${resolved_ref}" >&2
    exit 1
  fi
}

ensure_neat_internals() {
  # Sync neat-internals from Vulcan package artifacts, then materialize plugins.
  local internals_ref
  if ! resolve_neat_internals_ref; then
    exit 1
  fi
  internals_ref="${NEAT_INTERNALS_REQUESTED_REF}"
  NEAT_INTERNALS_RESOLVED_REF="${internals_ref}"

  local marker_file="${NEAT_INTERNALS_DIR}/.internals"
  local deb_cache_dir="${NEAT_INTERNALS_DEB_DIR}"

  local tmp_dir
  # Use an isolated temp workspace so partial failures do not pollute tree state.
  tmp_dir="$(mktemp -d /tmp/sima-neat-internals-XXXXXX)"

  local artifact_dir="${tmp_dir}/package"
  local plugins_list_file="${tmp_dir}/plugin-files.list"

  if [[ -f "${marker_file}" ]] && [[ -d "${NEAT_INTERNALS_PLUGIN_DIR}" ]]; then
    local current_tag
    current_tag="$(tr -d '[:space:]' < "${marker_file}")"
    # Cache hit requires matching resolved Vulcan spec and a known plugin sentinel.
    if [[ "${current_tag}" == "${internals_ref}" ]] &&
       [[ -f "${NEAT_INTERNALS_PLUGIN_DIR}/libgstneatdecoder.so" ]] &&
       compgen -G "${deb_cache_dir}/neat-*.deb" >/dev/null 2>&1; then
      echo "Using cached neat-internals plugins/debs (${internals_ref})."
      rm -rf "${tmp_dir}"
      return 0
    fi
  fi

  fetch_neat_internals_vulcan_artifacts "${internals_ref}" "${artifact_dir}"
  internals_ref="${NEAT_INTERNALS_RESOLVED_REF:-${internals_ref}}"

  if ! collect_plugin_files_from_debs "${artifact_dir}" "${plugins_list_file}" "${deb_cache_dir}"; then
    echo "ERROR: Vulcan internals artifact did not contain .deb packages." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  copy_plugins_to_neat_internals "${plugins_list_file}"

  printf '%s\n' "${internals_ref}" > "${marker_file}"
  rm -rf "${tmp_dir}"
}

ensure_neat_llima() {
  # Sync LLiMa packages from Vulcan artifacts. Only core/dev are installed into
  # the SDK sysroot; cli is cached for the final DevKit install artifact.
  local llima_ref
  if ! resolve_neat_llima_ref; then
    exit 1
  fi
  llima_ref="${NEAT_LLIMA_REQUESTED_REF}"
  NEAT_LLIMA_RESOLVED_REF="${llima_ref}"

  local marker_file="${NEAT_INTERNALS_DIR}/.llima"
  local deb_cache_dir="${NEAT_LLIMA_DEB_DIR}"

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sima-neat-llima-XXXXXX)"

  local artifact_dir="${tmp_dir}/package"
  local using_cached_debs=0
  if [[ -f "${marker_file}" ]] &&
     [[ "$(tr -d '[:space:]' < "${marker_file}")" == "${llima_ref}" ]] &&
     compgen -G "${deb_cache_dir}/sima-lmm-*-Linux-core.deb" >/dev/null 2>&1 &&
     compgen -G "${deb_cache_dir}/sima-lmm-*-Linux-dev.deb" >/dev/null 2>&1 &&
     compgen -G "${deb_cache_dir}/sima-lmm-*-Linux-cli.deb" >/dev/null 2>&1; then
    echo "Using cached LLiMa debs (${llima_ref})."
    artifact_dir="${deb_cache_dir}"
    using_cached_debs=1
  else
    fetch_neat_llima_vulcan_artifacts "${llima_ref}" "${artifact_dir}"
    llima_ref="${NEAT_LLIMA_RESOLVED_REF:-${llima_ref}}"
  fi

  local core_deb dev_deb cli_deb
  core_deb="$(find "${artifact_dir}" -maxdepth 3 -type f -name 'sima-lmm-*-Linux-core.deb' | sort | head -n 1)"
  dev_deb="$(find "${artifact_dir}" -maxdepth 3 -type f -name 'sima-lmm-*-Linux-dev.deb' | sort | head -n 1)"
  cli_deb="$(find "${artifact_dir}" -maxdepth 3 -type f -name 'sima-lmm-*-Linux-cli.deb' | sort | head -n 1)"
  if [[ -z "${core_deb}" || -z "${dev_deb}" || -z "${cli_deb}" ]]; then
    echo "ERROR: Expected sima-lmm core/dev/cli debs were not found in Vulcan LLiMa artifact." >&2
    find "${artifact_dir}" -maxdepth 3 -type f -name '*.deb' -printf '  %f\n' | sort >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  if [[ "${using_cached_debs}" != "1" ]]; then
    mkdir -p "${deb_cache_dir}"
    rm -f "${deb_cache_dir}"/sima-lmm-*.deb
    cp -f "${core_deb}" "${deb_cache_dir}/$(basename "${core_deb}")"
    cp -f "${dev_deb}" "${deb_cache_dir}/$(basename "${dev_deb}")"
    cp -f "${cli_deb}" "${deb_cache_dir}/$(basename "${cli_deb}")"
  fi

  local -a llima_debs=("${core_deb}" "${dev_deb}")
  local install_root="${SYSROOT:-}"

  if [[ "${ELXR_SDK}" == "ON" ]]; then
    if [[ -z "${install_root}" ]]; then
      echo "ERROR: eLxr SDK mode requires SYSROOT to be set before installing LLiMa packages." >&2
      rm -rf "${tmp_dir}"
      exit 1
    fi
    if ! mkdir -p "${install_root}" 2>/dev/null; then
      if ! run_privileged mkdir -p "${install_root}"; then
        echo "ERROR: Unable to create SYSROOT directory: ${install_root}" >&2
        rm -rf "${tmp_dir}"
        exit 1
      fi
    fi
    echo "Installing LLiMa .deb payloads into eLxr sysroot: ${install_root}"
    ensure_llima_sdk_sysroot_deps
    local deb_path
    for deb_path in "${llima_debs[@]}"; do
      echo "  $(basename "${deb_path}")"
      if ! dpkg-deb -x "${deb_path}" "${install_root}" 2>/dev/null; then
        if ! run_privileged dpkg-deb -x "${deb_path}" "${install_root}"; then
          echo "ERROR: Failed to install $(basename "${deb_path}") into SYSROOT=${install_root}" >&2
          rm -rf "${tmp_dir}"
          exit 1
        fi
      fi
    done
  else
    if ! command -v dpkg >/dev/null 2>&1; then
      echo "ERROR: dpkg is required to install LLiMa .deb packages." >&2
      rm -rf "${tmp_dir}"
      exit 1
    fi
    echo "Installing LLiMa .deb packages into host system..."
    if command -v apt >/dev/null 2>&1; then
      mapfile -t llima_deb_abs_files < <(for deb in "${llima_debs[@]}"; do realpath "${deb}"; done)
      if ! run_privileged apt install -y --allow-downgrades -o Dpkg::Options::=--force-overwrite "${llima_deb_abs_files[@]}"; then
        echo "ERROR: Failed to install LLiMa packages via apt." >&2
        rm -rf "${tmp_dir}"
        exit 1
      fi
    else
      if ! run_privileged dpkg -i "${llima_debs[@]}"; then
        echo "ERROR: Failed to install LLiMa packages." >&2
        rm -rf "${tmp_dir}"
        exit 1
      fi
    fi
  fi

  local cmake_config runtime_lib
  if [[ "${ELXR_SDK}" == "ON" ]]; then
    cmake_config="${install_root}/usr/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake"
    runtime_lib="${install_root}/usr/lib/aarch64-linux-gnu/libsima_lmm_runtime.so"
  else
    cmake_config="/usr/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake"
    runtime_lib="/usr/lib/aarch64-linux-gnu/libsima_lmm_runtime.so"
  fi

  if [[ ! -f "${cmake_config}" ]]; then
    echo "ERROR: SimaLMM CMake package not found after install: ${cmake_config}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi
  if [[ ! -f "${runtime_lib}" ]]; then
    echo "ERROR: LLiMa runtime library not found after install: ${runtime_lib}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  printf '%s\n' "${llima_ref}" > "${marker_file}"
  rm -rf "${tmp_dir}"
}

copy_deb_usr_include_to_header_cache() {
  local deb_path="$1"
  local extract_dir="$2"

  mkdir -p "${extract_dir}"
  dpkg-deb -x "${deb_path}" "${extract_dir}"
  if [[ -d "${extract_dir}/usr/include" ]]; then
    mkdir -p "${NEAT_DEP_HEADERS_DIR}/usr"
    cp -a "${extract_dir}/usr/include" "${NEAT_DEP_HEADERS_DIR}/usr/"
  fi
}

ensure_neat_internals_headers() {
  # Header-only bootstrap for x86 analysis jobs. Do not install arm64 runtime/plugin packages.
  local internals_ref
  if ! resolve_neat_internals_ref; then
    exit 1
  fi
  internals_ref="${NEAT_INTERNALS_REQUESTED_REF}"
  NEAT_INTERNALS_RESOLVED_REF="${internals_ref}"

  local marker_file="${NEAT_DEP_HEADERS_DIR}/.internals_headers"

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sima-neat-internals-headers-XXXXXX)"
  local artifact_dir="${tmp_dir}/package"
  local deb_extract_dir="${tmp_dir}/deb-extract"

  if [[ -f "${marker_file}" ]] &&
     [[ "$(tr -d '[:space:]' < "${marker_file}")" == "${internals_ref}" ]] &&
     [[ -f "${NEAT_DEP_HEADERS_DIR}/usr/include/simaai/gstsimaaitensorbuffer.h" ]] &&
     [[ -f "${NEAT_DEP_HEADERS_DIR}/usr/include/gst/SimaTensorSetMetaAbi.h" ]]; then
    echo "Using cached neat-internals headers (${internals_ref})."
    rm -rf "${tmp_dir}"
    return 0
  fi

  fetch_neat_internals_vulcan_artifacts "${internals_ref}" "${artifact_dir}"
  internals_ref="${NEAT_INTERNALS_RESOLVED_REF:-${internals_ref}}"

  local dev_deb
  dev_deb="$(find "${artifact_dir}" -type f -name 'neat-internals-dev_*.deb' | sort | head -n 1)"
  if [[ -z "${dev_deb}" ]]; then
    echo "ERROR: neat-internals-dev package was not found in Vulcan artifact ${internals_ref}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mkdir -p "${NEAT_INTERNALS_DEB_DIR}"
  cp -f "${dev_deb}" "${NEAT_INTERNALS_DEB_DIR}/$(basename "${dev_deb}")"
  copy_deb_usr_include_to_header_cache "${dev_deb}" "${deb_extract_dir}"

  if [[ ! -f "${NEAT_DEP_HEADERS_DIR}/usr/include/simaai/gstsimaaitensorbuffer.h" ||
        ! -f "${NEAT_DEP_HEADERS_DIR}/usr/include/gst/SimaTensorSetMetaAbi.h" ]]; then
    echo "ERROR: neat-internals headers are incomplete under ${NEAT_DEP_HEADERS_DIR}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mkdir -p "${NEAT_DEP_HEADERS_DIR}"
  printf '%s\n' "${internals_ref}" > "${marker_file}"
  rm -rf "${tmp_dir}"
}

ensure_neat_llima_headers() {
  # Header-only bootstrap for x86 analysis jobs. GenAI tidy may still be skipped
  # unless a full SimaLMM CMake package/runtime is available.
  local llima_ref
  if ! resolve_neat_llima_ref; then
    exit 1
  fi
  llima_ref="${NEAT_LLIMA_REQUESTED_REF}"
  NEAT_LLIMA_RESOLVED_REF="${llima_ref}"

  local marker_file="${NEAT_DEP_HEADERS_DIR}/.llima_headers"

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sima-neat-llima-headers-XXXXXX)"
  local artifact_dir="${tmp_dir}/package"
  local deb_extract_dir="${tmp_dir}/deb-extract"

  if [[ -f "${marker_file}" ]] &&
     [[ "$(tr -d '[:space:]' < "${marker_file}")" == "${llima_ref}" ]] &&
     [[ -f "${NEAT_DEP_HEADERS_DIR}/usr/include/sima_lmm/chat.hpp" ]]; then
    echo "Using cached LLiMa headers (${llima_ref})."
    rm -rf "${tmp_dir}"
    return 0
  fi

  fetch_neat_llima_vulcan_artifacts "${llima_ref}" "${artifact_dir}"
  llima_ref="${NEAT_LLIMA_RESOLVED_REF:-${llima_ref}}"

  local dev_deb
  dev_deb="$(find "${artifact_dir}" -maxdepth 3 -type f -name 'sima-lmm-*-Linux-dev.deb' | sort | head -n 1)"
  if [[ -z "${dev_deb}" ]]; then
    echo "ERROR: sima-lmm dev package was not found in Vulcan LLiMa artifact." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mkdir -p "${NEAT_LLIMA_DEB_DIR}"
  cp -f "${dev_deb}" "${NEAT_LLIMA_DEB_DIR}/$(basename "${dev_deb}")"
  copy_deb_usr_include_to_header_cache "${dev_deb}" "${deb_extract_dir}"

  if [[ ! -f "${NEAT_DEP_HEADERS_DIR}/usr/include/sima_lmm/chat.hpp" ]]; then
    echo "ERROR: LLiMa headers are incomplete under ${NEAT_DEP_HEADERS_DIR}" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mkdir -p "${NEAT_DEP_HEADERS_DIR}"
  printf '%s\n' "${llima_ref}" > "${marker_file}"
  rm -rf "${tmp_dir}"
}

ensure_dependency_headers() {
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo "Skipping NEAT/LLiMa header artifact extraction on macOS."
    return 0
  fi
  ensure_neat_internals_headers
  ensure_neat_llima_headers
}

collect_install_artifact_files() {
  local -n out_files_ref="$1"
  out_files_ref=()

  local -A seen_basenames=()
  local file
  local basename_file

  for file in dist/*.deb ./*.deb; do
    [[ -e "${file}" ]] || continue
    basename_file="$(basename "${file}")"
    [[ -n "${seen_basenames[${basename_file}]:-}" ]] && continue
    seen_basenames["${basename_file}"]=1
    out_files_ref+=("${file}")
  done

  for file in "${NEAT_INTERNALS_DEB_DIR}"/neat-*.deb \
              "${NEAT_INTERNALS_DEB_DIR}"/simaai-common*.deb \
              "${NEAT_INTERNALS_DEB_DIR}"/neat-appcomplex_*.deb; do
    [[ -e "${file}" ]] || continue
    basename_file="$(basename "${file}")"
    [[ -n "${seen_basenames[${basename_file}]:-}" ]] && continue
    seen_basenames["${basename_file}"]=1
    out_files_ref+=("${file}")
  done

  for file in "${NEAT_LLIMA_DEB_DIR}"/sima-lmm-*.deb; do
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

  for file in dist/manifest.json "${NEAT_DEPS_MANIFEST}"; do
    [[ -e "${file}" ]] || continue
    basename_file="manifest.json"
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

  if [[ "${ELXR_SDK}" == "ON" && "${ALLOW_FUZZ_IN_NEAT_SDK}" != "ON" ]]; then
    echo "ERROR: --fuzz is not supported in local NEAT SDK environments by default." >&2
    echo "Run fuzz builds on Modalix ARM64 runners/devkits where libFuzzer targets execute natively." >&2
    echo "Set SIMANEAT_ALLOW_FUZZ_IN_NEAT_SDK=ON only for the Vulcan CI container fuzz lane." >&2
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
  echo "Neat LLiMa     : ${INSTALL_NEAT_LLIMA}"
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
    -DSIMANEAT_BUILD_TESTS="${BUILD_TESTS}"
    -DSIMANEAT_BUILD_TUTORIALS="${BUILD_TUTORIALS}"
    -DSIMANEAT_BUILD_PYTHON="${BUILD_PYTHON}"
    -DSIMANEAT_BUILD_DOCS="${BUILD_DOCS}"
    -DSIMANEAT_STRICT_WARNINGS="${STRICT_WARNINGS}"
    -DSIMA_ENABLE_ASAN="${SIMA_ENABLE_ASAN}"
    -DSIMA_ENABLE_UBSAN="${SIMA_ENABLE_UBSAN}"
    -DSIMA_ENABLE_TSAN="${SIMA_ENABLE_TSAN}"
    -DSIMANEAT_SANITIZER_GATE_ONLY_EXTRAS="${SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS}"
    -DFUZZING="${BUILD_FUZZ}"
  )

  if [[ -f "${NEAT_PACKAGE_BUILDINFO_JSON}" ]]; then
    local buildinfo_json_path="${NEAT_PACKAGE_BUILDINFO_JSON}"
    if [[ "${buildinfo_json_path}" != /* ]]; then
      buildinfo_json_path="${REPO_ROOT}/${buildinfo_json_path}"
    fi
    cmake_args+=("-DSIMANEAT_BUILDINFO_JSON=${buildinfo_json_path}")
  fi

  if [[ "${DOCS_ONLY}" == "ON" ]]; then
    # The docs job only needs the Doxygen target and public headers.  Some
    # x64 docs runners do not have the ARM NEAT/GStreamer runtime artifacts
    # installed, so do not make those runtime libraries configure-time
    # requirements for docs-only builds.
    cmake_args+=(
      -DSIMANEAT_DOCS_ONLY_CONFIGURE=ON
      -DSIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=OFF
      -DSIMANEAT_REQUIRE_LLIMA_ARTIFACTS=OFF
    )
  fi

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
    if [[ -d "${SYSROOT}/usr/include/c++/12" ]]; then
      export CXXFLAGS="${CXXFLAGS:-} -isystem ${SYSROOT}/usr/include/c++/12"
    fi
    if [[ -d "${SYSROOT}/usr/include/aarch64-linux-gnu/c++/12" ]]; then
      export CXXFLAGS="${CXXFLAGS:-} -isystem ${SYSROOT}/usr/include/aarch64-linux-gnu/c++/12"
    fi

    cmake_args+=(
      -DCMAKE_SYSROOT="${SYSROOT}"
      -DCMAKE_FIND_ROOT_PATH="${SYSROOT}"
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
      -DCMAKE_PREFIX_PATH="${SYSROOT}/usr;${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/lib/cmake"
      -DPKG_CONFIG_EXECUTABLE="${pkg_config_executable}"
      -DPython3_EXECUTABLE="${ELXR_HOST_PYTHON_EXECUTABLE}"
      -DPython_EXECUTABLE="${ELXR_HOST_PYTHON_EXECUTABLE}"
      -DPython_INCLUDE_DIR="${ELXR_TARGET_PYTHON_INCLUDE_DIR}"
      -DPYNEAT_EXT_SUFFIX=".cpython-${ELXR_TARGET_PYTHON_VERSION/./}-$(elxr_ext_platform_triplet).so"
      -DSIMANEAT_CTEST_FOR_DEVKIT=ON
    )
  fi

  cmake "${cmake_args[@]}"
}

build_docs_site() {
  # Shared docs pipeline used by both --doc and --all/--no-doc flows.
  cd "${REPO_ROOT}"
  echo
  echo "Refreshing compatibility snapshot..."
  sync_compatibility_snapshot
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
  echo "Running autodoc (pull external repo docs)..."
  python3 tools/autodoc.py \
    --conf "${REPO_ROOT}/tools/autodoc.conf.json" \
    --repo-root "${REPO_ROOT}" \
    --build-dir "${BUILD_DIR}" \
    --out-root "${REPO_ROOT}/docs" || true
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
  # For dev-only builds, avoid building tests/tutorials by targeting core lib.
  if [[ "${BUILD_TESTS}" == "OFF" && "${BUILD_TUTORIALS}" == "OFF" && "${BUILD_DOCS}" == "OFF" ]]; then
    cmake --build "${BUILD_DIR}" --target sima_neat_libraries -j"${BUILD_JOBS}"
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

compute_neat_package_version() {
  "${REPO_ROOT}/tools/compute_version.sh"
}

compute_neat_platform_version() {
  python3 - "${NEAT_DEPS_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
version = str(data.get("platform-version", "")).strip()
if not version:
    raise SystemExit(f"Missing or empty 'platform-version' in {manifest_path}")
print(version)
PY
}

generate_package_buildinfo_json() {
  local output_path="${NEAT_PACKAGE_BUILDINFO_JSON}"
  local package_version
  local platform_version
  package_version="$(compute_neat_package_version)"
  platform_version="$(compute_neat_platform_version)"

  # Best-effort cross-toolchain version string for the compatibility block.
  local toolchain_version=""
  local cxx_bin="${ELXR_CXX:-aarch64-linux-gnu-g++}"
  if command -v "${cxx_bin}" >/dev/null 2>&1; then
    toolchain_version="$("${cxx_bin}" --version 2>/dev/null | head -1 | sed 's/(.*) //')"
  fi

  python3 scripts/build/generate_package_buildinfo.py \
    --repo-root "${REPO_ROOT}" \
    --output "${output_path}" \
    --package-name "${NEAT_PACKAGE_NAME}" \
    --package-version "${package_version}" \
    --platform-version "${platform_version}" \
    --vulcan-env "${NEAT_VULCAN_ENV}" \
    --internals-ref "${NEAT_INTERNALS_RESOLVED_REF}" \
    --llima-ref "${NEAT_LLIMA_RESOLVED_REF}" \
    --toolchain "${toolchain_version}"
  echo "Generated package build info: ${output_path}"
}

sync_compatibility_snapshot() {
  # Refresh the committed buildinfo snapshot the Compatibility doc renders from,
  # whenever a freshly generated buildinfo is available. Standalone --doc builds
  # (no package build) keep the existing committed snapshot.
  local src="${NEAT_PACKAGE_BUILDINFO_JSON}"
  if [[ "${src}" != /* ]]; then
    src="${REPO_ROOT}/${src}"
  fi
  local dst="${REPO_ROOT}/website/src/data/buildinfo.json"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}"
    echo "Synced compatibility snapshot: ${dst}"
  else
    echo "No fresh buildinfo.json; keeping committed compatibility snapshot at ${dst}."
  fi
}

write_resolved_neat_internals_manifest_if_needed() {
  if [[ -z "${NEAT_INTERNALS_RESOLVED_REF}" && -z "${NEAT_LLIMA_RESOLVED_REF}" ]]; then
    return 0
  fi

  local output_path="${NEAT_INTERNALS_RESOLVED_MANIFEST}"
  mkdir -p "$(dirname "${output_path}")"
  python3 - "${NEAT_DEPS_MANIFEST}" "${NEAT_INTERNALS_RESOLVED_REF}" "${NEAT_LLIMA_RESOLVED_REF}" "${output_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
internals_ref = sys.argv[2]
llima_ref = sys.argv[3]
output_path = Path(sys.argv[4])

data = json.loads(manifest_path.read_text(encoding="utf-8"))
platform_version = str(data.get("platform-version", "")).strip()
if not platform_version:
    raise SystemExit(f"Missing or empty 'platform-version' in {manifest_path}")

if internals_ref:
    data["internals"] = internals_ref
if llima_ref:
    data["llima"] = llima_ref
output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  echo "Resolved deps manifest: ${output_path} (internals=${NEAT_INTERNALS_RESOLVED_REF:-<unchanged>}, llima=${NEAT_LLIMA_RESOLVED_REF:-<unchanged>})"
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
  local pyproject_path="${REPO_ROOT}/pyproject.toml"
  local pyproject_backup
  local pyneat_package_version

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

  pyneat_package_version="$(compute_neat_package_version)"
  echo "Using pyneat package version: ${pyneat_package_version}"

  pyproject_backup="$(mktemp)"
  cp "${pyproject_path}" "${pyproject_backup}"
  restore_pyneat_pyproject() {
    if [[ -n "${pyproject_backup:-}" && -f "${pyproject_backup}" ]]; then
      cp "${pyproject_backup}" "${pyproject_path}"
      rm -f "${pyproject_backup}"
    fi
  }

  python3 - "${pyproject_path}" "${pyneat_package_version}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
version = sys.argv[2]
text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r'(?m)^version = "[^"]*"$',
    f'version = "{version}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit(f"Failed to update version in {path}")
path.write_text(updated, encoding="utf-8")
PY

  local wheel_build_status=0
  set +e
  (
    set -e
    rm -rf dist
    if [[ "${ELXR_SDK}" == "ON" ]]; then
      echo "Using eLxr wheel target platform: ${ELXR_WHEEL_HOST_PLATFORM}"
      echo "Preparing non-isolated wheel backend environment for cross-build..."
      "${wheel_python}" -m pip install --upgrade pip build scikit-build-core nanobind==2.5.0 ninja wheel
      local py_abi
      local py_triplet
      local pyneat_ext_suffix
      py_abi="${ELXR_TARGET_PYTHON_VERSION/./}"
      py_triplet="$(elxr_ext_platform_triplet)"
      pyneat_ext_suffix=".cpython-${py_abi}-${py_triplet}.so"
      echo "Using eLxr extension suffix override: ${pyneat_ext_suffix}"
      local wheel_cmake_args="-DPYNEAT_EXT_SUFFIX=${pyneat_ext_suffix} -DPython3_EXECUTABLE=${ELXR_HOST_PYTHON_EXECUTABLE} -DPython_EXECUTABLE=${ELXR_HOST_PYTHON_EXECUTABLE}"
      if [[ -n "${SYSROOT:-}" ]]; then
        wheel_cmake_args+=" -DPython_INCLUDE_DIR=${ELXR_TARGET_PYTHON_INCLUDE_DIR}"
        wheel_cmake_args+=" -DCMAKE_SYSROOT=${SYSROOT}"
        wheel_cmake_args+=" -DCMAKE_FIND_ROOT_PATH=${SYSROOT}"
        wheel_cmake_args+=" -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER"
        wheel_cmake_args+=" -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY"
        wheel_cmake_args+=" -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY"
        wheel_cmake_args+=" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY"
        wheel_cmake_args+=" -DCMAKE_PREFIX_PATH=${SYSROOT}/usr\\;${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake\\;${SYSROOT}/usr/lib/cmake"
        wheel_cmake_args+=" -DSimaLMM_DIR=${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/SimaLMM"
        wheel_cmake_args+=" -DSIMANEAT_REQUIRE_LLIMA_ARTIFACTS=ON"
      fi
      # In eLxr cross-builds, PEP517 isolation may pull target-arch build tools
      # (notably ninja), which are not executable on the host container.
      # Build without isolation and force Makefiles to keep host tools executable.
      local backend_pythonpath
      backend_pythonpath="$("${wheel_python}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
      _PYTHON_HOST_PLATFORM="${ELXR_WHEEL_HOST_PLATFORM}" \
        PYTHONPATH="${backend_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
        CMAKE_ARGS="${wheel_cmake_args}" \
        CMAKE_GENERATOR="Unix Makefiles" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" SIMANEAT_BUILD_PYTHON=ON \
        "${ELXR_HOST_PYTHON_EXECUTABLE}" -m build --wheel --outdir dist --no-isolation
      mapfile -t built_wheels < <(find dist -maxdepth 1 -type f -name 'pyneat-*.whl' | sort)
      if [[ "${#built_wheels[@]}" -ne 1 ]]; then
        echo "ERROR: expected exactly one pyneat wheel, found ${#built_wheels[@]}." >&2
        printf '  %s\n' "${built_wheels[@]}" >&2
        exit 1
      fi
      "${wheel_python}" -m wheel tags \
        --remove \
        --python-tag "cp${py_abi}" \
        --abi-tag "cp${py_abi}" \
        --platform-tag "${ELXR_WHEEL_HOST_PLATFORM//-/_}" \
        "${built_wheels[0]}"
    else
      CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" SIMANEAT_BUILD_PYTHON=ON \
        "${wheel_python}" -m build --wheel --outdir dist
    fi
    echo "Built wheel(s):"
    ls -lh dist/*.whl
  )
  wheel_build_status=$?
  set -e

  restore_pyneat_pyproject
  if [[ "${wheel_build_status}" -ne 0 ]]; then
    exit "${wheel_build_status}"
  fi
}

run_install_sanity_check() {
  echo
  echo "Running install sanity check..."
  local install_test_dir="/tmp/sima-neat-install-test"
  rm -rf "${install_test_dir}"
  local core_install_dir="${install_test_dir}/core"
  local dev_install_dir="${install_test_dir}/dev"

  cmake --install "${BUILD_DIR}" --component core --prefix "${core_install_dir}"
  cmake --install "${BUILD_DIR}" --component dev --prefix "${dev_install_dir}"

  echo "Installed core files:"
  find "${core_install_dir}" | sed 's|^|  |'
  echo "Installed dev files:"
  find "${dev_install_dir}" | sed 's|^|  |'

  local -a core_real_libs=()
  local -a core_soname_links=()
  local expected_abi=""
  expected_abi="$(sed -n 's/^set(SimaNeat_ABI_VERSION "\([0-9][0-9]*\)")$/\1/p' \
    "${BUILD_DIR}/SimaNeatConfig.cmake")"
  if [[ ! "${expected_abi}" =~ ^[1-9][0-9]*$ ]]; then
    echo
    echo "ERROR: unable to read a valid public ABI from ${BUILD_DIR}/SimaNeatConfig.cmake."
    echo "Refusing to package a library without an explicit ABI contract."
    exit 1
  fi
  mapfile -t core_real_libs < <(find "${core_install_dir}/lib" -maxdepth 1 -type f -name 'libsima_neat.so.*' | sort)
  mapfile -t core_soname_links < <(find "${core_install_dir}/lib" -maxdepth 1 -type l -name 'libsima_neat.so.*' | sort)
  if [[ "${#core_real_libs[@]}" -lt 1 ]]; then
    echo
    echo "ERROR: versioned libsima_neat shared object missing from core install tree."
    echo "Refusing to package an incomplete core .deb."
    exit 1
  fi
  if [[ ! -L "${core_install_dir}/lib/libsima_neat.so.${expected_abi}" ]]; then
    echo
    echo "ERROR: expected ABI ${expected_abi} SONAME link missing from core install tree."
    echo "Refusing to package a core .deb with a stale or missing public-library ABI."
    exit 1
  fi
  if [[ "${#core_soname_links[@]}" -lt 1 ]]; then
    echo
    echo "ERROR: SONAME libsima_neat.so.* symlink missing from core install tree."
    echo "Refusing to package an incomplete core .deb."
    exit 1
  fi
  if [[ -e "${core_install_dir}/lib/libsima_neat.so" ||
        -e "${core_install_dir}/lib/libsima_neat.a" ||
        -d "${core_install_dir}/include" ||
        -d "${core_install_dir}/lib/cmake/SimaNeat" ]]; then
    echo
    echo "ERROR: core install tree contains development files."
    echo "Refusing to package an incorrectly split core .deb."
    exit 1
  fi
  if [[ ! -L "${dev_install_dir}/lib/libsima_neat.so" ||
        ! -f "${dev_install_dir}/lib/libsima_neat.a" ||
        ! -f "${dev_install_dir}/include/neat.h" ||
        ! -f "${dev_install_dir}/lib/cmake/SimaNeat/SimaNeatConfig.cmake" ]]; then
    echo
    echo "ERROR: dev install tree is missing headers, CMake config, libsima_neat.so linker symlink, or libsima_neat.a."
    echo "Refusing to package an incomplete dev .deb."
    exit 1
  fi
  local dev_link_target=""
  dev_link_target="$(readlink "${dev_install_dir}/lib/libsima_neat.so")"
  if [[ "$(basename "${dev_link_target}")" != "libsima_neat.so.${expected_abi}" ]]; then
    echo
    echo "ERROR: dev linker symlink targets '${dev_link_target}', expected ABI ${expected_abi}."
    echo "Refusing to package mismatched core and development packages."
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
    echo "Building DEB packages (core + dev)..."
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

      if [[ ! -x "${install_prefix}/build.sh" ]]; then
        echo "ERROR: extras install tree is missing root build.sh." >&2
        echo "Expected layout: build.sh next to lib/ and share/." >&2
        exit 1
      fi

      if [[ ! -d "${install_prefix}/lib/sima-neat" ]] && [[ ! -d "${install_prefix}/share/sima-neat" ]]; then
        echo "ERROR: extras install tree is empty under ${install_prefix}." >&2
        exit 1
      fi

      # Keep extras version aligned with the generated core .deb version.
      core_deb="$(ls -1 ./sima-neat-*-Linux-core.deb 2>/dev/null | head -n1 || true)"
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

stage_package_artifacts_to_dist() {
  # Keep dist/ as the complete local artifact directory for full builds.
  if [[ "${SKIP_DIST}" == "ON" || "${BUILD_ALL}" != "ON" ]]; then
    return 0
  fi

  mkdir -p dist
  rm -f \
    dist/*-Linux-core.deb \
    dist/*-Linux-dev.deb \
    dist/*-Linux-extras.tar.gz \
    dist/neat-*.deb \
    dist/simaai-common*.deb \
    dist/appcomplex_*.deb \
    dist/sima-lmm-*.deb \
    "dist/${NEAT_PACKAGE_INSTALL_SCRIPT}" \
    "dist/${NEAT_INSTALL_MANIFEST}" \
    dist/metadata*.json \
    dist/manifest.json \
    dist/resolved-deps-manifest.json

  local staged_any=OFF
  local file
  for file in ./*.deb ./*extras.tar.gz; do
    [[ -e "${file}" ]] || continue
    mv -f "${file}" "dist/$(basename "${file}")"
    staged_any=ON
  done
  for file in "${NEAT_INTERNALS_DEB_DIR}"/*.deb "${NEAT_LLIMA_DEB_DIR}"/*.deb; do
    [[ -e "${file}" ]] || continue
    cp -f "${file}" "dist/$(basename "${file}")"
    staged_any=ON
  done

  # The Core package declares an explicit LLiMa ABI range. Fail before
  # publishing/installing a bundle when the copied LLiMa DEBs do not satisfy
  # that range (for example Core 0.2.x combined with LLiMa 0.3.x).
  python3 "${REPO_ROOT}/tools/validate_neat_package_bundle.py" "${REPO_ROOT}/dist"

  if [[ -f "tools/install_neat_framework.sh" ]]; then
    cp -f "tools/install_neat_framework.sh" "dist/install_neat_framework.sh"
    chmod +x "dist/install_neat_framework.sh"
    staged_any=ON
  fi

  if [[ "${staged_any}" == "ON" ]]; then
    echo
    echo "Moved package artifacts into dist/:"
    ls -lh dist/*.deb dist/*.whl dist/*extras.tar.gz dist/install_neat_framework.sh dist/manifest.json 2>/dev/null || true
  fi
}

write_dist_platform_version_manifest_field() {
  if [[ "${SKIP_DIST}" == "ON" || "${BUILD_ALL}" != "ON" ]]; then
    return 0
  fi
  if [[ ! -d dist ]]; then
    return 0
  fi

  python3 - "${NEAT_DEPS_MANIFEST}" "dist/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])

source = json.loads(source_path.read_text(encoding="utf-8"))
platform_version = str(source.get("platform-version", "")).strip()
if not platform_version:
    raise SystemExit(f"Missing or empty platform-version in {source_path}")

if target_path.exists():
    target = json.loads(target_path.read_text(encoding="utf-8"))
else:
    target = {}
if not isinstance(target, dict):
    raise SystemExit(f"{target_path} must contain a JSON object")

target["platform-version"] = platform_version
target_path.write_text(json.dumps(target, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY

  echo "Updated dist/manifest.json platform-version from ${NEAT_DEPS_MANIFEST}"
}

append_dist_manifest_matches() {
  local manifest_path="$1"
  local pattern="$2"
  local -a matches=()
  local file

  while IFS= read -r file; do
    matches+=("${file}")
  done < <(find dist -maxdepth 1 -type f -name "${pattern}" | sort)

  for file in "${matches[@]}"; do
    basename "${file}" >> "${manifest_path}"
  done
}

write_install_manifest() {
  if [[ "${SKIP_DIST}" == "ON" || "${BUILD_ALL}" != "ON" ]]; then
    return 0
  fi

  local manifest_path="dist/${NEAT_INSTALL_MANIFEST}"
  if [[ ! -d dist ]]; then
    echo "ERROR: Cannot write install manifest; missing dist/." >&2
    exit 1
  fi

  {
    echo "# Generated by build.sh. The installer must only consume artifacts listed here."
    echo "# Keep this file next to ${NEAT_PACKAGE_INSTALL_SCRIPT}."
  } > "${manifest_path}"

  append_dist_manifest_matches "${manifest_path}" 'simaai-common*.deb'
  append_dist_manifest_matches "${manifest_path}" 'simaai-memory-lib_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'simaai-memory-lib-dev_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'libcamera_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'libcamera-dev_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'libcamera-tools_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-common_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-appcomplex_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-ev74-firmware_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-runtime_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-gst-plugins_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'neat-internals-dev_*.deb'
  append_dist_manifest_matches "${manifest_path}" 'sima-lmm-*.deb'
  append_dist_manifest_matches "${manifest_path}" 'sima-neat-*-Linux-core.deb'
  append_dist_manifest_matches "${manifest_path}" 'sima-neat-*-Linux-dev.deb'
  append_dist_manifest_matches "${manifest_path}" '*.whl'

  echo "Built install manifest: ${manifest_path}"
}

require_sima_cli_packages_build() {
  if ! command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required to build package metadata." >&2
    echo "Set SIMA_CLI_BIN to a sima-cli executable with packages build support if needed." >&2
    exit 1
  fi
  if ! "${SIMA_CLI_BIN}" packages build --help >/dev/null 2>&1; then
    echo "ERROR: sima-cli packages build support is required to build package metadata." >&2
    echo "Set SIMA_CLI_BIN to a sima-cli executable with packages build support if needed." >&2
    exit 1
  fi
}

sima_cli_packages_build_supports_platform_compatibility() {
  local help_text
  help_text="$("${SIMA_CLI_BIN}" packages build --help 2>/dev/null || true)"
  [[ "${help_text}" == *"--board-platform"* && "${help_text}" == *"--palette-platform"* ]]
}

read_package_compatibility_args() {
  local manifest_path="${NEAT_DEPS_MANIFEST}"
  local -n out_ref="$1"

  mapfile -t out_ref < <(python3 - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
platform_version = str(data.get("platform-version", "")).strip()
if not platform_version:
    raise SystemExit(f"Missing or empty 'platform-version' in {manifest_path}")

board_platform = f"modalix@{platform_version}"
for arg in ("--board-platform", board_platform, "--palette-platform", platform_version):
    print(arg)
PY
  )
}

collect_single_dist_artifact() {
  local pattern="$1"
  local label="$2"
  local -n out_ref="$3"
  local -a matches=()

  while IFS= read -r file; do
    matches+=("${file}")
  done < <(find dist -maxdepth 1 -type f -name "${pattern}" | sort)

  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "ERROR: Expected exactly one ${label} in dist; found ${#matches[@]}." >&2
    printf '  %s\n' "${matches[@]}" >&2 || true
    exit 1
  fi

  out_ref="${matches[0]}"
}

generate_package_metadata_if_requested() {
  if [[ "${SKIP_DIST}" == "ON" ]]; then
    echo
    echo "Skipping package metadata generation (--no-dist)."
    return 0
  fi
  if [[ "${BUILD_ALL}" != "ON" ]]; then
    echo
    echo "Skipping package metadata generation (requires --all)."
    return 0
  fi
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo
    echo "Skipping package metadata generation on macOS."
    return 0
  fi

  echo
  echo "Building package metadata variants..."
  require_sima_cli_packages_build

  local core_deb=""
  local dev_deb=""
  local extras_tar=""
  local wheel_path=""
  collect_single_dist_artifact 'sima-neat-*-Linux-core.deb' 'NEAT core .deb' core_deb
  collect_single_dist_artifact 'sima-neat-*-Linux-dev.deb' 'NEAT dev .deb' dev_deb
  collect_single_dist_artifact '*-Linux-extras.tar.gz' 'extras tarball' extras_tar
  collect_single_dist_artifact '*.whl' 'Python wheel' wheel_path

  local install_script_path="dist/${NEAT_PACKAGE_INSTALL_SCRIPT}"
  if [[ ! -f "${install_script_path}" ]]; then
    echo "ERROR: Missing install script: ${install_script_path}" >&2
    exit 1
  fi
  if [[ ! -f "dist/${NEAT_INSTALL_MANIFEST}" ]]; then
    echo "ERROR: Missing install manifest: dist/${NEAT_INSTALL_MANIFEST}" >&2
    exit 1
  fi

  local core_deb_basename
  local package_version
  local extras_basename
  core_deb_basename="$(basename "${core_deb}")"
  package_version="${core_deb_basename#sima-neat-}"
  package_version="${package_version%-Linux-core.deb}"
  extras_basename="$(basename "${extras_tar}")"

  local -a package_compatibility_args=()
  if sima_cli_packages_build_supports_platform_compatibility; then
    read_package_compatibility_args package_compatibility_args
    echo "Using package compatibility: ${package_compatibility_args[*]}"
  else
    echo "sima-cli packages build does not support platform compatibility flags; metadata platforms will remain empty."
  fi

  rm -f dist/metadata.json dist/metadata-minimal.json dist/metadata-all.json dist/metadata-pyneat.json dist/metadata-extras.json

  "${SIMA_CLI_BIN}" packages build dist \
    --name "${NEAT_PACKAGE_NAME}" \
    --version "${package_version}" \
    --description "${NEAT_PACKAGE_DESCRIPTION}" \
    --install-script "${NEAT_PACKAGE_INSTALL_SCRIPT}" \
    --selectables "${NEAT_EXTRAS_SELECTABLE_NAME}:${extras_basename}" \
    "${package_compatibility_args[@]}"

  "${SIMA_CLI_BIN}" packages build dist \
    --name "${NEAT_PACKAGE_NAME}" \
    --version "${package_version}" \
    --description "${NEAT_PACKAGE_DESCRIPTION}" \
    --install-script "${NEAT_PACKAGE_INSTALL_SCRIPT}" \
    --exclude "${extras_basename}" \
    --variant minimal \
    "${package_compatibility_args[@]}"

  "${SIMA_CLI_BIN}" packages build dist \
    --name "${NEAT_PACKAGE_NAME}" \
    --version "${package_version}" \
    --description "${NEAT_PACKAGE_DESCRIPTION}" \
    --install-script "${NEAT_PACKAGE_INSTALL_SCRIPT}" \
    --variant all \
    "${package_compatibility_args[@]}"

  "${SIMA_CLI_BIN}" packages build dist \
    --name "${NEAT_PACKAGE_NAME}" \
    --version "${package_version}" \
    --description "PyNeat wheel for ${NEAT_PACKAGE_DESCRIPTION}" \
    --install-script ":" \
    --exclude ".deb" \
    --exclude ".tar.gz" \
    --exclude ".sh" \
    --exclude ".txt" \
    --exclude ".json" \
    --download-compatible-files-only \
    --variant pyneat \
    "${package_compatibility_args[@]}"

  "${SIMA_CLI_BIN}" packages build dist \
    --name "${NEAT_PACKAGE_NAME}" \
    --version "${package_version}" \
    --description "SiMa.ai Neat extras for ${NEAT_PACKAGE_DESCRIPTION}" \
    --install-script ":" \
    --exclude ".deb" \
    --exclude ".whl" \
    --exclude ".sh" \
    --exclude ".txt" \
    --exclude ".json" \
    --variant extras

  WHEEL_BASENAME="$(basename "${wheel_path}")" python3 - <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path("dist/metadata-pyneat.json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
wheel_name = os.environ["WHEEL_BASENAME"]
metadata.setdefault("installation", {})["post-message"] = (
    "[bold]PyNeat wheel downloaded.[/bold]\n"
    f"The PyNeat wheel has been downloaded to this directory: {wheel_name}\n"
    "Install it into a Python 3.11 virtual environment with:\n"
    f"  pip install ./{wheel_name}\n\n"
    "PyNeat requires the matching Neat Library runtime packages on this system. "
    "If they are not already installed, run:\n"
    "  sima-cli neat install core\n"
)
metadata_path.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")
PY

  EXTRAS_BASENAME="${extras_basename}" python3 - <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path("dist/metadata-extras.json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
extras_name = os.environ["EXTRAS_BASENAME"]
resources = metadata.get("resources", [])
if resources != [extras_name]:
    raise SystemExit(
        f"metadata-extras.json must contain only {extras_name!r}; got {resources!r}"
    )
metadata.setdefault("installation", {})["post-message"] = (
    "[bold]SiMa.ai Neat extras downloaded.[/bold]\n"
    f"The extras archive has been downloaded and extracted from: {extras_name}\n"
    "It contains tutorial source, prebuilt tutorial binaries, and supporting "
    "test/sample files.\n"
)
metadata_path.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")
PY

  echo "Built package metadata:"
  ls -lh dist/metadata.json dist/metadata-minimal.json dist/metadata-all.json dist/metadata-pyneat.json dist/metadata-extras.json
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
  if compgen -G "dist/*.deb" >/dev/null 2>&1 || compgen -G "dist/*extras.tar.gz" >/dev/null 2>&1; then
    echo "Dist package artifacts:"
    ls -lh dist/*.deb dist/*extras.tar.gz 2>/dev/null || true
  fi
  if compgen -G "dist/metadata*.json" >/dev/null 2>&1; then
    echo "Package metadata:"
    ls -lh dist/metadata*.json
  fi
}

should_deploy_to_devkit() {
  [[ "${INSTALL_AFTER_BUILD}" == "ON" ]] || return 1
  [[ "${ELXR_SDK}" == "ON" ]] || return 1
  [[ -n "${DEVKIT_SYNC_DEVKIT_IP:-}" ]] || return 1
  [[ "${OS_NAME}" != "Darwin" ]] || return 1

  compgen -G "./*.deb" >/dev/null 2>&1 ||
    compgen -G "dist/*.deb" >/dev/null 2>&1 ||
    [[ "${BUILD_PYTHON}" == "ON" && -n "$(compgen -G 'dist/*.whl' || true)" ]] ||
    return 1
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
  detect_elxr_host_python
  detect_elxr_target_python

  if [[ "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
    ensure_dependency_headers
  fi

  if [[ "${OS_NAME}" != "Darwin" && "${INSTALL_NEAT_INTERNALS}" == "ON" ]]; then
    ensure_neat_internals
  fi
  if [[ "${OS_NAME}" != "Darwin" && "${INSTALL_NEAT_LLIMA}" == "ON" ]]; then
    ensure_neat_llima
  fi

  if [[ "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
    echo
    echo "System dependencies and dependency headers installed. Exiting due to --install-deps-only."
    exit 0
  fi

  detect_build_jobs
  configure_fuzz_toolchain_if_needed
  generate_platform_version_artifacts
  print_build_config
  clean_build_dir_if_requested
  write_resolved_neat_internals_manifest_if_needed
  generate_package_buildinfo_json
  configure_cmake
  build_docs_only_if_requested
  build_targets
  copy_test_images
  build_docs_if_requested
  build_python_wheel_if_requested
  run_install_sanity_check
  build_deb_if_requested
  build_extras_archive_if_requested
  stage_package_artifacts_to_dist
  write_dist_platform_version_manifest_field
  write_install_manifest
  generate_package_metadata_if_requested
  print_artifact_summary
  install_artifacts_into_current_environment_if_requested
  deploy_artifacts_to_devkit_if_requested
}

main "$@"

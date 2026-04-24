#!/usr/bin/env bash
set -euo pipefail

# Resolve TUTORIALS_DIR / REPO_ROOT based on where this script lives.
# Supported layouts (extras takes precedence when both are reachable, since a
# user running build.sh from inside the extracted extras folder expects the
# output to land there):
#   1. Extras tarball:     extras-root/build.sh     -> tutorials at ./share/sima-neat/tutorials
#   2. Source tree:        core/tutorials/build.sh  -> tutorials at ../tutorials, build at ../build/...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "${SCRIPT_DIR}/share/sima-neat/tutorials/CMakeLists.txt" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
  TUTORIALS_DIR="${SCRIPT_DIR}/share/sima-neat/tutorials"
elif [[ -f "${SCRIPT_DIR}/../tutorials/CMakeLists.txt" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
  TUTORIALS_DIR="${REPO_ROOT}/tutorials"
else
  echo "build.sh: cannot locate tutorials/CMakeLists.txt (checked ./share/sima-neat/tutorials and ../tutorials)" >&2
  exit 1
fi
BUILD_DIR="${REPO_ROOT}/build/tutorials-standalone"
BUILD_TYPE="Release"
TARGET="all"
DO_CLEAN="OFF"
LIST_TARGETS="OFF"
JOBS=""
SIMANEAT_DIR="${SimaNeat_DIR:-}"

auto_jobs() {
  local cpu_count=1
  if command -v nproc >/dev/null 2>&1; then
    cpu_count="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  fi

  # Cap parallel compile jobs by available memory to avoid OOM kills.
  # Rule of thumb: ~3 GiB RAM per concurrent C++ compile unit.
  if [[ -r /proc/meminfo ]]; then
    local mem_kb mem_jobs
    mem_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if [[ -n "${mem_kb}" ]]; then
      mem_jobs=$(( mem_kb / (3 * 1024 * 1024) ))
      if (( mem_jobs < 1 )); then
        mem_jobs=1
      fi
      if (( mem_jobs < cpu_count )); then
        cpu_count="${mem_jobs}"
      fi
    fi
  fi

  if (( cpu_count < 1 )); then
    cpu_count=1
  fi
  echo "${cpu_count}"
}

usage() {
  cat <<EOF
Usage: tutorials/build.sh [options]

Options:
  --target <name>      Build one target (e.g. tutorial_v2_001_model_in_5_minutes)
  --build-dir <path>   Build directory (default: build/tutorials-standalone)
  --build-type <type>  CMake build type (default: Release)
  --clean              Remove build directory before configure
  --list-targets       List available tutorial targets and exit
  --simaneat-dir <dir> Path to directory containing SimaNeatConfig.cmake
  -j, --jobs <N>       Parallel build jobs (default: auto)
  -h, --help           Show this help

Examples:
  tutorials/build.sh
  tutorials/build.sh --target tutorial_v2_001_run_your_first_model
  tutorials/build.sh --simaneat-dir /usr/lib/aarch64-linux-gnu/cmake/SimaNeat
  tutorials/build.sh --list-targets
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN="ON"
      shift
      ;;
    --list-targets)
      LIST_TARGETS="ON"
      shift
      ;;
    --simaneat-dir)
      SIMANEAT_DIR="${2:-}"
      shift 2
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      shift 2
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

if [[ "${DO_CLEAN}" == "ON" ]]; then
  rm -rf "${BUILD_DIR}"
fi

discover_simaneat_dir() {
  if [[ -n "${SIMANEAT_DIR}" ]]; then
    return 0
  fi

  local -a candidates=()
  if [[ -n "${SYSROOT:-}" ]]; then
    candidates+=(
      "${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
      "${SYSROOT}/usr/lib/cmake/SimaNeat"
    )
  fi
  candidates+=(
    "/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
    "/usr/lib/cmake/SimaNeat"
    "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
    "/opt/toolchain/aarch64/modalix/usr/lib/cmake/SimaNeat"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/SimaNeatConfig.cmake" ]]; then
      SIMANEAT_DIR="${candidate}"
      return 0
    fi
  done
}

discover_simaneat_dir

cmake_args=(
  -S "${TUTORIALS_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)
if [[ -n "${SIMANEAT_DIR}" ]]; then
  cmake_args+=("-DSimaNeat_DIR=${SIMANEAT_DIR}")
  echo "Using SimaNeat_DIR=${SIMANEAT_DIR}"
else
  cat >&2 <<'EOF'
Failed to locate SimaNeatConfig.cmake.
Install NEAT core package first, or pass one of:
  tutorials/build.sh --simaneat-dir <path-to-cmake/SimaNeat>
  SimaNeat_DIR=<path-to-cmake/SimaNeat> tutorials/build.sh
EOF
  exit 1
fi

cmake "${cmake_args[@]}"

if [[ "${LIST_TARGETS}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" --target help | sed -n '/tutorial_/p'
  exit 0
fi

build_cmd=(cmake --build "${BUILD_DIR}")
if [[ "${TARGET}" != "all" ]]; then
  build_cmd+=(--target "${TARGET}")
fi
if [[ -n "${JOBS}" ]]; then
  build_cmd+=(--parallel "${JOBS}")
else
  AUTO_JOBS="$(auto_jobs)"
  echo "Auto-selected parallel jobs: ${AUTO_JOBS}"
  build_cmd+=(--parallel "${AUTO_JOBS}")
fi

"${build_cmd[@]}"

echo
if [[ "${TARGET}" == "all" ]]; then
  echo "Built tutorials under: ${BUILD_DIR}"
else
  echo "Built target '${TARGET}' under: ${BUILD_DIR}"
fi

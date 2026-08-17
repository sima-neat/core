#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -f "${SCRIPT_DIR}/share/sima-pcie-host/tutorials/CMakeLists.txt" ]]; then
  EXTRAS_ROOT="${SCRIPT_DIR}"
  TUTORIALS_DIR="${SCRIPT_DIR}/share/sima-pcie-host/tutorials"
elif [[ -f "${SCRIPT_DIR}/CMakeLists.txt" ]]; then
  EXTRAS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
  TUTORIALS_DIR="${SCRIPT_DIR}"
else
  echo "build.sh: cannot locate PCIe host tutorials/CMakeLists.txt" >&2
  exit 1
fi

BUILD_DIR="${EXTRAS_ROOT}/build/tutorials-standalone"
BUILD_TYPE="Release"
TARGET="all"
LIST_TARGETS="OFF"
DO_CLEAN="OFF"
JOBS=""
SIMAPCIEHOST_DIR="${SimaPCIeHost_DIR:-}"

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --target <name>            Build one PCIe tutorial target
  --build-dir <path>         Build directory (default: build/tutorials-standalone)
  --build-type <type>        CMake build type (default: Release)
  --simapciehost-dir <path>  Directory containing SimaPCIeHostConfig.cmake
  --list-targets             List available PCIe tutorial targets and exit
  --clean                    Reconfigure from an empty build directory
  -j, --jobs <N>             Parallel build jobs
  -h, --help                 Show this help
EOF
}

absolute_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path%/}"
  else
    printf '%s/%s\n' "${PWD%/}" "${path%/}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      [[ -n "${TARGET}" ]] || { echo "build.sh: --target requires a value" >&2; exit 1; }
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      [[ -n "${BUILD_DIR}" ]] || { echo "build.sh: --build-dir requires a value" >&2; exit 1; }
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="${2:-}"
      [[ -n "${BUILD_TYPE}" ]] || { echo "build.sh: --build-type requires a value" >&2; exit 1; }
      shift 2
      ;;
    --simapciehost-dir)
      SIMAPCIEHOST_DIR="${2:-}"
      [[ -n "${SIMAPCIEHOST_DIR}" ]] || { echo "build.sh: --simapciehost-dir requires a value" >&2; exit 1; }
      shift 2
      ;;
    --list-targets)
      LIST_TARGETS="ON"
      shift
      ;;
    --clean)
      DO_CLEAN="ON"
      shift
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "build.sh: --jobs must be a positive integer" >&2; exit 1; }
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "build.sh: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

BUILD_DIR="$(absolute_path "${BUILD_DIR}")"
if [[ "${DO_CLEAN}" == "ON" && -d "${BUILD_DIR}" ]]; then
  cmake -E remove_directory "${BUILD_DIR}"
fi

cmake_args=(
  -S "${TUTORIALS_DIR}"
  -B "${BUILD_DIR}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
)
if [[ -n "${SIMAPCIEHOST_DIR}" ]]; then
  cmake_args+=("-DSimaPCIeHost_DIR=${SIMAPCIEHOST_DIR}")
fi
cmake "${cmake_args[@]}"

if [[ "${LIST_TARGETS}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" --target help | sed -n 's/^\.\.\. \(tutorial_[^ ]*\)$/\1/p' | sort
  exit 0
fi

build_args=(--build "${BUILD_DIR}" --target "${TARGET}")
if [[ -n "${JOBS}" ]]; then
  build_args+=(--parallel "${JOBS}")
fi
cmake "${build_args[@]}"

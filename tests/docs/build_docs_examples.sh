#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLES_DIR="${ROOT_DIR}/tests/docs"
ENVIRONMENT=""
EXAMPLE="all"
BUILD_ROOT="${DOCS_EXAMPLE_BUILD_ROOT:-/tmp/sima-neat-doc-examples}"
INSTALL_PREFIX=""
INSTALL_FROM_BUILD=""
SDK_SYSROOT="${SYSROOT:-/opt/toolchain/aarch64/modalix}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --environment sdk|devkit [options]

Builds documentation C++ example fixtures with the same clean CMake pattern
shown in the user docs.

Options:
  --environment sdk|devkit      Required. Select SDK/sysroot or native DevKit mode.
  --example NAME|all            Example to build: hello-neat, run-an-app, or all.
  --build-root DIR              Build output root. Default: ${BUILD_ROOT}
  --install-prefix DIR          Installed SimaNeat prefix to use before system paths.
  --install-from-build DIR      Install core/dev components from a core build tree first.
  --sdk-sysroot DIR             SDK target sysroot. Default: ${SDK_SYSROOT}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --environment)
      ENVIRONMENT="${2:?missing value for --environment}"
      shift 2
      ;;
    --example)
      EXAMPLE="${2:?missing value for --example}"
      shift 2
      ;;
    --build-root)
      BUILD_ROOT="${2:?missing value for --build-root}"
      shift 2
      ;;
    --install-prefix)
      INSTALL_PREFIX="${2:?missing value for --install-prefix}"
      shift 2
      ;;
    --install-from-build)
      INSTALL_FROM_BUILD="${2:?missing value for --install-from-build}"
      shift 2
      ;;
    --sdk-sysroot)
      SDK_SYSROOT="${2:?missing value for --sdk-sysroot}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${ENVIRONMENT}" in
  sdk|devkit) ;;
  "")
    echo "--environment is required." >&2
    usage >&2
    exit 1
    ;;
  *)
    echo "Unsupported environment: ${ENVIRONMENT}" >&2
    usage >&2
    exit 1
    ;;
esac

case "${EXAMPLE}" in
  all)
    examples=(hello-neat run-an-app)
    ;;
  hello-neat|run-an-app)
    examples=("${EXAMPLE}")
    ;;
  *)
    echo "Unsupported example: ${EXAMPLE}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ -n "${INSTALL_FROM_BUILD}" ]]; then
  INSTALL_FROM_BUILD="$(cd "${INSTALL_FROM_BUILD}" && pwd)"
  if [[ -z "${INSTALL_PREFIX}" ]]; then
    INSTALL_PREFIX="${BUILD_ROOT}/installed-sima-neat"
  fi
  rm -rf "${INSTALL_PREFIX}"
  cmake --install "${INSTALL_FROM_BUILD}" --prefix "${INSTALL_PREFIX}" --component core
  cmake --install "${INSTALL_FROM_BUILD}" --prefix "${INSTALL_PREFIX}" --component dev
fi

cmake_prefix_parts=()
if [[ -n "${INSTALL_PREFIX}" ]]; then
  cmake_prefix_parts+=("${INSTALL_PREFIX}")
fi

if [[ "${ENVIRONMENT}" == "sdk" ]]; then
  if [[ ! -d "${SDK_SYSROOT}" ]]; then
    echo "SDK sysroot not found: ${SDK_SYSROOT}" >&2
    exit 1
  fi
  export SYSROOT="${SDK_SYSROOT}"
  cmake_prefix_parts+=(
    "${SDK_SYSROOT}/usr"
    "${SDK_SYSROOT}/usr/lib"
    "${SDK_SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
fi

cmake_prefix_path=""
if (( ${#cmake_prefix_parts[@]} > 0 )); then
  IFS=';'
  cmake_prefix_path="${cmake_prefix_parts[*]}"
  unset IFS
fi

mkdir -p "${BUILD_ROOT}"

for example in "${examples[@]}"; do
  source_dir="${EXAMPLES_DIR}/${example}"
  build_dir="${BUILD_ROOT}/${ENVIRONMENT}/${example}"
  if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
    echo "Example fixture not found: ${source_dir}" >&2
    exit 1
  fi

  echo "[docs-examples] building ${example} (${ENVIRONMENT})"
  rm -rf "${build_dir}"
  cmake_args=(
    -S "${source_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
  )
  if [[ -n "${cmake_prefix_path}" ]]; then
    cmake_args+=("-DCMAKE_PREFIX_PATH=${cmake_prefix_path}")
  fi
  cmake "${cmake_args[@]}"
  cmake --build "${build_dir}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-3}"
done

echo "[docs-examples] all requested examples built successfully."

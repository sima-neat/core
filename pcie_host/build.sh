#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build"
BUILD_TYPE="Release"
BUILD_TESTS="OFF"
BUILD_EXAMPLES="OFF"
CLEAN_BUILD="OFF"
MAKE_DEB="ON"
PACKAGE_DIR="${SCRIPT_DIR}/packaging"
PLUGIN_NAME="libgstneatpciehost.so"
TENSOR_META_HEADER="gst/SimaTensorSetMetaAbi.h"

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --clean             Remove build directory before configure
  --with-tests        Build unit tests
  --with-examples     Build OpenCV examples (does not install into DEB)
  --no-deb            Skip DEB package generation
  --build-dir <dir>   Build directory (default: build)
  --debug             Build type Debug (default: Release)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN_BUILD="ON"
      shift
      ;;
    --with-tests)
      BUILD_TESTS="ON"
      shift
      ;;
    --with-examples)
      BUILD_EXAMPLES="ON"
      shift
      ;;
    --no-deb)
      MAKE_DEB="OFF"
      shift
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      if [[ -z "${BUILD_DIR}" ]]; then
        echo "ERROR: --build-dir requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

detect_multiarch() {
  if command -v dpkg-architecture >/dev/null 2>&1; then
    dpkg-architecture -qDEB_HOST_MULTIARCH
    return
  fi

  case "$(uname -m)" in
    x86_64)
      echo "x86_64-linux-gnu"
      ;;
    aarch64|arm64)
      echo "aarch64-linux-gnu"
      ;;
    *)
      echo "ERROR: cannot determine Debian multiarch for $(uname -m)" >&2
      exit 1
      ;;
  esac
}

HOST_MULTIARCH="$(detect_multiarch)"
case "${BUILD_DIR}" in
  /*)
    BUILD_DIR_ABS="${BUILD_DIR}"
    ;;
  *)
    BUILD_DIR_ABS="${SCRIPT_DIR}/${BUILD_DIR}"
    ;;
esac
PLUGIN_SOURCE="${SCRIPT_DIR}/artifacts/${HOST_MULTIARCH}/${PLUGIN_NAME}"
HEADER_SOURCE="${SCRIPT_DIR}/artifacts/include/${TENSOR_META_HEADER}"
PLUGIN_STAGE_DIR="${BUILD_DIR_ABS}/artifacts/neatpciehost/${HOST_MULTIARCH}"
PLUGIN_STAGE="${PLUGIN_STAGE_DIR}/${PLUGIN_NAME}"
INCLUDE_STAGE_DIR="${BUILD_DIR_ABS}/artifacts/neatpciehost/${HOST_MULTIARCH}/include"
HEADER_STAGE="${INCLUDE_STAGE_DIR}/${TENSOR_META_HEADER}"

echo "========================================"
echo " SiMa NEAT PCIe host build configuration"
echo "========================================"
echo "Build type      : ${BUILD_TYPE}"
echo "Build tests     : ${BUILD_TESTS}"
echo "Build examples  : ${BUILD_EXAMPLES}"
echo "Generate DEB    : ${MAKE_DEB}"
echo "Clean build     : ${CLEAN_BUILD}"
echo "Build dir       : ${BUILD_DIR}"
echo "Build dir abs   : ${BUILD_DIR_ABS}"
echo "Host multiarch  : ${HOST_MULTIARCH}"
echo "Plugin source   : ${PLUGIN_SOURCE}"
echo "Plugin stage    : ${PLUGIN_STAGE}"
echo "Header source   : ${HEADER_SOURCE}"
echo "Header stage    : ${HEADER_STAGE}"
echo "========================================"
echo

if [[ "${CLEAN_BUILD}" == "ON" ]]; then
  echo "Cleaning build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

if [[ -f "${PLUGIN_SOURCE}" ]]; then
  mkdir -p "${PLUGIN_STAGE_DIR}"
  cp -f "${PLUGIN_SOURCE}" "${PLUGIN_STAGE}"
elif [[ "${MAKE_DEB}" == "ON" ]]; then
  echo "ERROR: missing neatpciehost plugin artifact: ${PLUGIN_SOURCE}" >&2
  echo "       expected layout: artifacts/${HOST_MULTIARCH}/${PLUGIN_NAME}" >&2
  exit 1
else
  echo "WARN: missing neatpciehost plugin artifact: ${PLUGIN_SOURCE}" >&2
  PLUGIN_STAGE=""
fi

if [[ -f "${HEADER_SOURCE}" ]]; then
  mkdir -p "$(dirname "${HEADER_STAGE}")"
  cp -f "${HEADER_SOURCE}" "${HEADER_STAGE}"
else
  echo "ERROR: missing neatpciehost tensor metadata ABI header: ${HEADER_SOURCE}" >&2
  echo "       expected layout: artifacts/include/${TENSOR_META_HEADER}" >&2
  exit 1
fi

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_LIBDIR="lib/${HOST_MULTIARCH}" \
  -DSIMAPCIE_BUILD_TESTS="${BUILD_TESTS}" \
  -DSIMAPCIE_BUILD_EXAMPLES="${BUILD_EXAMPLES}" \
  -DSIMAPCIE_NEATPCIEHOST_PLUGIN="${PLUGIN_STAGE}" \
  -DSIMAPCIE_NEATPCIEHOST_INCLUDE_DIR="${INCLUDE_STAGE_DIR}"

cmake --build "${BUILD_DIR}" -j 2

if [[ "${MAKE_DEB}" == "ON" ]]; then
  echo
  echo "Building DEB packages..."
  mkdir -p "${PACKAGE_DIR}"
  cpack --config "${BUILD_DIR}/CPackConfig.cmake" -G DEB -B "${PACKAGE_DIR}"
fi

echo
echo "========================================"
echo " Build completed successfully"
echo "========================================"
ls -lh "${PACKAGE_DIR}"/*.deb 2>/dev/null || true

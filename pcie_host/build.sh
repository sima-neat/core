#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build"
BUILD_TYPE="Release"
BUILD_TESTS="OFF"
BUILD_EXAMPLES="OFF"
CLEAN_BUILD="OFF"
MAKE_DEB="ON"
INSTALL_DEPS_ONLY="OFF"
PACKAGE_DIR="${SCRIPT_DIR}/dist"
INSTALLER_SOURCE="${SCRIPT_DIR}/scripts/install_pciehost.sh"
INSTALLER_STAGE="${PACKAGE_DIR}/install_pciehost.sh"
PLUGIN_NAME="libgstneatpciehost.so"
TENSOR_META_HEADER="gst/SimaTensorSetMetaAbi.h"
DEPS_MANIFEST="${SIMAPCIE_DEPS_MANIFEST:-${CORE_ROOT}/deps/manifest.json}"
ARTIFACT_REPOSITORY="${SIMAPCIE_PCIE_HOST_ARTIFACT_REPOSITORY:-internals}"
ARTIFACT_MANIFEST_KEY="internals"
VULCAN_ENV="${SIMAPCIE_VULCAN_ENV:-${NEAT_VULCAN_ENV:-production}}"
VULCAN_BASE_URL="${SIMAPCIE_VULCAN_BASE_URL:-${NEAT_VULCAN_BASE_URL:-}}"
ARTIFACT_REQUESTED_REF=""
ARTIFACT_RESOLVED_REF=""
ARTIFACT_SNAP_POLICY="OFF"
ARTIFACT_SNAP_TAG_POLICY="OFF"

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --clean             Remove build directory before configure
  --with-tests        Build unit and hardware smoke tests
  --with-examples     Build OpenCV examples (does not install into DEB)
  --no-deb            Skip DEB package generation
  --install-deps-only Install PCIe host build dependencies, then exit
  --build-dir <dir>   Build directory (default: build)
  --debug             Build type Debug (default: Release)
  -h, --help          Show this help
EOF
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  echo "ERROR: root privileges or sudo are required to run: $*" >&2
  return 1
}

install_build_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: --install-deps-only currently supports apt-based hosts only." >&2
    return 1
  fi

  local packages=(
    build-essential
    ca-certificates
    cmake
    curl
    dpkg-dev
    file
    pkg-config
    tar
    libglib2.0-dev
    libgstreamer1.0-dev
    libgstreamer-plugins-base1.0-dev
    nlohmann-json3-dev
  )
  if [[ "${BUILD_TESTS}" == "ON" ]]; then
    packages+=(libopencv-dev)
  fi

  run_privileged apt-get update
  run_privileged apt-get install -y --no-install-recommends "${packages[@]}"
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
    --install-deps-only)
      INSTALL_DEPS_ONLY="ON"
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

if [[ "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
  install_build_deps
  exit 0
fi

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
     git -C "${CORE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${CORE_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null
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
     git -C "${CORE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${CORE_ROOT}" describe --tags --exact-match HEAD 2>/dev/null || true
    return 0
  fi
  printf '\n'
}

resolve_artifact_ref() {
  ARTIFACT_SNAP_POLICY="OFF"
  ARTIFACT_SNAP_TAG_POLICY="OFF"

  if [[ ! -f "${DEPS_MANIFEST}" ]]; then
    echo "ERROR: missing dependency manifest: ${DEPS_MANIFEST}" >&2
    return 1
  fi

  local manifest_spec
  if ! manifest_spec="$(manifest_dependency_spec "${ARTIFACT_MANIFEST_KEY}" "${DEPS_MANIFEST}")"; then
    return 1
  fi

  local branch spec tag
  if [[ "${manifest_spec}" == "__SNAP__" ]]; then
    ARTIFACT_SNAP_POLICY="ON"
    tag="$(current_core_tag)"
    if [[ -n "${tag}" ]]; then
      ARTIFACT_SNAP_TAG_POLICY="ON"
      branch="${tag}"
    else
      branch="$(current_core_branch)"
      if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        echo "Could not determine current branch for PCIe host artifact snap; using develop." >&2
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

  ARTIFACT_REQUESTED_REF="${branch}:${spec}"
}

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

deb_arch_from_multiarch() {
  case "$1" in
    x86_64-linux-gnu)
      echo "amd64"
      ;;
    aarch64-linux-gnu)
      echo "arm64"
      ;;
    *)
      echo "ERROR: cannot determine Debian architecture for multiarch '$1'" >&2
      exit 1
      ;;
  esac
}

package_version() {
  python3 - "${DEPS_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
print(json.loads(manifest.read_text(encoding="utf-8"))["package-version"])
PY
}

artifact_available() {
  [[ -f "${PLUGIN_SOURCE}" && -f "${HEADER_SOURCE}" ]]
}

install_artifact_from_vulcan_ref() {
  local ref_spec="$1"
  local ref="${ref_spec%%:*}"
  local spec="${ref_spec#*:}"
  local target="${ARTIFACT_REPOSITORY}/pcie-host/${HOST_MULTIARCH}@${ref}:${spec}"
  local install_dir="${SCRIPT_DIR}/artifacts"
  local sima_cli_args=(neat install --env "${VULCAN_ENV}" -d "${install_dir}")

  if [[ -n "${VULCAN_BASE_URL}" ]]; then
    sima_cli_args+=(--base-url "${VULCAN_BASE_URL}")
  fi
  sima_cli_args+=("${target}")

  mkdir -p "${install_dir}"
  rm -rf \
    "${install_dir}/${HOST_MULTIARCH}" \
    "${install_dir}/pcie-host-artifact-${HOST_MULTIARCH}" \
    "${install_dir}/pcie-host-artifact-${HOST_MULTIARCH}.tar.gz"

  echo "Installing PCIe host artifact ${target} from Vulcan env ${VULCAN_ENV}..."
  if ! sima-cli "${sima_cli_args[@]}"; then
    return 1
  fi

  if ! artifact_available; then
    echo "ERROR: installed PCIe host artifact is incomplete under artifacts/${HOST_MULTIARCH}" >&2
    echo "       expected ${PLUGIN_SOURCE}" >&2
    echo "       expected ${HEADER_SOURCE}" >&2
    return 1
  fi

  ARTIFACT_RESOLVED_REF="${ref}:${spec}"
}

ensure_artifact_downloaded() {
  if ! command -v sima-cli >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required to install PCIe host artifacts." >&2
    echo "       Install sima-cli before running this build." >&2
    exit 1
  fi

  if ! resolve_artifact_ref; then
    exit 1
  fi

  local requested_ref="${ARTIFACT_REQUESTED_REF}"
  if install_artifact_from_vulcan_ref "${requested_ref}"; then
    echo "Using PCIe host artifact ${ARTIFACT_RESOLVED_REF}."
    return 0
  fi

  if [[ "${ARTIFACT_SNAP_POLICY}" == "ON" &&
        "${ARTIFACT_SNAP_TAG_POLICY}" != "ON" &&
        "${requested_ref}" != "develop:latest" ]]; then
    echo "No PCIe host artifact found for '${requested_ref}'; retrying develop:latest." >&2
    if install_artifact_from_vulcan_ref "develop:latest"; then
      echo "Using PCIe host artifact ${ARTIFACT_RESOLVED_REF}."
      return 0
    fi
  fi

  echo "ERROR: failed to install PCIe host artifact for ${requested_ref}" >&2
  exit 1
}

HOST_MULTIARCH="$(detect_multiarch)"
DEB_ARCH="$(deb_arch_from_multiarch "${HOST_MULTIARCH}")"
PACKAGE_VERSION="$(package_version)"
case "${BUILD_DIR}" in
  /*)
    BUILD_DIR_ABS="${BUILD_DIR}"
    ;;
  *)
    BUILD_DIR_ABS="${SCRIPT_DIR}/${BUILD_DIR}"
    ;;
esac
PLUGIN_SOURCE="${SCRIPT_DIR}/artifacts/${HOST_MULTIARCH}/${PLUGIN_NAME}"
HEADER_SOURCE="${SCRIPT_DIR}/artifacts/${HOST_MULTIARCH}/include/${TENSOR_META_HEADER}"
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
echo "Debian arch     : ${DEB_ARCH}"
echo "Package version : ${PACKAGE_VERSION}"
echo "Artifact repo   : ${ARTIFACT_REPOSITORY}"
echo "Vulcan env      : ${VULCAN_ENV}"
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

ensure_artifact_downloaded

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
  echo "       expected layout: artifacts/${HOST_MULTIARCH}/include/${TENSOR_META_HEADER}" >&2
  exit 1
fi

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_LIBDIR="lib/${HOST_MULTIARCH}" \
  -DSIMAPCIE_BUILD_TESTS="${BUILD_TESTS}" \
  -DSIMAPCIE_BUILD_HARDWARE_TESTS="${BUILD_TESTS}" \
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

if [[ -f "${INSTALLER_SOURCE}" ]]; then
  cp -f "${INSTALLER_SOURCE}" "${INSTALLER_STAGE}"
  chmod 0755 "${INSTALLER_STAGE}"
fi

if [[ "${BUILD_TESTS}" == "ON" ]]; then
  echo
  echo "Building PCIe host extras archive..."
  extras_name="sima-pcie-host-${PACKAGE_VERSION}-Linux-${DEB_ARCH}-extras"
  extras_dir="${PACKAGE_DIR}/${extras_name}"
  extras_tar="${PACKAGE_DIR}/${extras_name}.tar.gz"
  rm -rf "${extras_dir}" "${extras_tar}" "${extras_tar}.sha256"
  mkdir -p "${PACKAGE_DIR}"
  cmake --install "${BUILD_DIR_ABS}" \
    --component PcieHostExtras \
    --prefix "${extras_dir}"
  tar -C "${extras_dir}" -czf "${extras_tar}" .
  sha256sum "${extras_tar}" > "${extras_tar}.sha256"
  rm -rf "${extras_dir}"
fi

echo
echo "========================================"
echo " Build completed successfully"
echo "========================================"
ls -lh "${PACKAGE_DIR}"/*.deb 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/*-extras.tar.gz "${PACKAGE_DIR}"/*-extras.tar.gz.sha256 2>/dev/null || true
ls -lh "${INSTALLER_STAGE}" 2>/dev/null || true

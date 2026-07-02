#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build"
BUILD_TYPE="Release"
BUILD_TESTS="OFF"
BUILD_PYTHON="OFF"
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
SIMAPCIE_PYTHON_EXECUTABLE="${SIMAPCIE_PYTHON_EXECUTABLE:-}"

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --all               Build tests, Python bindings, and DEB packages
  --clean             Remove build directory before configure
  --with-tests        Build unit and hardware smoke tests
  --python            Build Python bindings and package a Python wheel
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

  local pkg_config_provider="pkg-config"
  if apt-cache show pkgconf >/dev/null 2>&1; then
    # GitHub-hosted Ubuntu 22.04 runners already use pkgconf as the pkg-config
    # provider. Keep that provider instead of forcing the older pkg-config
    # package, which removes pkgconf and can break module discovery on the image.
    pkg_config_provider="pkgconf"
  fi

  local packages=(
    build-essential
    ca-certificates
    cmake
    curl
    dpkg-dev
    file
    "${pkg_config_provider}"
    tar
    libunwind-dev
    libglib2.0-dev
    libgstreamer1.0-dev
    libgstreamer-plugins-base1.0-dev
    nlohmann-json3-dev
  )
  if [[ "${BUILD_TESTS}" == "ON" ]]; then
    packages+=(libopencv-dev)
  fi
  if [[ "${BUILD_PYTHON}" == "ON" ]]; then
    packages+=(
      python3
      python3-dev
      python3-numpy
      python3-pip
      python3-pytest
      python3-venv
    )
  fi

  run_privileged apt-get update
  run_privileged apt-get install -y --no-install-recommends "${packages[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      BUILD_TESTS="ON"
      BUILD_PYTHON="ON"
      MAKE_DEB="ON"
      shift
      ;;
    --clean)
      CLEAN_BUILD="ON"
      shift
      ;;
    --with-tests)
      BUILD_TESTS="ON"
      shift
      ;;
    --python)
      BUILD_PYTHON="ON"
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

resolve_python_executable() {
  if [[ -n "${SIMAPCIE_PYTHON_EXECUTABLE}" ]]; then
    printf '%s\n' "${SIMAPCIE_PYTHON_EXECUTABLE}"
    return
  fi
  if [[ -x /usr/bin/python3 ]]; then
    printf '%s\n' /usr/bin/python3
    return
  fi
  command -v python3
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

generate_package_metadata() {
  mkdir -p "${PACKAGE_DIR}"
  rm -f "${PACKAGE_DIR}/metadata.json" "${PACKAGE_DIR}/metadata-pypciehost.json"

  local git_commit="${GITHUB_SHA:-}"
  if [[ -z "${git_commit}" ]] &&
     command -v git >/dev/null 2>&1 &&
     git -C "${CORE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git -C "${CORE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  fi
  git_commit="${git_commit:-unknown}"

  local git_branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
  if [[ -z "${git_branch}" ]]; then
    git_branch="$(current_core_branch)"
  fi
  git_branch="${git_branch:-unknown}"

  echo
  echo "Writing PCIe host package metadata..."
  PACKAGE_DIR="${PACKAGE_DIR}" \
  DEB_ARCH="${DEB_ARCH}" \
  PACKAGE_VERSION="${PACKAGE_VERSION}" \
  GIT_COMMIT="${git_commit}" \
  GIT_BRANCH="${git_branch}" \
  python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

package_dir = Path(os.environ["PACKAGE_DIR"])
deb_arch = os.environ["DEB_ARCH"]
version = os.environ["PACKAGE_VERSION"]
commit = os.environ["GIT_COMMIT"]
branch = os.environ["GIT_BRANCH"]
commit_folder = commit[:12] if commit != "unknown" else version


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_entry(path: Path) -> tuple[str, str, int]:
    checksum = sha256_file(path)
    if path.suffix != ".sh":
        path.with_name(path.name + ".sha256").write_text(
            f"{checksum}  {path.name}\n", encoding="utf-8")
    return path.name, checksum, path.stat().st_size


runtime_debs = sorted(
    path for path in package_dir.glob(f"sima-pcie-host_*_{deb_arch}.deb")
    if not path.name.startswith("sima-pcie-host-dev_"))
dev_debs = sorted(package_dir.glob(f"sima-pcie-host-dev_*_{deb_arch}.deb"))
extras_tars = sorted(package_dir.glob(f"sima-pcie-host-*-Linux-{deb_arch}-extras.tar.gz"))
wheels = sorted(package_dir.glob("pypciehost-*.whl"))
installer = package_dir / "install_pciehost.sh"

full_paths = runtime_debs + dev_debs + extras_tars + wheels
if installer.is_file():
    full_paths.append(installer)

if runtime_debs and full_paths:
    resources = []
    resource_checksums = {}
    resource_sizes = {}
    total_size = 0
    for path in full_paths:
        resource, checksum, size = resource_entry(path)
        resources.append(resource)
        resource_checksums[resource] = checksum
        resource_sizes[resource] = size
        total_size += size

    install_script = "./install_pciehost.sh --python" if wheels else "./install_pciehost.sh"
    post_message = "[bold]sima-pcie-host installed successfully.[/bold]\n"
    if wheels:
        post_message = (
            "[bold]sima-pcie-host and pypciehost installed successfully.[/bold]\n"
            "Activate pypciehost with: source ~/pypciehost/bin/activate\n"
        )

    metadata = {
        "name": f"gh:sima-neat/pciehost/{deb_arch}",
        "version": commit_folder,
        "release": "",
        "description": (
            f"SiMa.ai NEAT PCIe host runtime, development, and Python packages ({deb_arch})"
            if wheels else
            f"SiMa.ai NEAT PCIe host runtime and development packages ({deb_arch})"
        ),
        "platforms": [{
            "type": "host",
            "os": ["linux"],
            "arch": [deb_arch],
        }],
        "resources": resources,
        "resources-checksum": resource_checksums,
        "selectable-resources": [],
        "installation": {
            "script": install_script,
            "post-message": post_message,
        },
        "artifact": {
            "type": "debian-packages",
            "repository": "core",
            "package_path": f"pciehost/{deb_arch}",
            "arches": [deb_arch],
        },
        "repository": os.environ.get("GITHUB_REPOSITORY", "sima-neat/core"),
        "branch": branch,
        "commit": commit,
        "commit_folder": commit_folder,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "size": {
            "download": f"{max(1, (total_size + 1024 * 1024 - 1) // (1024 * 1024))}MB",
            "install": f"{max(1, (total_size + 1024 * 1024 - 1) // (1024 * 1024))}MB",
        },
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if len(wheels) == 1:
        wheel = wheels[0].name
        pypciehost_metadata = dict(metadata)
        pypciehost_metadata.update({
            "name": f"gh:sima-neat/pypciehost/{deb_arch}",
            "description": f"pypciehost wheel for SiMa.ai NEAT PCIe host ({deb_arch})",
            "resources": [wheel],
            "resources-checksum": {wheel: resource_checksums[wheel]},
            "installation": {
                "script": ":",
                "post-message": (
                    "[bold]pypciehost wheel downloaded.[/bold]\n"
                    f"Install it with: python3 -m pip install ./{wheel}\n"
                    "pypciehost requires the matching sima-pcie-host runtime package. "
                    f"If it is not installed yet, run: sima-cli neat install core/pciehost/{deb_arch}\n"
                ),
            },
            "artifact": {
                "type": "python-wheel",
                "repository": "core",
                "package_path": f"pciehost/{deb_arch}",
                "arches": [deb_arch],
            },
            "size": {
                "download": f"{max(1, (resource_sizes[wheel] + 1024 * 1024 - 1) // (1024 * 1024))}MB",
                "install": "1MB",
            },
        })
        (package_dir / "metadata-pypciehost.json").write_text(
            json.dumps(pypciehost_metadata, indent=2) + "\n", encoding="utf-8")
PY
}

HOST_MULTIARCH="$(detect_multiarch)"
DEB_ARCH="$(deb_arch_from_multiarch "${HOST_MULTIARCH}")"
PACKAGE_VERSION="$(package_version)"
PYTHON_EXECUTABLE="$(resolve_python_executable)"
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
echo "Build Python    : ${BUILD_PYTHON}"
echo "Generate DEB    : ${MAKE_DEB}"
echo "Clean build     : ${CLEAN_BUILD}"
echo "Build dir       : ${BUILD_DIR}"
echo "Build dir abs   : ${BUILD_DIR_ABS}"
echo "Host multiarch  : ${HOST_MULTIARCH}"
echo "Debian arch     : ${DEB_ARCH}"
echo "Package version : ${PACKAGE_VERSION}"
if [[ "${BUILD_PYTHON}" == "ON" ]]; then
  echo "Python exe      : ${PYTHON_EXECUTABLE}"
fi
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
  -DSIMAPCIE_BUILD_PYTHON="${BUILD_PYTHON}" \
  -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" \
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

if [[ "${BUILD_PYTHON}" == "ON" ]]; then
  echo
  echo "Building PCIe host Python wheel..."
  python_stage="${BUILD_DIR_ABS}/python/stage"
  rm -f "${PACKAGE_DIR}"/pypciehost-"${PACKAGE_VERSION}"-*.whl \
    "${PACKAGE_DIR}"/pypciehost-"${PACKAGE_VERSION}"-*.whl.sha256 \
    "${PACKAGE_DIR}"/simapciehost-"${PACKAGE_VERSION}"-*.whl \
    "${PACKAGE_DIR}"/simapciehost-"${PACKAGE_VERSION}"-*.whl.sha256
  rm -rf "${python_stage}"
  mkdir -p "${PACKAGE_DIR}"
  cmake --install "${BUILD_DIR_ABS}" \
    --component PcieHostPython \
    --prefix "${python_stage}"
  wheel_path="$("${PYTHON_EXECUTABLE}" "${SCRIPT_DIR}/python/build_wheel.py" \
    --stage-dir "${python_stage}" \
    --output-dir "${PACKAGE_DIR}" \
    --version "${PACKAGE_VERSION}")"
  sha256sum "${wheel_path}" > "${wheel_path}.sha256"
fi

generate_package_metadata

echo
echo "========================================"
echo " Build completed successfully"
echo "========================================"
ls -lh "${PACKAGE_DIR}"/*.deb 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/*-extras.tar.gz "${PACKAGE_DIR}"/*-extras.tar.gz.sha256 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/pypciehost-*.whl 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/metadata*.json 2>/dev/null || true
ls -lh "${INSTALLER_STAGE}" 2>/dev/null || true

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/host-platform.sh
source "${SCRIPT_DIR}/scripts/host-platform.sh"
cd "${SCRIPT_DIR}"

# Reject the retired unverified header override before dependency installation.
if [[ -n "${SIMAPCIE_SVC_INCLUDE_DIR:-}" ]]; then
  echo "ERROR: SIMAPCIE_SVC_INCLUDE_DIR is no longer accepted. Install Scheduler's pcie_service_headers component using a reviewed SIMA_SCHEDULER_PCIE_SERVICE_HEADERS_DEB (or explicit legacy header cohort), then set SimaScheduler_DIR to that installed host profile." >&2
  exit 2
fi

BUILD_DIR="build"
BUILD_TYPE="Release"
BUILD_TESTS="OFF"
BUILD_PYTHON="OFF"
BUILD_EXAMPLES="OFF"
BUILD_TUTORIALS="ON"
CLEAN_BUILD="OFF"
INSTALL_DEPS_ONLY="OFF"
MAKE_DEB="ON"
PACKAGE_DIR="${SCRIPT_DIR}/dist"
INSTALLER_SOURCE="${SCRIPT_DIR}/scripts/install_pciehost.sh"
INSTALLER_STAGE="${PACKAGE_DIR}/install_pciehost.sh"
DEPS_MANIFEST="${SIMAPCIE_DEPS_MANIFEST:-${CORE_ROOT}/deps/manifest.json}"
SIMAPCIE_PYTHON_EXECUTABLE="${SIMAPCIE_PYTHON_EXECUTABLE:-}"
SCHEDULER_SPEC=""
SCHEDULER_ENVIRONMENT="production"
SCHEDULER_SELECTION=""

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --all               Build tests, Python bindings, and DEB packages
  --clean             Remove build directory before configure
  --with-tests        Build unit and hardware tests
  --with-examples     Build PCIe host C++ examples
  --python            Build Python bindings and package a Python wheel
  --install-deps-only  Install native build prerequisites, then exit without building
  --no-deb            Skip DEB package generation
  --scheduler-spec <sha|latest> Override the manifest's Scheduler revision
  --scheduler-environment <name> Publication environment (default: production)
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
    echo "ERROR: automatic dependency installation supports apt-based hosts only." >&2
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
    zlib1g-dev
  )
  if [[ "${BUILD_TESTS}" == "ON" || "${BUILD_TUTORIALS}" == "ON" ]]; then
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

  # Match the native profiles before making any package changes.
  source /etc/os-release
  [[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]]
  run_privileged apt-get update
  run_privileged apt-get install -y --no-install-recommends "${packages[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      BUILD_TESTS="ON"
      BUILD_PYTHON="ON"
      BUILD_EXAMPLES="ON"
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
    --with-examples)
      BUILD_EXAMPLES="ON"
      shift
      ;;
    --python)
      BUILD_PYTHON="ON"
      shift
      ;;
    --install-deps-only)
      INSTALL_DEPS_ONLY="ON"
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
    --scheduler-spec|--scheduler-environment)
      option="$1"
      [[ -n "${2:-}" ]] || { echo "ERROR: $option requires a value" >&2; exit 2; }
      if [[ "$option" == --scheduler-spec ]]; then SCHEDULER_SPEC="$2"; else SCHEDULER_ENVIRONMENT="$2"; fi
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

# Refuse inherited SDK flags rather than silently mixing host and target ABIs.
for name in CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_TOOLCHAIN_FILE PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_LIBDIR PKG_CONFIG_PATH CMAKE_LIBRARY_PATH CMAKE_INCLUDE_PATH CMAKE_PREFIX_PATH CMAKE_FIND_ROOT_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
  [[ -z "${!name:-}" ]] || { echo "ERROR: native PCIe build requires clean host environment ($name is set)" >&2; exit 2; }
done
install_build_deps
if [[ "${INSTALL_DEPS_ONLY}" == "ON" ]]; then
  echo "Native PCIe host build prerequisites are ready; no PCIe package or device installation requested."
  exit 0
fi

current_core_branch() {
  printf '%s\n' "${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-$(git -C "$CORE_ROOT" branch --show-current)}}"
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



generate_package_metadata() {
  mkdir -p "${PACKAGE_DIR}"
  rm -f "${PACKAGE_DIR}/metadata.json" \
    "${PACKAGE_DIR}/metadata-extras.json" \
    "${PACKAGE_DIR}/metadata-pyneatpcie.json"

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
  HOST_DISTRO="${HOST_DISTRO}" \
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
host_distro = os.environ["HOST_DISTRO"]
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
wheels = sorted(package_dir.glob("pyneatpcie-*.whl"))
installer = package_dir / "install_pciehost.sh"

selection = json.loads((package_dir / "scheduler-native-selection.json").read_text())
native_paths = [package_dir / name for name in selection["resources"]]
full_paths = runtime_debs + dev_debs + extras_tars + wheels + native_paths + [
    package_dir / "scheduler-native-selection.json", package_dir / "pcie_delivery.py"]
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

    install_script = (
        "bash ./install_pciehost.sh --python" if wheels else
        "bash ./install_pciehost.sh"
    )
    post_message = "[bold]sima-pcie-host installed successfully.[/bold]\n"
    if wheels:
        post_message = (
            "[bold]sima-pcie-host and pyneatpcie installed successfully.[/bold]\n"
            "Activate pyneatpcie with: source ~/pyneatpcie/bin/activate\n"
        )

    metadata = {
        "name": f"gh:sima-neat/pciehost/{host_distro}/{deb_arch}",
        "version": commit_folder,
        "release": "",
        "description": (
            "SiMa.ai NEAT PCIe host runtime, development, and Python packages "
            f"({host_distro}, {deb_arch})"
            if wheels else
            "SiMa.ai NEAT PCIe host runtime and development packages "
            f"({host_distro}, {deb_arch})"
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
            "package_path": f"pciehost/{host_distro}/{deb_arch}",
            "distro": host_distro,
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

    if len(extras_tars) == 1:
        extras = extras_tars[0].name
        extracted_dir = extras.removesuffix(".tar.gz")
        extras_metadata = dict(metadata)
        extras_metadata.update({
            "description": (
                "SiMa.ai NEAT PCIe host tutorial extras "
                f"({host_distro}, {deb_arch})"
            ),
            "resources": [extras],
            "resources-checksum": {extras: resource_checksums[extras]},
            "selectable-resources": [],
            "installation": {
                "script": ":",
                "post-message": (
                    "[bold]PCIe host tutorial extras downloaded and extracted.[/bold]\n"
                    f"Enter the tutorial directory with: cd {extracted_dir}\n"
                    "The matching PCIe host package must already be installed.\n"
                ),
            },
            "artifact": {
                "type": "tutorial-extras",
                "repository": "core",
                "package_path": f"pciehost/{host_distro}/{deb_arch}",
                "distro": host_distro,
                "arches": [deb_arch],
            },
            "size": {
                "download": f"{max(1, (resource_sizes[extras] + 1024 * 1024 - 1) // (1024 * 1024))}MB",
                "install": f"{max(1, (resource_sizes[extras] + 1024 * 1024 - 1) // (1024 * 1024))}MB",
            },
        })
        (package_dir / "metadata-extras.json").write_text(
            json.dumps(extras_metadata, indent=2) + "\n", encoding="utf-8")

    if len(wheels) == 1:
        wheel = wheels[0].name
        pyneatpcie_metadata = dict(metadata)
        pyneatpcie_metadata.update({
            "name": f"gh:sima-neat/pyneatpcie/{host_distro}/{deb_arch}",
            "description": (
                "pyneatpcie wheel for SiMa.ai NEAT PCIe host "
                f"({host_distro}, {deb_arch})"
            ),
            "resources": [wheel],
            "resources-checksum": {wheel: resource_checksums[wheel]},
            "installation": {
                "script": ":",
                "post-message": (
                    "[bold]pyneatpcie wheel downloaded.[/bold]\n"
                    f"Install it with: python3 -m pip install ./{wheel}\n"
                    "pyneatpcie requires the matching sima-pcie-host runtime package. "
                    "If it is not installed yet, run: sima-cli neat install "
                    f"core/pciehost/{host_distro}/{deb_arch}\n"
                ),
            },
            "artifact": {
                "type": "python-wheel",
                "repository": "core",
                "package_path": f"pciehost/{host_distro}/{deb_arch}",
                "distro": host_distro,
                "arches": [deb_arch],
            },
            "size": {
                "download": f"{max(1, (resource_sizes[wheel] + 1024 * 1024 - 1) // (1024 * 1024))}MB",
                "install": "1MB",
            },
        })
        (package_dir / "metadata-pyneatpcie.json").write_text(
            json.dumps(pyneatpcie_metadata, indent=2) + "\n", encoding="utf-8")
PY
}

HOST_DISTRO="$(simapcie_detect_host_distro)"
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

echo "========================================"
echo " SiMa NEAT PCIe host build configuration"
echo "========================================"
echo "Build type      : ${BUILD_TYPE}"
echo "Build tests     : ${BUILD_TESTS}"
echo "Build Python    : ${BUILD_PYTHON}"
echo "Build examples  : ${BUILD_EXAMPLES}"
echo "Build tutorials : ${BUILD_TUTORIALS}"
echo "Generate DEB    : ${MAKE_DEB}"
echo "Clean build     : ${CLEAN_BUILD}"
echo "Build dir       : ${BUILD_DIR}"
echo "Build dir abs   : ${BUILD_DIR_ABS}"
echo "Host distro     : ${HOST_DISTRO}"
echo "Host multiarch  : ${HOST_MULTIARCH}"
echo "Debian arch     : ${DEB_ARCH}"
echo "Package version : ${PACKAGE_VERSION}"
if [[ "${BUILD_PYTHON}" == "ON" ]]; then
  echo "Python exe      : ${PYTHON_EXECUTABLE}"
fi
echo "========================================"
echo

if [[ "${CLEAN_BUILD}" == "ON" ]]; then
  echo "Cleaning build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# Core acquires Scheduler's native publication under deps/manifest.json policy.
# An explicitly installed local component remains usable for a no-DEB build.
GENAI_CMAKE_ARGS=()
if [[ "$MAKE_DEB" == OFF && -n "${SimaScheduler_DIR:-}" ]]; then
  GENAI_CMAKE_ARGS+=("-DSimaScheduler_DIR=${SimaScheduler_DIR}")
else
  native_root="$(mktemp -d "${TMPDIR:-/tmp}/core-native-scheduler.XXXXXX")"
  scheduler_spec_args=()
  if [[ -n "$SCHEDULER_SPEC" ]]; then scheduler_spec_args+=(--spec "$SCHEDULER_SPEC"); fi
  python3 scripts/acquire_scheduler.py --branch "$(current_core_branch)" \
    "${scheduler_spec_args[@]}" --environment "$SCHEDULER_ENVIRONMENT" \
    --distro "$HOST_DISTRO" --arch "$DEB_ARCH" --output "$native_root/selected"
  SCHEDULER_SELECTION="$native_root/selected/selection.json"
  # Acquisition stages the authentic Scheduler publication and its external
  # userspace provider privately. Building Core must not install a PCIe driver
  # or execute the Platform package's maintainer scripts on the build host.
  scheduler_dir="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["scheduler_dir"])' "$SCHEDULER_SELECTION")"
  native_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["package_version"])' "$SCHEDULER_SELECTION")"
  GENAI_CMAKE_ARGS+=("-DSimaScheduler_DIR=${scheduler_dir}"
                    "-DSIMAPCIE_SCHEDULER_PACKAGE_VERSION=${native_version}")
fi
cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_LIBDIR="lib/${HOST_MULTIARCH}" \
  -DSIMAPCIE_BUILD_TESTS="${BUILD_TESTS}" \
  -DSIMAPCIE_BUILD_HARDWARE_TESTS="${BUILD_TESTS}" \
  -DSIMAPCIE_BUILD_EXAMPLES="${BUILD_EXAMPLES}" \
  -DSIMAPCIE_BUILD_TUTORIALS="${BUILD_TUTORIALS}" \
  -DSIMAPCIE_BUILD_PYTHON="${BUILD_PYTHON}" \
  -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  "${GENAI_CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}" -j 2

if [[ "${MAKE_DEB}" == "ON" ]]; then
  echo
  echo "Building DEB packages..."
  mkdir -p "${PACKAGE_DIR}"
  cpack --config "${BUILD_DIR}/CPackConfig.cmake" -G DEB -B "${PACKAGE_DIR}"
  python3 scripts/pcie_delivery.py compose --selection "$SCHEDULER_SELECTION" --output "$PACKAGE_DIR"
  python3 scripts/pcie_delivery.py verify --directory "$PACKAGE_DIR"
fi

if [[ -f "${INSTALLER_SOURCE}" ]]; then
  mkdir -p "${PACKAGE_DIR}"
  cp -f "${INSTALLER_SOURCE}" "${INSTALLER_STAGE}"
  chmod 0755 "${INSTALLER_STAGE}"
fi

if [[ "${BUILD_TESTS}" == "ON" || "${BUILD_TUTORIALS}" == "ON" ]]; then
  echo
  echo "Building PCIe host extras archive..."
  extras_name="sima-pcie-host-${PACKAGE_VERSION}-Linux-${DEB_ARCH}-extras"
  extras_tar="${PACKAGE_DIR}/${extras_name}.tar.gz"
  extras_stage="$(mktemp -d /tmp/sima-pcie-host-extras-stage.XXXXXX)"
  extras_dir="${extras_stage}/prefix"
  trap 'rm -rf "${extras_stage}"' EXIT
  rm -f "${extras_tar}" "${extras_tar}.sha256"
  mkdir -p "${PACKAGE_DIR}"
  cmake --install "${BUILD_DIR_ABS}" \
    --component PcieHostExtras \
    --prefix "${extras_dir}"

  if [[ ! -x "${extras_dir}/build.sh" ]]; then
    echo "ERROR: PCIe host extras install tree is missing root build.sh." >&2
    echo "       Expected build.sh next to lib/ and share/." >&2
    exit 1
  fi
  if [[ ! -d "${extras_dir}/share/sima-pcie-host/tutorials" ]]; then
    echo "ERROR: PCIe host extras install tree is missing tutorial sources." >&2
    exit 1
  fi
  for tutorial_asset in labrador.jpg street-scene.png; do
    asset_path="${extras_dir}/share/sima-pcie-host/tutorials/assets/${tutorial_asset}"
    if [[ ! -s "${asset_path}" ]]; then
      echo "ERROR: PCIe host extras install tree is missing tutorial asset ${tutorial_asset}." >&2
      exit 1
    fi
  done

  tar -C "${extras_dir}" -czf "${extras_tar}" .
  sha256sum "${extras_tar}" > "${extras_tar}.sha256"
  rm -rf "${extras_stage}"
  trap - EXIT
fi

if [[ "${BUILD_PYTHON}" == "ON" ]]; then
  echo
  echo "Building PCIe host Python wheel..."
  python_stage="${BUILD_DIR_ABS}/python/stage"
  rm -f "${PACKAGE_DIR}"/pyneatpcie-"${PACKAGE_VERSION}"-*.whl \
    "${PACKAGE_DIR}"/pyneatpcie-"${PACKAGE_VERSION}"-*.whl.sha256 \
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

if [[ "$MAKE_DEB" == ON ]]; then generate_package_metadata; fi

echo
echo "========================================"
echo " Build completed successfully"
echo "========================================"
ls -lh "${PACKAGE_DIR}"/*.deb 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/*-extras.tar.gz "${PACKAGE_DIR}"/*-extras.tar.gz.sha256 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/pyneatpcie-*.whl 2>/dev/null || true
ls -lh "${PACKAGE_DIR}"/metadata*.json 2>/dev/null || true
ls -lh "${INSTALLER_STAGE}" 2>/dev/null || true

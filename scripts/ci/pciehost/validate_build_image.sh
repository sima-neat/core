#!/usr/bin/env bash
# Runs inside the candidate image. No package installation or driver setup.
set -euo pipefail
CORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${CORE_ROOT}"
: "${CORE_VALIDATION_SHA:?Core validation revision is required}"
: "${EXPECTED_UBUNTU:?Ubuntu release is required}"
: "${EXPECTED_ARCH:?Native architecture is required}"
source /etc/os-release
[[ "${ID}" == ubuntu && "${VERSION_ID}" == "${EXPECTED_UBUNTU}" ]]
[[ "$(dpkg --print-architecture)" == "${EXPECTED_ARCH}" ]]
[[ "$(cat /etc/vulcan-build-image)" == "${EXPECTED_UBUNTU} ${EXPECTED_ARCH} media" ]]
git config --global --add safe.directory "${CORE_ROOT}"
[[ "$(git rev-parse HEAD)" == "${CORE_VALIDATION_SHA}" ]]

# Make an accidental return to APT a hard failure during the consumer build.
mkdir -p /tmp/no-apt
printf '#!/bin/sh\necho "APT is forbidden during image validation" >&2\nexit 97\n' > /tmp/no-apt/apt-get
cp /tmp/no-apt/apt-get /tmp/no-apt/apt
chmod +x /tmp/no-apt/apt-get /tmp/no-apt/apt
export PATH="/tmp/no-apt:${PATH}"
./pcie_host/build.sh --all --skip-deps --install-deps-only
./scripts/ci/install_sima_cli_main.sh
export PATH="${HOME}/.sima-cli/.venv/bin:${PATH}"
# The checkout is pinned; this branch selects the intended dependency policy.
export GITHUB_REF_NAME="${CORE_DEPENDENCY_BRANCH:-develop}"
./pcie_host/build.sh --clean --all --skip-deps --jobs "$(nproc)" --scheduler-environment production
ctest --test-dir pcie_host/build --output-on-failure --no-tests=error
python3 pcie_host/dist/pcie_delivery.py verify --directory pcie_host/dist

stage="$(mktemp -d /tmp/pcie-image-smoke.XXXXXX)"
python3 pcie_host/dist/pcie_delivery.py stage --directory pcie_host/dist --destination "${stage}/prefix"
mv "${stage}/prefix" "${stage}/relocated"
prefix="${stage}/relocated"
multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
export LD_LIBRARY_PATH="${prefix}/usr/lib/${multiarch}/neat/runtime:${prefix}/usr/lib"
cmake -S pcie_host/tests/package_smoke -B "${stage}/consumer" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${prefix}/usr"
cmake --build "${stage}/consumer" --parallel "$(nproc)"
"${stage}/consumer/sima_pcie_host_package_smoke"
python3 -m venv "${stage}/venv"
"${stage}/venv/bin/python" -m pip install pcie_host/dist/pyneatpcie-*.whl
"${stage}/venv/bin/python" - <<'PY'
import pyneatpcie
assert pyneatpcie.__version__
for name in ('Model', 'Runtime', 'ModelConfig', 'Completion', 'Tensor'):
    assert getattr(pyneatpcie, name)
PY
printf 'Validated Core %s on Ubuntu %s %s\n' "${CORE_VALIDATION_SHA}" "${EXPECTED_UBUNTU}" "${EXPECTED_ARCH}"

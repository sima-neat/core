#!/usr/bin/env bash
# Provisions the toolchain that builds the amd64 model archive helper.
#
# The Core build runs in an AArch64 SDK container, so producing the amd64 helper needs a cross
# compiler plus the helper's host dependencies. Ubuntu serves AArch64 from ports.ubuntu.com and
# amd64 from archive.ubuntu.com, so each suite must be pinned to its architecture or apt fails to
# resolve either one. The packages are extracted instead of installed because amd64 zlib's
# linux-libc-dev dependency cannot coexist with the SDK's AArch64 package.
set -euo pipefail

AMD64_HELPER_ROOT="${NEAT_AMD64_HELPER_ROOT:-/opt/neat-amd64-helper}"
if command -v x86_64-linux-gnu-g++ >/dev/null 2>&1 &&
   [[ -f "${AMD64_HELPER_ROOT}/usr/include/nlohmann/json.hpp" &&
      -f "${AMD64_HELPER_ROOT}/usr/include/zlib.h" &&
      -e "${AMD64_HELPER_ROOT}/usr/lib/x86_64-linux-gnu/libz.so" ]]; then
  echo "amd64 cross toolchain already present."
  exit 0
fi

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

UBUNTU_SOURCES="/etc/apt/sources.list.d/ubuntu.sources"
if [[ -f "${UBUNTU_SOURCES}" ]] && ! grep -q '^Architectures:' "${UBUNTU_SOURCES}"; then
  ${SUDO} sed -i '/^URIs: http:\/\/ports.ubuntu.com/a Architectures: arm64' "${UBUNTU_SOURCES}"
fi

${SUDO} tee /etc/apt/sources.list.d/amd64.sources >/dev/null <<'EOF'
Types: deb
URIs: http://archive.ubuntu.com/ubuntu
Suites: noble noble-updates
Components: main universe
Architectures: amd64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF

${SUDO} dpkg --add-architecture amd64
${SUDO} apt-get update -qq
${SUDO} apt-get install -y -qq --no-install-recommends \
  g++-x86-64-linux-gnu

package_dir="$(mktemp -d /tmp/neat-amd64-helper.XXXXXX)"
trap 'rm -rf "${package_dir}"' EXIT
(
  cd "${package_dir}"
  apt-get download nlohmann-json3-dev zlib1g:amd64 zlib1g-dev:amd64 >/dev/null
)
${SUDO} mkdir -p "${AMD64_HELPER_ROOT}"
for package in "${package_dir}"/*.deb; do
  ${SUDO} dpkg-deb -x "${package}" "${AMD64_HELPER_ROOT}"
done

x86_64-linux-gnu-g++ --version | head -1

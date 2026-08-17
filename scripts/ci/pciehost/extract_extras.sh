#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

mapfile -t extras_archives < <(
  find "${PACKAGE_DIR}" "${WORKSPACE}" -type f \
    -name 'sima-pcie-host-*-Linux-amd64-extras.tar.gz' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{print $2}'
)
if [[ "${#extras_archives[@]}" -eq 0 ]]; then
  echo "ERROR: unable to locate PCIe host extras archive after install." >&2
  find "${PACKAGE_DIR}" -maxdepth 3 -type f -printf '  %p\n' >&2 || true
  exit 1
fi

extras_tar="${extras_archives[0]}"
echo "Extracting ${extras_tar}"
rm -rf "${EXTRAS_DIR}"
mkdir -p "${EXTRAS_DIR}"
tar -xzf "${extras_tar}" -C "${EXTRAS_DIR}"

mapfile -t ctest_files < <(
  find "${EXTRAS_DIR}" -type f -name CTestTestfile.cmake \
    -path '*/tests/CTestTestfile.cmake' \
    -print | sort
)
if [[ "${#ctest_files[@]}" -eq 0 ]]; then
  echo "ERROR: unable to locate PCIe host hardware CTest files in extras." >&2
  find "${EXTRAS_DIR}" -maxdepth 5 -type f -printf '  %p\n' >&2 || true
  exit 1
fi

TEST_DIR="$(dirname "${ctest_files[0]}")"
export TEST_DIR
write_state_var TEST_DIR "${TEST_DIR}"
echo "Using PCIe host hardware test dir: ${TEST_DIR}"

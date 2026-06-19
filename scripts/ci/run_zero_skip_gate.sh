#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-zero-skip}"
TEST_LABEL_REGEX="${TEST_LABEL_REGEX:-strict}"
JUNIT_PATH="${JUNIT_PATH:-${BUILD_DIR}/zero_skip_junit.xml}"
BUILD_TARGETS="${BUILD_TARGETS:-}"

mkdir -p "${BUILD_DIR}"

echo "[zero-skip] configure ${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}"

echo "[zero-skip] build tests"
if [[ -n "${BUILD_TARGETS}" ]]; then
  cmake --build "${BUILD_DIR}" --target ${BUILD_TARGETS} -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
else
  cmake --build "${BUILD_DIR}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
fi

echo "[zero-skip] run Modalix preflight"
ctest --test-dir "${BUILD_DIR}" -R unit_modalix_contract_preflight_test --output-on-failure

echo "[zero-skip] run ctest labels=${TEST_LABEL_REGEX} exclude=long"
ctest \
  --test-dir "${BUILD_DIR}" \
  --label-regex "${TEST_LABEL_REGEX}" \
  --label-exclude "long" \
  --output-on-failure \
  --output-junit "${JUNIT_PATH}" \
  --no-tests=error

export JUNIT_PATH
python3 - <<'PY'
import os
import sys
import xml.etree.ElementTree as ET

path = os.environ['JUNIT_PATH']
root = ET.parse(path).getroot()

skipped = []
for testcase in root.iter('testcase'):
    if testcase.find('skipped') is not None:
        name = testcase.attrib.get('name', '<unknown>')
        skipped.append(name)

print(f"[zero-skip] skipped_non_long={len(skipped)}")
if skipped:
    for name in skipped:
        print(f"  - {name}")
    sys.exit(1)
PY

echo "[zero-skip] PASS"

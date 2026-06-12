#!/usr/bin/env bash
# Cross-compile the Phase-0 coherency suites for aarch64 and run them on the
# Modalix DevKit via devkit-run. Includes the negative control (CMOs disabled),
# which MUST fail on real silicon — that is the proof the hazard tests have teeth.
#
# Requires: aarch64-linux-gnu-g++, and devkit-run on PATH (board over NFS /workspace).
set -euo pipefail
cd "$(dirname "$0")"

CXX=aarch64-linux-gnu-g++
FLAGS="-std=c++17 -O2 -pthread"
WS_DIR="/workspace/core_graph_changes/tests/coherency"

echo "== cross-compiling for aarch64 =="
$CXX $FLAGS segment_coherency.cpp coherency_host_suite.cpp           -o coherency_host_arm
$CXX $FLAGS segment_coherency.cpp coherency_devkit_suite.cpp   -ldl  -o coherency_devkit_arm
$CXX $FLAGS -DSEGCOH_BREAK_INVALIDATE -DSEGCOH_BREAK_CLEAN \
            segment_coherency.cpp coherency_devkit_suite.cpp   -ldl  -o coherency_devkit_arm_broken

# Files must be owned by uid 1000 so the board's `sima` user can chmod+exec them
# over NFS. (Harmless if you are not root; devkit-run will report otherwise.)
chown 1000:1000 coherency_*_arm coherency_devkit_arm_broken 2>/dev/null || true
chmod 0755       coherency_*_arm coherency_devkit_arm_broken

echo "== host suite on A65 (logic/optimality/fuzzer) =="
devkit-run "$WS_DIR/coherency_host_arm"

echo "== devkit suite on A65 (real cacheable DDR + real civac/cvac + DMA device) =="
devkit-run "$WS_DIR/coherency_devkit_arm"

echo "== NEGATIVE CONTROL (CMOs disabled) — expected: all hazard tests FAIL =="
if devkit-run "$WS_DIR/coherency_devkit_arm_broken"; then
  echo "ERROR: negative control PASSED — tests are not exercising the cache hazard!" >&2
  exit 1
else
  echo "OK: negative control failed as required (hazard tests have teeth)."
fi

#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-codex-graph-sdk}"

ctest --test-dir "${BUILD_DIR}" --output-on-failure \
  -R 'graph_migration_phase[123]|graph_migration_unified_yolov8|graph_migration_unified_preproc'

# EVO remains hardware/devkit-only; run it when explicitly requested so this wrapper
# can still be used as the non-EVO Phase 3 closure pass.
if [[ "${SIMA_GRAPH_PHASE3_RUN_EVO:-0}" != "0" ]]; then
  tests/graph_migration/legacy/run_evo_route_matrix_12x4_legacy.sh
fi

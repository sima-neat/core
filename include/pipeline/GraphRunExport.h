/**
 * @file
 * @ingroup diagnostics
 * @brief Compatibility header for Run JSON export.
 *
 * New code should include `pipeline/RunExport.h` and use `RunExportOptions`,
 * `run_to_json`, and `save_run_json`.  The old graph-run names remain as
 * source-compatible aliases because the JSON schema is still named
 * `sima.neat.graph_run` for wire compatibility.
 */
#pragma once

#include "pipeline/RunExport.h"

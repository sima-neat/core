/**
 * @file
 * @ingroup diagnostics
 * @brief Export a built NEAT Run's graph topology and runtime metrics as JSON.
 */
#pragma once

#include "pipeline/Run.h"

#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

/**
 * @brief Options for exporting a built `Run` as graph-run JSON.
 *
 * The export is intentionally a snapshot: it does not stop or mutate the Run.
 * Call `run.close()` first if you want final throughput/latency/power numbers
 * after all outputs have drained.
 */
struct GraphRunExportOptions {
  /// Human-readable graph/run label in the JSON. Empty means "run".
  std::string label;
  /// Include runtime counters, latency, power, and per-segment pipeline strings.
  bool include_metrics = true;
  /// Include board power summary when enabled on the Run.
  bool include_power = true;
  /// Pretty-print indentation. 0 emits compact JSON.
  int indent = 2;
  /// Free-form string metadata copied to the top-level "metadata" object.
  std::vector<std::pair<std::string, std::string>> metadata;
};

/// Serialize a built `Run` to `sima.neat.graph_run` JSON. Returns empty string on error.
std::string graph_run_to_json(const Run& run, const GraphRunExportOptions& opt = {},
                              std::string* err = nullptr);

/// Atomically write `graph_run_to_json(run, opt)` to `path` using `path + ".tmp"` then rename.
bool save_graph_run_json(const Run& run, const std::string& path,
                         const GraphRunExportOptions& opt = {}, std::string* err = nullptr);

} // namespace simaai::neat

/**
 * @file
 * @brief Reusable public Graph fragments for named endpoint routing.
 *
 * These helpers intentionally return ordinary `simaai::neat::Graph` objects. They do not expose
 * low-level runtime graph nodes, port handles, or node IDs. Users can inspect, compose, save, and
 * build the returned fragments exactly like any other Graph.
 */
#pragma once

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::graphs {
namespace detail {

inline void require_unique_endpoint_name(std::unordered_set<std::string>* used,
                                         std::string_view name, const char* helper) {
  if (!used) {
    return;
  }
  if (name.empty()) {
    throw std::runtime_error(std::string(helper) + ": endpoint name must not be empty");
  }
  if (!used->insert(std::string(name)).second) {
    throw std::runtime_error(std::string(helper) + ": duplicate endpoint name '" +
                             std::string(name) + "'");
  }
}

} // namespace detail

/**
 * @brief Build a reusable Graph that branches one named input to several named outputs.
 *
 * Example:
 *
 * @code
 * Graph branch = graphs::Branch("image", {"preview", "model"});
 * @endcode
 *
 * Conceptually:
 *
 * @code
 * image -> preview
 *       -> model
 * @endcode
 *
 * The returned object is a normal public `Graph` containing one `Input(input)`, one
 * `Output(name)` per requested output, and endpoint `connect(input, output)` edges. At build time
 * the compiler lowers this to the appropriate internal branch/fan-out runtime machinery while
 * preserving the public endpoint names for `Run::pull(name)`, diagnostics, and visualization.
 */
inline Graph Branch(std::string input, std::vector<std::string> outputs) {
  if (outputs.empty()) {
    throw std::runtime_error("graphs::Branch: at least one output endpoint is required");
  }

  std::unordered_set<std::string> used;
  detail::require_unique_endpoint_name(&used, input, "graphs::Branch");
  for (const auto& output : outputs) {
    detail::require_unique_endpoint_name(&used, output, "graphs::Branch");
  }

  Graph g(input);
  g.add(nodes::Input(input));
  for (const auto& output : outputs) {
    g.add(nodes::Output(output));
  }
  for (const auto& output : outputs) {
    g.connect(input, output);
  }
  return g;
}

/**
 * @brief Build a reusable Graph that combines several named inputs into one named output.
 *
 * Example:
 *
 * @code
 * Graph join = graphs::Combine({"left", "right"}, "pair", CombinePolicy::ByFrame);
 * @endcode
 *
 * `policy` defines how samples from the named inputs are matched:
 *
 * - `CombinePolicy::ByFrame`: exact `Sample::frame_id` match.
 * - `CombinePolicy::ByPts`: exact `Sample::pts_ns` / Presentation Timestamp match.
 * - `CombinePolicy::None`: fail closed if multiple producers reach the output.
 *
 * The helper only constructs public Graph topology. The compiler is responsible for lowering the
 * multi-producer output to the internal combine stage for `ByFrame` / `ByPts`.
 */
inline Graph Combine(std::vector<std::string> inputs, std::string output,
                     CombinePolicy policy = CombinePolicy::ByFrame) {
  if (inputs.empty()) {
    throw std::runtime_error("graphs::Combine: at least one input endpoint is required");
  }

  std::unordered_set<std::string> used;
  for (const auto& input : inputs) {
    detail::require_unique_endpoint_name(&used, input, "graphs::Combine");
  }
  detail::require_unique_endpoint_name(&used, output, "graphs::Combine");

  Graph g(output);
  for (const auto& input : inputs) {
    g.add(nodes::Input(input));
  }
  OutputOptions opt;
  opt.combine_policy = policy;
  g.add(nodes::Output(output, opt));
  for (const auto& input : inputs) {
    g.connect(input, output);
  }
  return g;
}

} // namespace simaai::neat::graphs

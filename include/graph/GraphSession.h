/**
 * @file
 * @ingroup graph
 * @brief `GraphSession` — builds an actor-style runtime from a `graph::Graph`.
 *
 * `GraphSession` is to runtime `Graph` what `pipeline::Session` is to a list of `Node`s:
 * the assembly stage that turns a topology into something runnable. It compiles the
 * graph (validation, port wiring, scheduling decisions) and produces a `GraphRun` that
 * the caller drives.
 *
 * @see Graph, GraphRun, StageExecutor
 * @see "Runtime graph" (§73 of the design deep dive)
 */
#pragma once

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"

#include <memory>
#include <utility>

namespace simaai::neat::graph {

/**
 * @brief Builder for a runtime graph — compile a `Graph` into a runnable `GraphRun`.
 *
 * Single-use: construct with the topology, then call `build()` once to produce the
 * runtime. The session retains ownership of the source graph for diagnostics.
 *
 * @ingroup graph
 */
class GraphSession {
public:
  /// Wrap a runtime `Graph` so it can be built into a `GraphRun`.
  explicit GraphSession(Graph graph);

  /**
   * @brief Compile the wrapped graph and return a runnable `GraphRun`.
   * @param opt Runtime tuning knobs (scheduler, queues, telemetry).
   * @throws SessionError on compile-time validation failures (cycles, port mismatches).
   */
  GraphRun build(const GraphRunOptions& opt = {});

private:
  Graph graph_;
};

} // namespace simaai::neat::graph

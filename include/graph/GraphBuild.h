/**
 * @file
 * @ingroup graph
 * @brief Build helpers for runtime `graph::Graph`.
 */
#pragma once

#include "graph/Graph.h"
#include "graph/GraphRun.h"

namespace simaai::neat::graph {

/**
 * @brief Compile a runtime graph and return a runnable `GraphRun`.
 *
 * This is the canonical low-level runtime graph build API. It replaces the old
 * runtime-graph builder object: a graph is built directly, with no intermediate public
 * builder object.
 *
 * @param graph Runtime graph topology. Ownership is moved into the build path.
 * @param opt Runtime tuning knobs (scheduler queues, push/pull timeouts, telemetry).
 * @throws NeatError on compile-time validation failures (cycles, port mismatches) or runtime
 * start failures.
 */
GraphRun build(Graph graph, const GraphRunOptions& opt = {});

} // namespace simaai::neat::graph

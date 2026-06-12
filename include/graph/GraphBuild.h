/**
 * @file
 * @ingroup graph
 * @brief Internal build helper for the actor-style runtime graph substrate.
 *
 * Public application code should call `simaai::neat::Graph::build()` and consume
 * the returned `simaai::neat::Run`. This header remains for runtime/compiler
 * internals and focused tests until the low-level graph substrate is fully hidden.
 */
#pragma once

#include "graph/Graph.h"
#include "graph/GraphRun.h"

namespace simaai::neat::graph {

/**
 * @brief Compile an internal runtime graph and return a runnable `GraphRun`.
 *
 * Do not use this from applications. Use public `simaai::neat::Graph::build()` so
 * pipeline nodes, runtime-stage nodes, model fragments, named endpoints, export,
 * metrics, and Python bindings all go through the unified public path.
 *
 * @param graph Runtime graph topology. Ownership is moved into the build path.
 * @param opt Runtime tuning knobs (scheduler queues, push/pull timeouts, telemetry).
 * @throws NeatError on compile-time validation failures (cycles, port mismatches) or runtime
 * start failures.
 */
GraphRun build(Graph graph, const GraphRunOptions& opt = {});

} // namespace simaai::neat::graph

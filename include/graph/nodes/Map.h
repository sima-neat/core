/**
 * @file
 * @ingroup graph_nodes
 * @brief Simple map-style stage helpers (Sample/Tensor).
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
struct Tensor;
} // namespace simaai::neat

namespace simaai::neat::graph::nodes {

/// In-place sample mutator used by `Map(SampleMapFn, ...)`.
using SampleMapFn = std::function<void(Sample&)>;
/// In-place sample + tensor mutator used by `TensorMap(TensorMapFn, ...)`.
using TensorMapFn = std::function<void(Sample&, simaai::neat::Tensor&)>;
/// Pure transform used by `Map(SampleMapTransformFn, ...)` — returns a new `Sample`.
using SampleMapTransformFn = std::function<Sample(Sample)>;

/**
 * @brief Build a stage node that applies an in-place mutator to each sample.
 *
 * @param fn       Mutator invoked once per incoming `Sample`.
 * @param label    Optional human-readable label.
 * @param options  Optional `StageNode` options.
 * @param out_fn   Optional output-spec resolver.
 * @return Shared pointer to a `graph::Node` exposing the map stage.
 */
std::shared_ptr<simaai::neat::graph::Node> Map(SampleMapFn fn, std::string label = {},
                                               StageNodeOptions options = {},
                                               StageNode::OutputSpecFn out_fn = {});

/**
 * @brief Build a stage node that applies an in-place mutator to each sample's primary tensor.
 *
 * @param fn       Mutator invoked with the `Sample` and its primary `Tensor`.
 * @param label    Optional human-readable label.
 * @param options  Optional `StageNode` options.
 * @param out_fn   Optional output-spec resolver.
 * @return Shared pointer to a `graph::Node` exposing the tensor-map stage.
 */
std::shared_ptr<simaai::neat::graph::Node> TensorMap(TensorMapFn fn, std::string label = {},
                                                     StageNodeOptions options = {},
                                                     StageNode::OutputSpecFn out_fn = {});

/**
 * @brief Build a stage node that returns a transformed `Sample` from each input.
 *
 * @param fn       Pure transform invoked once per incoming `Sample`.
 * @param label    Optional human-readable label.
 * @param options  Optional `StageNode` options.
 * @param out_fn   Optional output-spec resolver.
 * @return Shared pointer to a `graph::Node` exposing the map-transform stage.
 */
std::shared_ptr<simaai::neat::graph::Node> Map(SampleMapTransformFn fn, std::string label = {},
                                               StageNodeOptions options = {},
                                               StageNode::OutputSpecFn out_fn = {});

} // namespace simaai::neat::graph::nodes

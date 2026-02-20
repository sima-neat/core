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

using SampleMapFn = std::function<void(Sample&)>;
using TensorMapFn = std::function<void(Sample&, simaai::neat::Tensor&)>;
using SampleMapTransformFn = std::function<Sample(Sample)>;

std::shared_ptr<simaai::neat::graph::Node> Map(SampleMapFn fn, std::string label = {},
                                               StageNodeOptions options = {},
                                               StageNode::OutputSpecFn out_fn = {});

std::shared_ptr<simaai::neat::graph::Node> TensorMap(TensorMapFn fn, std::string label = {},
                                                     StageNodeOptions options = {},
                                                     StageNode::OutputSpecFn out_fn = {});

std::shared_ptr<simaai::neat::graph::Node> Map(SampleMapTransformFn fn, std::string label = {},
                                               StageNodeOptions options = {},
                                               StageNode::OutputSpecFn out_fn = {});

} // namespace simaai::neat::graph::nodes

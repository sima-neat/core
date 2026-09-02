#pragma once

#include "graph/Compiler.h"

#include <cstddef>
#include <unordered_set>

namespace simaai::neat::graph::internal {

CompiledGraph
compile_with_pipeline_boundaries(const Graph& graph, const CompilerOptions& options,
                                 const std::unordered_set<std::size_t>& pipeline_boundary_edges);

} // namespace simaai::neat::graph::internal

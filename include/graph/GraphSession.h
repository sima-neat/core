/**
 * @file
 * @ingroup graph
 * @brief GraphSession builds hybrid runtimes from a Graph.
 */
#pragma once

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"

#include <memory>
#include <utility>

namespace simaai::neat::graph {

class GraphSession {
public:
  explicit GraphSession(Graph graph);

  GraphRun build(const GraphRunOptions& opt = {});

private:
  Graph graph_;
};

} // namespace simaai::neat::graph

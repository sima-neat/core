#include "pipeline/Graph.h"

#include "model/Model.h"

namespace simaai::neat {

Graph& Graph::add(const Model& model) {
  Model::RouteOptions opt;
  opt.include_input = false;
  opt.include_output = false;

  return add(model.graph(opt));
}

} // namespace simaai::neat

#include "pipeline/internal/ModelRouteOrigin.h"

#include <utility>

namespace simaai::neat::pipeline_internal {

namespace {

// The innermost scope of this thread.
thread_local const ModelRouteScope* t_innermost = nullptr;

} // namespace

ModelRouteScope::ModelRouteScope(const Graph& graph,
                                 std::shared_ptr<const ModelRouteOrigin> origin) noexcept
    : graph_(&graph), origin_(std::move(origin)), outer_(t_innermost) {
  t_innermost = this;
}

ModelRouteScope::~ModelRouteScope() {
  t_innermost = outer_;
}

std::shared_ptr<const ModelRouteOrigin> ModelRouteScope::of(const Graph& graph) noexcept {
  for (const ModelRouteScope* scope = t_innermost; scope; scope = scope->outer_) {
    if (scope->graph_ == &graph) {
      return scope->origin_;
    }
  }
  return nullptr;
}

} // namespace simaai::neat::pipeline_internal

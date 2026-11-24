#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The mark a Model's own build (Model::build, and Model::run, benchmark and execution_plan through
// it) puts on the build of the Graph it made of the model's node list: every node between that
// Graph's Input and its Output is a stage of one Model, and the build serves a call of the
// customer's. Only the Dataflow backend reads it (DataflowGraphCompiler.cpp): it qualifies those
// nodes as a model route and names the customer's call in what it does not carry. GStreamer never
// reads it, so nothing a GStreamer Run exports changes, and the Graph itself carries nothing.

#include <cstddef>
#include <memory>
#include <string>

namespace simaai::neat {
class Graph;
} // namespace simaai::neat

namespace simaai::neat::pipeline_internal {

struct ModelRouteOrigin {
  /// The customer's call the build serves: "Model::build", "Model::run", "Model::benchmark" or
  /// "Model::execution_plan".
  std::string call;
  /// The model, as its package names it.
  std::string model;
  /// The model's inputs (its ingress contracts).
  std::size_t inputs = 1;
  /// The model's one input tensor goes to the MLA as it is pushed: no stage in front of the MLA.
  bool direct_mla = false;
  /// Model::execution_plan: the build records the plan GStreamer's build records for the same
  /// Graph (SegmentBuildPlan), which the call reads.
  bool record_plan = false;
};

// While it lives, the builds this thread makes of `graph` are a Model's own build of `origin`.
// Scopes nest (a build inside a build marks its own Graph); one Graph is marked by one scope.
class ModelRouteScope {
public:
  ModelRouteScope(const Graph& graph, std::shared_ptr<const ModelRouteOrigin> origin) noexcept;
  ~ModelRouteScope();
  ModelRouteScope(const ModelRouteScope&) = delete;
  ModelRouteScope& operator=(const ModelRouteScope&) = delete;

  // The origin of a build of `graph` on this thread; null when no scope marks it.
  static std::shared_ptr<const ModelRouteOrigin> of(const Graph& graph) noexcept;

private:
  const Graph* graph_;
  std::shared_ptr<const ModelRouteOrigin> origin_;
  const ModelRouteScope* outer_;
};

} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Dataflow backend's build side: which runtime a Graph::build() takes
// (DataflowBackendSelect.cpp; Graph::build's four entries call DataflowGraphAccess::build before
// gst_init_once()), and whether a graph qualifies for Dataflow and with which Run configuration
// (DataflowGraphCompiler.cpp).

#include "pipeline/GraphOptions.h"
#include "pipeline/Run.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/runtime/DataflowRun.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {
class Graph;
namespace pipeline_internal {
struct ModelRouteOrigin;
} // namespace pipeline_internal
} // namespace simaai::neat

namespace simaai::neat::runtime {

struct RunCore;
struct DataflowGraphView; // DataflowGraphCompiler.cpp: what the compiler reads of a Graph

// Friend of Graph (Graph.h): reads its composition view. Hidden explicitly, since Graph.h's forward
// declaration fixed the class's visibility before the pragma below.
struct __attribute__((visibility("hidden"))) DataflowGraphAccess {
  // The Graph's composition as the compiler needs it, read as compile_public_graph reads it.
  static DataflowGraphView view(const Graph& graph);

  // The mark of a Model's own build of the Graph on this thread (ModelRouteScope); null for any
  // other build.
  static std::shared_ptr<const pipeline_internal::ModelRouteOrigin>
  model_origin(const Graph& graph);

  // nullptr: this Run is for GStreamer. Otherwise the RunCore of a Run on Dataflow. Throws when
  // the Run is for Dataflow and Dataflow does not carry it or cannot build it: nothing is built
  // on GStreamer in its place.
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const TensorList& seed, const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const Sample& seed, const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const std::vector<cv::Mat>& seed, const char* where);
};

#pragma GCC visibility push(hidden)

// One stderr line per process and key: the options Dataflow ignores or clamps.
void dataflow_warn_once(const std::string& key, const std::string& message);

// The seed of a seeded build: the Run's representative input, which binds the (one) input's spec.
// A view of the caller's tensors, which outlive the build.
struct DataflowSeed {
  std::span<const Tensor> tensors; ///< empty: no seed
  std::string unsupported;         ///< a seed this backend cannot carry
};

// The options of the graph's public Input when it qualifies with exactly one; InputOptions{}
// otherwise. What a cv::Mat seed is converted under. Defined with the runtime only.
InputOptions dataflow_seed_input_options(const Graph& graph);

#if defined(SIMANEAT_WITH_DATAFLOW)

// The Run configuration of `graph` under `opt` when the graph qualifies (public Inputs feeding
// public Outputs, each Output fed by one Input; or one Input, a Model's stages in a chain, one
// Output) and Dataflow carries its options and `seed`; nullopt with `why` set otherwise: who is
// not carried and the fact about it. Throws what GStreamer's build throws for a mistake both
// backends refuse (an endpoint name two endpoints share, a seed for a graph of several Inputs),
// naming `where`.
std::optional<DataflowRunConfig> plan_dataflow_run(const Graph& graph, const RunOptions& opt,
                                                   const DataflowSeed& seed, const char* where,
                                                   pipeline_internal::DataflowNotCarried* why);

// The options Dataflow does not read, when set away from their defaults: one warning per process
// and option.
void warn_ignored_dataflow_options(const DataflowRunConfig& config);

#endif // SIMANEAT_WITH_DATAFLOW

#pragma GCC visibility pop

} // namespace simaai::neat::runtime

// Which runtime a Graph::build() takes: RunOptions::backend and SIMA_NEAT_BACKEND decide.
// Graph::build's four entries call DataflowGraphAccess::build before gst_init_once(): nullptr is
// a Run for GStreamer, by that choice alone. A Run for Dataflow is built on Dataflow or the build
// throws; nothing is built on GStreamer in its place:
//   - a graph, an option or a model route the planner does not carry (DataflowGraphCompiler.cpp),
//     a link deeper than the backend's, a seed or a declaration no input can bind: the
//     not-carried error (DataflowNotCarried.h), before anything is built;
//   - an error while the Run is built (its models loaded, its kernels checked, each input's Plan
//     compiled and its Session started where the input binds at the build): that error, whose
//     action is the GStreamer backend unless GStreamer would meet the same error (the error
//     table's backend_action, DataflowErrorMap.h).
// An input that binds at its first push (no seed, no declared shape) compiles its Plan there: an
// error then is the push's, on the Dataflow Run that exists, and names the push.
// Compiles without the runtime: a Run for Dataflow then fails.

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/runtime/RunCore.h"

#include "pipeline/internal/InputStreamUtil.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/runtime/DataflowInputBinding.h"
#include "pipeline/runtime/DataflowModelRoute.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::runtime {

namespace {

constexpr const char* kBackendEnv = "SIMA_NEAT_BACKEND";

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

// SIMA_NEAT_BACKEND, read at every Graph::build so a test can set it per run.
std::string backend_env() {
  return lower(pipeline_internal::env_str(kBackendEnv, ""));
}

} // namespace

void dataflow_warn_once(const std::string& key, const std::string& message) {
  static std::mutex mu;
  static std::set<std::string> seen;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
      return;
    }
  }
  std::fprintf(stderr, "[WARN] %s\n", message.c_str());
}

namespace {

// With SIMA_NEAT_BACKEND set, every Graph::build on Dataflow says so and what it built: the
// evidence a test log needs.
void trace_choice(const char* where, const char* backend, const std::string& detail) {
  if (pipeline_internal::env_str(kBackendEnv, "").empty()) {
    return;
  }
  std::fprintf(stderr, "[NEAT] %s: backend=%s (%s)\n", where ? where : "Graph::build", backend,
               detail.c_str());
}

// Whether this Run is for Dataflow: RunOptions::backend says it, and RunBackend::Auto reads
// SIMA_NEAT_BACKEND.
bool wants_dataflow(const RunOptions& opt) {
  if (opt.backend != RunBackend::Auto) {
    return opt.backend == RunBackend::Dataflow;
  }
  const std::string env = backend_env();
  if (env.empty() || env == "auto" || env == "gstreamer") {
    return false;
  }
  if (env != "dataflow") {
    dataflow_warn_once("env:" + env,
                       std::string(kBackendEnv) + "=" + env +
                           " is not a backend (dataflow, gstreamer, auto); using GStreamer");
    return false;
  }
  return true;
}

using pipeline_internal::DataflowNotCarried;

// A Run for Dataflow that Dataflow does not carry: the build's error.
[[noreturn]] void decline(const char* where, const DataflowNotCarried& what) {
  pipeline_internal::throw_dataflow_not_carried(where ? where : "Graph::build", what);
}

// ---- seeds -------------------------------------------------------------------------------------
// The seed of a seeded build is the Run's representative input: it binds the input's spec and is
// not pushed (GStreamer's preflight pushes and pulls it once, which leaves no output behind
// either). One tensor binds a leaf, several a record (DataflowSeed).
DataflowSeed seed_from_sample(const Sample& sample) {
  DataflowSeed seed;
  if (sample.kind == SampleKind::Bundle) {
    if (sample.fields.size() == 1U) {
      return seed_from_sample(sample.fields.front());
    }
    // Core pushes a Bundle one field at a time (RunCore::push_samples): each field is an item,
    // and they share no binding.
    seed.unsupported = "it is a Bundle of " + std::to_string(sample.fields.size()) +
                       " fields (several items for one input), which bind no one spec: seed "
                       "with one of its fields";
    return seed;
  }
  if (sample_has_tensor_list(sample)) {
    seed.tensors = sample.tensors;
    return seed;
  }
  if (sample.tensor.has_value()) {
    seed.tensors = std::span<const Tensor>(&*sample.tensor, 1);
    return seed;
  }
  seed.unsupported = "it is a Sample without a tensor";
  return seed;
}

#if defined(SIMANEAT_WITH_DATAFLOW)

// The [NEAT] line of a Run that takes Dataflow: every input and the links of its Outputs.
std::string describe_choice(const DataflowRunConfig& config,
                            const std::vector<std::optional<DataflowInputBinding>>& bindings) {
  std::string out;
  for (std::size_t i = 0; i < config.inputs.size(); ++i) {
    const DataflowInputConfig& input = config.inputs[i];
    out += (i == 0 ? "" : "; ") + std::string("input ") + dataflow_names_text(input.names) +
           (bindings[i] ? " " + bindings[i]->text : " bound by its first push") + " -> " +
           (input.model ? input.model->text + " -> " : std::string());
    for (std::size_t k = 0; k < input.outputs.size(); ++k) {
      const DataflowOutputConfig& output = config.outputs[input.outputs[k]];
      out += (k == 0 ? "" : ", ") + std::string("output ") + dataflow_names_text(output.names) +
             " link depth " + std::to_string(output.link_depth) + " (max of queue_depth " +
             std::to_string(output.ingress_depth) + ", max_buffers " +
             std::to_string(output.terminal_depth) + ")" +
             (output.terminal_drop                                 ? " drop oldest (Output drop)"
              : config.link_policy == OverflowPolicy::KeepLatest   ? " try_push keep latest"
              : config.link_policy == OverflowPolicy::DropIncoming ? " try_push drop incoming"
                                                                   : " block");
    }
  }
  return out;
}

std::shared_ptr<RunCore> build_impl(const Graph& graph, const RunOptions& opt, RunMode mode,
                                    const DataflowSeed& seed, const char* where) {
  if (mode != RunMode::Async) {
    decline(where, {error_codes::kPipelineShape, "a synchronous Run",
                    "Graph::run(inputs) builds one, and the backend runs the Runs of Graph::build"});
  }
  DataflowNotCarried declined;
  std::optional<DataflowRunConfig> planned = plan_dataflow_run(graph, opt, seed, where, &declined);
  if (!planned) {
    decline(where, declined);
  }
  const DataflowRunConfig& config = *planned;
  for (const DataflowOutputConfig& output : config.outputs) {
    if (output.link_depth > pipeline_internal::df::kMaxLinkDepth) {
      decline(where, {error_codes::kOptionOutOfRange,
                      "option RunOptions::queue_depth = " + std::to_string(output.ingress_depth) +
                          " with OutputOptions::max_buffers = " +
                          std::to_string(output.terminal_depth),
                      "the link from an Input to its Output is as deep as the larger of the two, " +
                          std::to_string(output.link_depth) + ", and a link holds at most " +
                          std::to_string(pipeline_internal::df::kMaxLinkDepth) + " items"});
    }
  }
  // Each input's spec, when a seed or the declaration gives it now; a declaration or a seed that
  // binds none is decided here, before a Run exists.
  std::string why;
  std::vector<std::optional<DataflowInputBinding>> bindings;
  for (const DataflowInputConfig& input : config.inputs) {
    bindings.push_back(bind_dataflow_item(input.input, seed.tensors, &why));
    if (!why.empty()) {
      decline(where, {error_codes::kInputShape,
                      "Input " + dataflow_names_text(input.names) +
                          (seed.tensors.empty() ? " as declared" : " with this seed"),
                      why});
    }
  }
  // A tensor pulled from a device pool is a GstSample over a GstDmaBufMemory
  // (DataflowDeviceMemory.cpp), so a Run with a device input starts GStreamer here, as a GStreamer
  // Run does at its build. A Run of host inputs only never does.
  for (const DataflowInputConfig& input : config.inputs) {
    if (input.input.memory_policy == InputMemoryPolicy::Ev74 ||
        input.input.memory_policy == InputMemoryPolicy::Dms0) {
      gst_init_once();
      break;
    }
  }

  auto core = std::make_shared<RunCore>();
  initialize_run_identity(*core);
  core->opt = config.run;
  core->mode = RunMode::Async;
  core->dataflow_ = make_dataflow_run_state(*core, config, seed.tensors);
  warn_ignored_dataflow_options(config);
  trace_choice(where, "dataflow", describe_choice(config, bindings));
  return core;
}

#else

// No runtime in this build: dispatch fails before it asks for a build.
template <class... Args> std::shared_ptr<RunCore> build_impl(const Args&...) {
  return nullptr;
}

#endif // SIMANEAT_WITH_DATAFLOW

// The common head of the four entries: a Run for GStreamer, or no runtime in this build.
template <class Build>
std::shared_ptr<RunCore> dispatch(const RunOptions& opt, const char* where, Build&& build) {
  if (!wants_dataflow(opt)) {
    return nullptr;
  }
#if defined(SIMANEAT_WITH_DATAFLOW)
  (void)where; // the build reports with it
  return build();
#else
  (void)build;
  decline(where, {error_codes::kPipelineShape, "this build of libsima_neat",
                  "it was configured without the Dataflow runtime (SIMANEAT_DATAFLOW_SOURCE_DIR)"});
#endif
}

} // namespace

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const char* where) {
  return dispatch(opt, where,
                  [&] { return build_impl(graph, opt, mode, DataflowSeed{}, where); });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const TensorList& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&] {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty tensor list");
    }
    DataflowSeed s;
    s.tensors = seed;
    return build_impl(graph, opt, mode, s, where);
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const Sample& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&] {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty sample list");
    }
    return build_impl(graph, opt, mode, seed_from_sample(seed), where);
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const std::vector<cv::Mat>& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&]() -> std::shared_ptr<RunCore> {
#if defined(SIMA_WITH_OPENCV)
    // GStreamer's own checks and texts (Graph::build_seeded_internal).
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty image list");
    }
    for (std::size_t i = 0; i < seed.size(); ++i) {
      if (seed[i].empty()) {
        throw std::runtime_error("Graph::build(inputs): empty image input at index " +
                                 std::to_string(i));
      }
    }
    DataflowSeed s;
    if (seed.size() != 1U) {
      s.unsupported = "it is " + std::to_string(seed.size()) +
                      " images (a record) where a cv::Mat seed is one image";
      return build_impl(graph, opt, mode, s, where);
    }
    // The seed only binds the spec: a CPU view, whatever the Input's memory policy.
    InputOptions input = dataflow_seed_input_options(graph);
    input.memory_policy = InputMemoryPolicy::SystemMemory;
    const Tensor view = tensor_from_cv_mat(seed.front(), input, "Graph::build(inputs)");
    s.tensors = std::span<const Tensor>(&view, 1);
    return build_impl(graph, opt, mode, s, where);
#else
    throw std::runtime_error("Graph::build(inputs): this build has no OpenCV");
#endif
  });
}

} // namespace simaai::neat::runtime

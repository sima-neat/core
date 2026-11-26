// Which runtime a Graph::build() takes: the process's backend (set_backend, BackendSetting.h),
// which every build fixes.
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
// An input of a graph Run that binds at its first push (no seed, no declared shape, no model route
// in front) compiles its Plan there: an error then is the push's, on the Dataflow Run that exists,
// and names the push. A plain linear Run without a seed or a declaration is not built, as on
// GStreamer.
// Compiles without the runtime: a Run for Dataflow then fails.

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/BackendSetting.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/ModelRouteOrigin.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
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
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::runtime {

namespace {

// With SIMA_GRAPH_DEBUG or SIMA_PIPELINE_DEBUG set, every Graph::build on Dataflow says so and
// what it built: the evidence a test log needs.
void trace_choice(const char* where, const char* backend, const std::string& detail) {
  if (!pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false) &&
      !pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false)) {
    return;
  }
  std::fprintf(stderr, "[NEAT] %s: backend=%s (%s)\n", where ? where : "Graph::build", backend,
               detail.c_str());
}

// Whether this Run is for Dataflow: the process's backend, which the build fixes (use_backend).
bool wants_dataflow() {
  return pipeline_internal::use_backend() == RunBackend::Dataflow;
}

using pipeline_internal::DataflowNotCarried;

// GraphOptions::verbose's progress lines on a Dataflow build: the steps GStreamer's build of the
// same entry prints (Graph::build in GraphBuildSource.cpp, the seeded entries in
// GraphBuildInput.cpp), in its order and numbering, then "Graph ready" with the build's time.
// The first step is printed when the build starts, the others once the Run is planned. A build
// that also has GStreamer record its plan (Model::execution_plan) prints only "Graph ready": that
// recording build prints GStreamer's steps before it, once.
class BuildProgress {
public:
  BuildProgress(const Graph& graph, bool seeded, bool quiet)
      : context_(DataflowGraphAccess::verbose(graph)),
        progress_(DataflowGraphAccess::verbose(graph), 4), seeded_(seeded),
        connected_(DataflowGraphAccess::connected(graph)), quiet_(quiet) {
    if (!quiet_) {
      progress_.step("Initializing runtime...");
    }
  }
  void planned() {
    if (quiet_) {
      return;
    }
    if (!seeded_) {
      progress_.step("Building graph...");
      progress_.step("Starting pipeline...");
    } else if (connected_) {
      progress_.step("Building graph...");
      progress_.step("Preparing input stream...");
      progress_.step("Starting pipeline...");
    } else {
      progress_.step("Preparing input stream...");
      progress_.step("Building graph...");
    }
  }
  void done() {
    progress_.done("Graph ready");
  }

private:
  pipeline_internal::ux::ScopedVerboseContext context_;
  pipeline_internal::ux::ProgressReporter progress_;
  bool seeded_;
  bool connected_;
  bool quiet_;
};

// A Run for Dataflow that Dataflow does not carry: the build's error.
[[noreturn]] void decline(const char* where, const DataflowNotCarried& what) {
  pipeline_internal::throw_dataflow_not_carried(where ? where : "Graph::build", what);
}

// ---- seeds -------------------------------------------------------------------------------------
// The seed of a seeded build is the Run's representative input: it binds the input's spec, and the
// build's preflight pushes and pulls it once, as GStreamer's does (build_impl). One tensor binds a
// leaf, several a record (DataflowSeed).
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

// What a seeded entry gives the build besides the seed's tensors: how the seed is pushed into the
// Run (DataflowRunState::preflight), in the entry's own form (a TensorList, a Sample, the cv::Mat
// images), and how GStreamer's path builds the same Graph with the same seed
// (session_build_segment_plan).
struct SeedForms {
  std::function<bool(DataflowRunState&, const char* where)> push;
  std::function<void()> gstreamer_build;
};

std::shared_ptr<RunCore> build_impl(const Graph& graph, const RunOptions& opt, RunMode mode,
                                    const DataflowSeed& seed, const char* where,
                                    const SeedForms& forms = {}) {
  // What is not carried names the customer's call: the Model's (ModelRouteOrigin) when a Model's
  // own build made the Graph. A mistake both backends refuse keeps GStreamer's text and `where`.
  const std::shared_ptr<const pipeline_internal::ModelRouteOrigin> origin =
      DataflowGraphAccess::model_origin(graph);
  BuildProgress progress(graph, !seed.tensors.empty() || !seed.unsupported.empty(),
                         origin && origin->record_plan && forms.gstreamer_build);
  const std::string call = origin ? origin->call : std::string(where ? where : "Graph::build");
  DataflowNotCarried declined;
  std::optional<DataflowRunConfig> planned = plan_dataflow_run(graph, opt, seed, where, &declined);
  if (!planned) {
    decline(call.c_str(), declined);
  }
  planned->where = call;
  if (mode == RunMode::Sync && planned->run.output_memory == OutputMemory::Auto) {
    // Graph::run's synchronous Run (driven once per call by Run::run, an asynchronous Run here):
    // its outputs are the caller's own, as a synchronous Run's Auto gives them on GStreamer.
    planned->run.output_memory = OutputMemory::Owned;
  }
  const DataflowRunConfig& config = *planned;
  for (const DataflowOutputConfig& output : config.outputs) {
    if (output.link_depth > pipeline_internal::df::kMaxLinkDepth) {
      decline(call.c_str(),
              {error_codes::kOptionOutOfRange,
               "option RunOptions::queue_depth = " + std::to_string(output.ingress_depth) +
                   " with OutputOptions::max_buffers = " + std::to_string(output.terminal_depth),
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
      decline(call.c_str(), {error_codes::kInputShape,
                             "Input " + dataflow_names_text(input.names) +
                                 (seed.tensors.empty() ? " as declared" : " with this seed"),
                             why});
    }
    if (!bindings.back() && !config.graph_run) {
      // GStreamer's build of a plain linear Run starts its one pipeline segment with the seed
      // and refuses to build without one: the same mistake, the same error. (A declaration that
      // gives the input's spec binds it here instead.)
      throw std::runtime_error(
          "RunCore::start_pipeline_segment: push-style segment requires an input seed");
    }
  }
  // A Run with a device input keeps the output a GStreamer Run gives: the drivers' lines off stdout
  // and stderr (Core's filter, which GStreamer's initialisation installs there). A Run of host
  // inputs only installs nothing.
  for (const DataflowInputConfig& input : config.inputs) {
    if (input.input.memory_policy == InputMemoryPolicy::Ev74 ||
        input.input.memory_policy == InputMemoryPolicy::Dms0) {
      pipeline_internal::ux::install_device_log_filters();
      break;
    }
  }

  progress.planned();
  auto core = std::make_shared<RunCore>();
  initialize_run_identity(*core);
  core->opt = config.run;
  core->mode = RunMode::Async;
  core->dataflow_ = make_dataflow_run_state(*core, config, seed.tensors);
  // What run_to_json and the run exports draw of the Run's topology: the plan GStreamer's build
  // compiles for the same Graph and seed, kept as it keeps it for a Run of one pipeline segment.
  // A plan that does not compile leaves the topology empty, as for a Run without one.
  try {
    std::optional<Sample> plan_seed;
    if (!seed.tensors.empty() && !DataflowGraphAccess::connected(graph)) {
      plan_seed = pipeline_internal::sample_from_tensors_for_input(
          TensorList(seed.tensors.begin(), seed.tensors.end()), dataflow_seed_input_options(graph));
    }
    core->graph_export_plan_ =
        std::make_unique<ExecutionGraphPlan>(compile_public_graph(graph, opt, plan_seed));
  } catch (const std::exception&) {
    core->graph_export_plan_.reset();
  }
  // RunOptions::power_monitor (deprecated): the board's rails sampled for the Run's lifetime, from
  // its build to its stop, as GStreamer's Run samples them (MeasureReport::power, the exports).
  if (core->opt.power_monitor.enabled) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    core->power_monitor = std::make_unique<PowerMonitor>(core->opt.power_monitor);
#pragma GCC diagnostic pop
    core->power_monitor->start();
  }
  // A seeded build pushes its seed and pulls the output once before it returns, where GStreamer's
  // does: every seeded Run of one Output (nothing this backend carries is a pipeline GStreamer
  // skips the preflight for). A first-frame refusal of the seed is then the build's error.
  int timeout_ms = 0;
  if (forms.push && config.outputs.size() == 1U &&
      session_build_startup_preflight(opt, mode, &timeout_ms)) {
    DataflowRunState& run = *core->dataflow_;
    run.preflight([&] { return forms.push(run, call.c_str()); }, timeout_ms, call.c_str());
  }
  // Model::execution_plan reads the plan GStreamer's build of this Graph records: the same
  // stages, made by GStreamer's own cold steps, so that both backends describe one route alike.
  if (origin && origin->record_plan && forms.gstreamer_build) {
    core->dataflow_plan_ = session_build_segment_plan(forms.gstreamer_build);
  }
  trace_choice(where, "dataflow", describe_choice(config, bindings));
  progress.done();
  return core;
}

#else

// No runtime in this build: dispatch fails before it asks for a build.
template <class... Args> std::shared_ptr<RunCore> build_impl(const Args&...) {
  return nullptr;
}

#endif // SIMANEAT_WITH_DATAFLOW

// The common head of the four entries: a Run for GStreamer, or no runtime in this build.
template <class Build> std::shared_ptr<RunCore> dispatch(const char* where, Build&& build) {
  if (!wants_dataflow() || session_build_recording_segment_plan()) {
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
  return dispatch(where, [&] { return build_impl(graph, opt, mode, DataflowSeed{}, where); });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const TensorList& seed,
                                                    const char* where) {
  return dispatch(where, [&] {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty tensor list");
    }
    DataflowSeed s;
    s.tensors = seed;
    return build_impl(graph, opt, mode, s, where,
                      {[&seed](DataflowRunState& run, const char* at) {
                         return run.push(std::nullopt, seed, true, at);
                       },
                       [&] { (void)const_cast<Graph&>(graph).build(seed, opt); }});
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const Sample& seed,
                                                    const char* where) {
  return dispatch(where, [&] {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty sample list");
    }
    return build_impl(graph, opt, mode, seed_from_sample(seed), where,
                      {[&seed](DataflowRunState& run, const char* at) {
                         return run.push(std::nullopt, seed, true, at);
                       },
                       [&] { (void)const_cast<Graph&>(graph).build(seed, opt); }});
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const std::vector<cv::Mat>& seed,
                                                    const char* where) {
  return dispatch(where, [&]() -> std::shared_ptr<RunCore> {
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
    return build_impl(graph, opt, mode, s, where,
                      {[&seed](DataflowRunState& run, const char* at) {
                         return run.push(std::nullopt, seed, true, at);
                       },
                       [&] { (void)const_cast<Graph&>(graph).build(seed, opt); }});
#else
    throw std::runtime_error("Graph::build(inputs): this build has no OpenCV");
#endif
  });
}

} // namespace simaai::neat::runtime

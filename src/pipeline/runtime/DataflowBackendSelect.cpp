// Which runtime a Graph::build() takes: RunOptions::backend and SIMA_NEAT_BACKEND decide.
// Graph::build's four entries call DataflowGraphAccess::build before gst_init_once(): nullptr keeps
// the Run on GStreamer. A Run that Dataflow cannot build stays there too, or fails loudly when
// RunBackend::Dataflow was asked for explicitly:
//   - a graph or an option the planner declines (DataflowGraphCompiler.cpp), a link deeper than the
//     runtime's, a seed or a declaration no input can bind: declined before anything is built;
//   - an error while the Run is built (its models loaded, its kernels checked, each input's Plan
//     compiled and its Session started where the input binds at the build): RunBackend::Auto
//     builds the Run on GStreamer instead, unless GStreamer would meet the same error (the
//     error table's retry_on_gstreamer, DataflowErrorMap.h: a lost device, a model file the MLA
//     does not load), which leaves Graph::build as it is.
// An input that binds at its first push (no seed, no declared shape) compiles its Plan there: an
// error then is the push's, on the Dataflow Run that exists, and names the push.
// Compiles without the runtime: an explicit request then fails.

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
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

// With SIMA_NEAT_BACKEND set, every Graph::build says which runtime it took and why: the evidence a
// test log needs to tell a Dataflow block from one that stayed on GStreamer.
void trace_choice(const char* where, const char* backend, const std::string& detail) {
  if (pipeline_internal::env_str(kBackendEnv, "").empty()) {
    return;
  }
  std::fprintf(stderr, "[NEAT] %s: backend=%s (%s)\n", where ? where : "Graph::build", backend,
               detail.c_str());
}

// The verdict: build on Dataflow, stay on GStreamer, or fail (explicit Dataflow only).
struct Want {
  bool dataflow = false;
  bool explicit_request = false;
};

Want wanted_backend(const RunOptions& opt) {
  if (opt.backend == RunBackend::GStreamer) {
    return {};
  }
  if (opt.backend == RunBackend::Dataflow) {
    return {true, true};
  }
  const std::string env = backend_env();
  if (env.empty() || env == "auto" || env == "gstreamer") {
    return {};
  }
  if (env != "dataflow") {
    dataflow_warn_once("env:" + env,
                       std::string(kBackendEnv) + "=" + env +
                           " is not a backend (dataflow, gstreamer, auto); using GStreamer");
    return {};
  }
  return {true, false};
}

// What an explicit RunBackend::Dataflow request that Dataflow cannot build is told to do.
constexpr const char* kAction = "Build this Run with RunBackend::GStreamer or RunBackend::Auto.";

// Why Dataflow does not take this Run: an error for an explicit request, a trace line for Auto.
std::shared_ptr<RunCore> decline(const Want& want, const char* where, const char* code,
                                 const std::string& reason) {
  if (want.explicit_request) {
    pipeline_internal::error_util::throw_session_error(
        code, std::string(where ? where : "Graph::build") + ": RunBackend::Dataflow: " + reason, {},
        kAction);
  }
  trace_choice(where, "gstreamer", std::string(kBackendEnv) + "=dataflow: " + reason);
  return nullptr;
}

#if defined(SIMANEAT_WITH_DATAFLOW)

// The first line of an error's text: the [NEAT] line of a Run that fell back.
std::string first_line(const char* text) {
  const std::string_view all(text ? text : "");
  return std::string(all.substr(0, all.find('\n')));
}

// Inside the handler of `error`, an error of an explicit RunBackend::Dataflow request while the
// Run is built: rethrown with a decline's action when its own text names none.
[[noreturn]] void rethrow_with_action(const NeatError& error) {
  GraphReport report = error.report();
  if (report.repro_note.find("\nHint: ") != std::string::npos) {
    throw;
  }
  report.repro_note = pipeline_internal::error_util::append_hint(report.repro_note, kAction);
  std::string message =
      pipeline_internal::error_util::decorate_error(report.error_code, report.repro_note);
  throw NeatError(std::move(message), std::move(report));
}

#endif // SIMANEAT_WITH_DATAFLOW

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
    // Core pushes a Bundle one field at a time (RunPush.cpp:1082-1086): each field is an item,
    // and they share no binding.
    seed.unsupported = "a Bundle seed of " + std::to_string(sample.fields.size()) +
                       " fields (several items for one input) does not bind one spec; seed with "
                       "one of its fields";
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
  seed.unsupported = "the seed Sample has no tensor";
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
                                    const DataflowSeed& seed, const char* where, const Want& want) {
  if (mode != RunMode::Async) {
    return decline(want, where, error_codes::kPipelineShape,
                   "synchronous runs (Graph::run) are not supported before H1");
  }
  DataflowDecline declined;
  std::optional<DataflowRunConfig> planned = plan_dataflow_run(graph, opt, seed, &declined);
  if (!planned) {
    return decline(want, where, declined.code, declined.reason);
  }
  const DataflowRunConfig& config = *planned;
  for (const DataflowOutputConfig& output : config.outputs) {
    if (output.link_depth > pipeline_internal::df::kMaxLinkDepth) {
      return decline(want, where, error_codes::kOptionOutOfRange,
                     "the link depth " + std::to_string(output.link_depth) +
                         " (the larger of RunOptions::queue_depth " +
                         std::to_string(output.ingress_depth) + " and OutputOptions::max_buffers " +
                         std::to_string(output.terminal_depth) + ") is above the Dataflow link's " +
                         std::to_string(pipeline_internal::df::kMaxLinkDepth));
    }
  }
  // Each input's spec, when a seed or the declaration gives it now; an unsupported declaration or
  // seed is decided here, before a Run exists.
  std::string why;
  std::vector<std::optional<DataflowInputBinding>> bindings;
  for (const DataflowInputConfig& input : config.inputs) {
    bindings.push_back(bind_dataflow_item(input.input, seed.tensors, &why));
    if (!why.empty()) {
      return decline(want, where, error_codes::kInputShape, why);
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
  try {
    core->dataflow_ = make_dataflow_run_state(*core, config, seed.tensors);
  } catch (const pipeline_internal::DataflowErrorOnEveryBackend&) {
    throw;
  } catch (const NeatError& error) {
    if (want.explicit_request) {
      rethrow_with_action(error);
    }
    trace_choice(where, "gstreamer",
                 std::string(kBackendEnv) + "=dataflow: the Run could not be built on Dataflow: " +
                     first_line(error.what()));
    return nullptr;
  }
  warn_ignored_dataflow_options(config);
  trace_choice(where, "dataflow", describe_choice(config, bindings));
  return core;
}

#else

// No runtime in this build: dispatch declines before it asks for a build.
template <class... Args> std::shared_ptr<RunCore> build_impl(const Args&...) {
  return nullptr;
}

#endif // SIMANEAT_WITH_DATAFLOW

// The common head of the four entries: no Dataflow wanted, or no runtime in this build.
template <class Build>
std::shared_ptr<RunCore> dispatch(const RunOptions& opt, const char* where, Build&& build) {
  const Want want = wanted_backend(opt);
  if (!want.dataflow) {
    return nullptr;
  }
#if defined(SIMANEAT_WITH_DATAFLOW)
  (void)where; // the build reports with it
  return build(want);
#else
  (void)build;
  return decline(want, where, error_codes::kPipelineShape,
                 "this build of libsima_neat has no Dataflow backend (configure it with "
                 "SIMANEAT_DATAFLOW_SOURCE_DIR)");
#endif
}

} // namespace

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const char* where) {
  return dispatch(opt, where, [&](const auto& want) {
    return build_impl(graph, opt, mode, DataflowSeed{}, where, want);
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const TensorList& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty tensor list");
    }
    DataflowSeed s;
    s.tensors = seed;
    return build_impl(graph, opt, mode, s, where, want);
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const Sample& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) {
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty sample list");
    }
    return build_impl(graph, opt, mode, seed_from_sample(seed), where, want);
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const std::vector<cv::Mat>& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) -> std::shared_ptr<RunCore> {
#if defined(SIMA_WITH_OPENCV)
    // GStreamer's own checks and texts (GraphBuildInput.cpp:2702-2711).
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
      s.unsupported =
          "a seed of " + std::to_string(seed.size()) + " images (a record) is not supported yet";
      return build_impl(graph, opt, mode, s, where, want);
    }
    // The seed only binds the spec: a CPU view, whatever the Input's memory policy.
    InputOptions input = dataflow_seed_input_options(graph);
    input.memory_policy = InputMemoryPolicy::SystemMemory;
    const Tensor view = tensor_from_cv_mat(seed.front(), input, "Graph::build(inputs)");
    s.tensors = std::span<const Tensor>(&view, 1);
    return build_impl(graph, opt, mode, s, where, want);
#else
    (void)want;
    return nullptr;
#endif
  });
}

} // namespace simaai::neat::runtime

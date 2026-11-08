// The Dataflow backend's compiler (H0 plan 3.4, 3.6): RunOptions::backend and SIMA_NEAT_BACKEND
// decide the runtime of a Graph::build(); a qualifying Input -> Output graph becomes one Feed, one
// Fetch and one link, and its Run state (DataflowRun.cpp). Everything else stays on GStreamer, or
// fails loudly when RunBackend::Dataflow was asked for explicitly.

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/PublicInputContract.h"
#include "pipeline/internal/SampleProjection.h"
#include "pipeline/runtime/RunCore.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/internal/DataflowErrorMap.h"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <span>
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

// One stderr line per process and key: the options Dataflow ignores, and an unknown env value.
void warn_once(const std::string& key, const std::string& message) {
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
    warn_once("env:" + env, std::string(kBackendEnv) + "=" + env +
                                " is not a backend (dataflow, gstreamer, auto); using GStreamer");
    return {};
  }
  return {true, false};
}

// Why Dataflow does not take this Run: an error for an explicit request, a trace line for Auto.
std::shared_ptr<RunCore> decline(const Want& want, const char* where, const char* code,
                                 const std::string& reason) {
  if (want.explicit_request) {
    pipeline_internal::error_util::throw_session_error(
        code, std::string(where ? where : "Graph::build") + ": RunBackend::Dataflow: " + reason, {},
        "Build this Run with RunBackend::GStreamer or RunBackend::Auto.");
  }
  trace_choice(where, "gstreamer", std::string(kBackendEnv) + "=dataflow: " + reason);
  return nullptr;
}

#if defined(SIMANEAT_WITH_DATAFLOW)

namespace df = pipeline_internal::df;

std::string join(const std::vector<std::string>& names) {
  std::string out = "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out += (i == 0 ? "" : ", ") + names[i];
  }
  return out + "]";
}

std::string node_text(const Node& node) {
  const std::string label = node.user_label();
  return label.empty() ? node.kind() : node.kind() + " '" + label + "'";
}

// ---- seeds -------------------------------------------------------------------------------------
// The seed of a seeded build is the Run's representative input: it binds the input's spec and is
// not pushed (GStreamer's preflight pushes and pulls it once, which leaves no output behind
// either).
struct Seed {
  std::optional<Tensor> tensor;
  std::string unsupported; // a seed this backend cannot carry
};

Seed seed_from_sample(const Sample& sample) {
  Seed seed;
  if (sample.kind == SampleKind::Bundle) {
    if (sample.fields.size() == 1U) {
      return seed_from_sample(sample.fields.front());
    }
    seed.unsupported = "a Bundle seed of " + std::to_string(sample.fields.size()) +
                       " fields (a record) is not supported yet";
    return seed;
  }
  if (sample_has_tensor_list(sample)) {
    if (sample.tensors.size() == 1U) {
      seed.tensor = sample.tensors.front();
      return seed;
    }
    seed.unsupported = "a seed of " + std::to_string(sample.tensors.size()) +
                       " tensors (a record) is not supported yet";
    return seed;
  }
  if (sample.tensor.has_value()) {
    seed.tensor = *sample.tensor;
    return seed;
  }
  seed.unsupported = "the seed Sample has no tensor";
  return seed;
}

// ---- qualification (plan 3.6) ------------------------------------------------------------------
struct Qualification {
  std::string reason; // empty: qualifies
  std::size_t input = 0;
  std::size_t output = 0;
};

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace

// What the compiler reads of a Graph's private composition (Graph::CompositionView), copied out by
// its friend DataflowGraphAccess. Hidden as DataflowGraphAccess is: Core-internal.
struct __attribute__((visibility("hidden"))) DataflowGraphView {
  struct Edge {
    std::size_t from = 0;
    std::size_t to = 0;
    GraphLinkPolicy policy = GraphLinkPolicy::Default;
  };
  bool model_route = false; // Graph::input_route_processor_
  bool linear = true;
  std::string name;
  std::vector<std::shared_ptr<Node>> vertices;
  std::vector<std::string> runtime_stages;         // kinds of runtime-stage vertices
  std::vector<std::optional<InputOptions>> inputs; // per vertex: a public Input's resolved options
  std::vector<Edge> edges;
  struct Named {
    std::size_t start = 0;
    std::size_t end = 0;
    std::string name;
  };
  std::vector<Named> named_fragments; // named Graph fragments, [start, end) vertex ranges
};

DataflowGraphView DataflowGraphAccess::view(const Graph& graph) {
  const auto composition = graph.composition_view_for_internal_compile();
  DataflowGraphView out;
  out.model_route = static_cast<bool>(graph.input_route_processor_);
  out.linear = composition.linear;
  out.name = composition.graph_name;
  out.vertices = composition.vertices;
  for (const auto& stage : composition.runtime_vertices) {
    if (stage) {
      out.runtime_stages.push_back(stage->kind());
    }
  }
  out.inputs.resize(composition.vertices.size());
  for (std::size_t vertex = 0; vertex < composition.vertices.size(); ++vertex) {
    if (dynamic_cast<const Input*>(composition.vertices[vertex].get())) {
      // The options compile_public_graph gives this Input (ExecutionGraphCompiler.cpp:3253-3267).
      out.inputs[vertex] = pipeline_internal::public_input_options(composition, vertex);
    }
  }
  for (const auto& edge : composition.edges) {
    out.edges.push_back({edge.from, edge.to, edge.link_options.policy});
  }
  for (const auto& named : composition.named_fragments) {
    out.named_fragments.push_back({named.start, named.end, named.name});
  }
  return out;
}

#if defined(SIMANEAT_WITH_DATAFLOW)

namespace {

// Exactly one public Input and one public Output, one connection between them, nothing else.
Qualification qualify(const DataflowGraphView& view) {
  Qualification q;
  if (view.model_route) {
    q.reason = "a Model input route is not supported before H1";
    return q;
  }
  if (!view.runtime_stages.empty()) {
    q.reason = "runtime stage " + view.runtime_stages.front() + " is not supported before H1";
    return q;
  }
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  for (std::size_t vertex = 0; vertex < view.vertices.size(); ++vertex) {
    const auto& node = view.vertices[vertex];
    if (!node) {
      continue;
    }
    if (dynamic_cast<const Input*>(node.get())) {
      inputs.push_back(vertex);
    } else if (dynamic_cast<const Output*>(node.get())) {
      outputs.push_back(vertex);
    } else {
      q.reason = "node " + node_text(*node) + " is not supported before H1";
      return q;
    }
  }
  if (inputs.empty()) {
    q.reason = "a graph with no public Input (a source graph) is not supported before H1";
    return q;
  }
  if (outputs.empty()) {
    q.reason = "a graph with no public Output is not supported before H1";
    return q;
  }
  if (inputs.size() > 1U || outputs.size() > 1U) {
    q.reason = std::to_string(inputs.size()) + " public Inputs and " +
               std::to_string(outputs.size()) +
               " public Outputs: more than one Input or Output is not supported before H0c";
    return q;
  }
  q.input = inputs.front();
  q.output = outputs.front();
  bool connected = false;
  for (const auto& edge : view.edges) {
    if (edge.from != q.input || edge.to != q.output) {
      q.reason = "an edge other than Input -> Output (fan-in or fan-out) is not supported "
                 "before H0c";
      return q;
    }
    if (edge.policy != GraphLinkPolicy::Default) {
      q.reason = "a realtime link policy is not supported before H1";
      return q;
    }
    connected = true;
  }
  if (!connected) {
    q.reason = "the public Input and Output are not connected";
    return q;
  }
  const auto* output = dynamic_cast<const Output*>(view.vertices[q.output].get());
  if (output->options().combine_policy != CombinePolicy::None) {
    q.reason = "an Output CombinePolicy (fan-in) is not supported before H0c";
    return q;
  }
  return q;
}

// The public names of one endpoint vertex, as compile_public_graph names it: its explicit name,
// else the name of each named fragment holding it (ExecutionGraphCompiler.cpp:2704-2855), kept only
// when the Graph lists it (its prune to Graph::inputs()/outputs(), :3330-3346). Sorted.
std::vector<std::string> endpoint_names(const DataflowGraphView& view, std::size_t vertex,
                                        const std::vector<std::string>& listed) {
  std::vector<std::string> names;
  const auto& node = view.vertices[vertex];
  std::string explicit_name;
  if (const auto* input = dynamic_cast<const Input*>(node.get())) {
    explicit_name = input->endpoint_name();
  } else if (const auto* output = dynamic_cast<const Output*>(node.get())) {
    explicit_name = output->endpoint_name();
  }
  if (!explicit_name.empty()) {
    names.push_back(explicit_name);
  } else {
    for (const auto& named : view.named_fragments) {
      if (!named.name.empty() && named.start <= vertex && vertex < named.end) {
        names.push_back(named.name);
      }
    }
  }
  std::erase_if(names, [&](const std::string& name) {
    return std::find(listed.begin(), listed.end(), name) == listed.end();
  });
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

// The options Dataflow does not read (plan 3.4), when set away from their defaults: one warning
// per process and option.
void warn_ignored_options(const DataflowRunConfig& config) {
  const InputOptions defaults{};
  const auto ignored = [](const char* option) {
    warn_once(std::string("ignored:") + option,
              std::string("Dataflow backend: ") + option +
                  " has no meaning on this backend and is ignored");
  };
  const InputOptions& in = config.input;
  if (in.is_live != defaults.is_live)
    ignored("InputOptions::is_live");
  if (in.do_timestamp != defaults.do_timestamp)
    ignored("InputOptions::do_timestamp");
  if (in.stream_type != defaults.stream_type)
    ignored("InputOptions::stream_type");
  if (!in.caps_override.empty())
    ignored("InputOptions::caps_override");
  if (in.max_bytes != defaults.max_bytes)
    ignored("InputOptions::max_bytes");
  if (in.block != defaults.block)
    ignored("InputOptions::block");
  if (in.fps_n != defaults.fps_n || in.fps_d != defaults.fps_d)
    ignored("InputOptions::fps_n/fps_d");
  if (!in.buffer_name.empty())
    ignored("InputOptions::buffer_name");
  if (in.preprocess_meta.has_value() && in.preprocess_meta->enabled)
    ignored("InputOptions::preprocess_meta");
  if (config.output.sync)
    ignored("OutputOptions::sync");
}

DataflowRunConfig derive_config(const Graph& graph, const DataflowGraphView& view,
                                const Qualification& q, const RunOptions& opt) {
  DataflowRunConfig config;
  config.input = *view.inputs[q.input];
  config.output = dynamic_cast<const Output*>(view.vertices[q.output].get())->options();
  config.run = session_build_resolve_build_opt(RunMode::Async, opt);
  config.run.backend = RunBackend::Dataflow;

  // One link carries both of Core's queues (Run.h:157-169): the public Output's max_buffers wins
  // for its depth; its drop flag makes it lossy for every push, as appsink drop=true is;
  // otherwise RunOptions' overflow policy applies to try_push (KeepLatest is DropOldest).
  if (config.output.max_buffers > 0) {
    config.link_depth = static_cast<std::uint32_t>(config.output.max_buffers);
  } else {
    config.link_depth =
        config.run.queue_depth > 0 ? static_cast<std::uint32_t>(config.run.queue_depth) : 4U;
    warn_once("clamp:max_buffers",
              "Dataflow backend: OutputOptions::max_buffers=" +
                  std::to_string(config.output.max_buffers) +
                  " (unbounded) has no equivalent on this backend; the link depth is clamped to " +
                  std::to_string(config.link_depth) + " (RunOptions::queue_depth, or 4)");
  }
  config.terminal_drop = config.output.drop;
  config.link_policy = config.run.overflow_policy;

  config.input_names = endpoint_names(view, q.input, graph.inputs());
  config.output_names = endpoint_names(view, q.output, graph.outputs());

  // GStreamer stamps an empty stream_id with the input's public name, or "default" when it has none
  // (RunPush.cpp:225-264), on its graph path, which a connected or named graph takes; a plain
  // linear Run leaves it empty.
  const bool graph_path =
      !view.linear || !config.input_names.empty() || !config.output_names.empty();
  if (graph_path) {
    config.default_stream = config.input_names.empty() ? "default" : config.input_names.front();
  }
  config.label = view.name;
  return config;
}

std::shared_ptr<RunCore> build_impl(const Graph& graph, const RunOptions& opt, RunMode mode,
                                    const Seed& seed, const char* where, const Want& want) {
  if (mode != RunMode::Async) {
    return decline(want, where, error_codes::kPipelineShape,
                   "synchronous runs (Graph::run) are not supported before H1");
  }
  const DataflowGraphView view = DataflowGraphAccess::view(graph);
  const Qualification q = qualify(view);
  if (!q.reason.empty()) {
    return decline(want, where, error_codes::kPipelineShape, q.reason);
  }
  if (!seed.unsupported.empty()) {
    return decline(want, where, error_codes::kInputShape, seed.unsupported);
  }
  DataflowRunConfig config = derive_config(graph, view, q, opt);
  // The input's spec, when a seed or the declaration gives it now; an unsupported declaration or
  // seed is decided here, before a Run exists.
  std::string why;
  const std::optional<DataflowInputBinding> binding =
      bind_dataflow_input(config.input, seed.tensor ? &*seed.tensor : nullptr, &why);
  if (!why.empty()) {
    return decline(want, where, error_codes::kInputShape, why);
  }
  warn_ignored_options(config);
  trace_choice(
      where, "dataflow",
      "Input -> Output, link depth " + std::to_string(config.link_depth) +
          (config.terminal_drop                                 ? ", drop oldest (Output drop)"
           : config.link_policy == OverflowPolicy::KeepLatest   ? ", try_push keep latest"
           : config.link_policy == OverflowPolicy::DropIncoming ? ", try_push drop incoming"
                                                                : ", block") +
          (binding ? ", input " + binding->text : ", input bound by the first push") + ", inputs " +
          join(config.input_names) + ", outputs " + join(config.output_names));

  auto core = std::make_shared<RunCore>();
  initialize_run_identity(*core);
  core->opt = config.run;
  core->mode = RunMode::Async;
  core->dataflow_ =
      make_dataflow_run_state(*core, std::move(config), seed.tensor ? &*seed.tensor : nullptr);
  return core;
}

} // namespace
#endif // SIMANEAT_WITH_DATAFLOW

namespace {

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
  return dispatch(opt, where, [&](const auto& want) -> std::shared_ptr<RunCore> {
#if defined(SIMANEAT_WITH_DATAFLOW)
    return build_impl(graph, opt, mode, Seed{}, where, want);
#else
    (void)graph;
    (void)mode;
    (void)want;
    return nullptr;
#endif
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const TensorList& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) -> std::shared_ptr<RunCore> {
#if defined(SIMANEAT_WITH_DATAFLOW)
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty tensor list");
    }
    Seed s;
    if (seed.size() == 1U) {
      s.tensor = seed.front();
    } else {
      s.unsupported =
          "a seed of " + std::to_string(seed.size()) + " tensors (a record) is not supported yet";
    }
    return build_impl(graph, opt, mode, s, where, want);
#else
    (void)graph;
    (void)mode;
    (void)seed;
    (void)want;
    return nullptr;
#endif
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const Sample& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) -> std::shared_ptr<RunCore> {
#if defined(SIMANEAT_WITH_DATAFLOW)
    if (seed.empty()) {
      throw std::runtime_error("Graph::build(inputs): empty sample list");
    }
    return build_impl(graph, opt, mode, seed_from_sample(seed), where, want);
#else
    (void)graph;
    (void)mode;
    (void)seed;
    (void)want;
    return nullptr;
#endif
  });
}

std::shared_ptr<RunCore> DataflowGraphAccess::build(const Graph& graph, const RunOptions& opt,
                                                    RunMode mode, const std::vector<cv::Mat>& seed,
                                                    const char* where) {
  return dispatch(opt, where, [&](const auto& want) -> std::shared_ptr<RunCore> {
#if defined(SIMANEAT_WITH_DATAFLOW) && defined(SIMA_WITH_OPENCV)
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
    Seed s;
    if (seed.size() != 1U) {
      s.unsupported =
          "a seed of " + std::to_string(seed.size()) + " images (a record) is not supported yet";
      return build_impl(graph, opt, mode, s, where, want);
    }
    // The seed only binds the spec: a CPU view, whatever the Input's memory policy.
    const DataflowGraphView view = DataflowGraphAccess::view(graph);
    const Qualification q = qualify(view);
    InputOptions input = q.reason.empty() ? *view.inputs[q.input] : InputOptions{};
    input.memory_policy = InputMemoryPolicy::SystemMemory;
    s.tensor = tensor_from_cv_mat(seed.front(), input, "Graph::build(inputs)");
    return build_impl(graph, opt, mode, s, where, want);
#else
    (void)graph;
    (void)mode;
    (void)seed;
    (void)want;
    return nullptr;
#endif
  });
}

#if defined(SIMANEAT_WITH_DATAFLOW)

// ---- the input's spec (plan 3.4: InputOptions and the seed give shape, bounds and format) -------

namespace {

std::optional<df::DType> dataflow_dtype(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return df::DType::UInt8;
  case TensorDType::Int8:
    return df::DType::Int8;
  case TensorDType::UInt16:
    return df::DType::UInt16;
  case TensorDType::Int16:
    return df::DType::Int16;
  case TensorDType::Int32:
    return df::DType::Int32;
  case TensorDType::BFloat16:
    return df::DType::BFloat16;
  case TensorDType::Float32:
    return df::DType::Float32;
  case TensorDType::Float64:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<ImageSpec::PixelFormat> pixel_format(FormatTag tag) {
  switch (tag) {
  case FormatTag::RGB:
    return ImageSpec::PixelFormat::RGB;
  case FormatTag::BGR:
    return ImageSpec::PixelFormat::BGR;
  case FormatTag::GRAY8:
    return ImageSpec::PixelFormat::GRAY8;
  case FormatTag::NV12:
    return ImageSpec::PixelFormat::NV12;
  case FormatTag::I420:
    return ImageSpec::PixelFormat::I420;
  case FormatTag::NV16:
    return ImageSpec::PixelFormat::NV16;
  default:
    return std::nullopt;
  }
}

// The declared format first, then the tensor's own, as describe_tensor_geometry_or_throw reads
// them (InputStreamUtil.cpp:1513-1516).
std::optional<ImageSpec::PixelFormat> image_format(const InputOptions& declared,
                                                   const Tensor* tensor) {
  if (auto fmt = pixel_format(declared.format.tag)) {
    return fmt;
  }
  if (tensor && tensor->semantic.image.has_value() &&
      tensor->semantic.image->format != ImageSpec::PixelFormat::UNKNOWN) {
    return tensor->semantic.image->format;
  }
  return std::nullopt;
}

// One axis: max_* bounds it (min 1), a declared size fixes it, otherwise the tensor's size does.
std::optional<df::Dim> axis_dim(int declared, int max, std::int64_t actual, std::string* why,
                                const char* axis) {
  if (max > 0) {
    if (actual > max) {
      *why = std::string("the seed's ") + axis + " " + std::to_string(actual) +
             " exceeds InputOptions::max_" + axis + " " + std::to_string(max);
      return std::nullopt;
    }
    return df::Dim(1, max);
  }
  if (declared > 0) {
    return df::Dim(declared);
  }
  if (actual > 0) {
    return df::Dim(actual);
  }
  return std::nullopt;
}

// Which axes are height, width and channels, as tensor_compat_dims_from_shape reads a shape
// (InputStreamUtil.cpp:115-160). -1: none.
struct AxisRoles {
  int height = -1;
  int width = -1;
  int depth = -1;
};

AxisRoles axis_roles(std::size_t rank, TensorLayout layout) {
  const int r = static_cast<int>(rank);
  AxisRoles roles;
  if (layout == TensorLayout::HW) {
    roles.height = r >= 2 ? r - 2 : -1;
    roles.width = r - 1;
  } else if (layout == TensorLayout::CHW) {
    roles.depth = r >= 3 ? r - 3 : -1;
    roles.height = r >= 2 ? r - 2 : -1;
    roles.width = r - 1;
  } else if (r >= 3) {
    roles.height = r - 3;
    roles.width = r - 2;
    roles.depth = r - 1;
  } else if (r == 2) {
    roles.height = 0;
    roles.width = 1;
  } else {
    roles.width = 0;
  }
  return roles;
}

// Planar NV12 or I420 at the bound: planes packed back to back at the bound's pitches.
df::Planar planar_layout(df::PixelFormat format, std::int64_t max_h, std::int64_t max_w) {
  df::Planar planar;
  planar.format = format;
  const std::int64_t half_w = (max_w + 1) / 2;
  const std::int64_t half_h = (max_h + 1) / 2;
  planar.plane[0] = {0, max_w};
  if (format == df::PixelFormat::NV12) {
    planar.planes = 2;
    planar.plane[1] = {max_w * max_h, half_w * 2};
  } else {
    planar.planes = 3;
    planar.plane[1] = {max_w * max_h, half_w};
    planar.plane[2] = {max_w * max_h + half_h * half_w, half_w};
  }
  return planar;
}

std::string describe_spec(const df::PortSpec& spec) {
  std::array<char, df::kDescribeMax> buffer{};
  df::Writer writer{std::span<char>(buffer)};
  df::describe(spec, writer);
  return std::string(writer.view());
}

} // namespace

std::optional<DataflowInputBinding> bind_dataflow_input(const InputOptions& declared,
                                                        const Tensor* tensor, std::string* why) {
  why->clear();
  if (declared.memory_policy == InputMemoryPolicy::Ev74 ||
      declared.memory_policy == InputMemoryPolicy::Dms0) {
    *why = "InputOptions::memory_policy Ev74 or Dms0 (device memory) is not supported before H0b";
    return std::nullopt;
  }
  if (declared.payload_type == PayloadType::Encoded || declared.format.tag == FormatTag::H264 ||
      declared.format.tag == FormatTag::H265 || declared.format.tag == FormatTag::ENCODED) {
    *why = "encoded input is not supported before H1";
    return std::nullopt;
  }
  const std::optional<ImageSpec::PixelFormat> fmt = image_format(declared, tensor);
  if (fmt == ImageSpec::PixelFormat::NV16) {
    *why = "NV16 input is not supported before H1";
    return std::nullopt;
  }
  const bool nv12 = fmt == ImageSpec::PixelFormat::NV12;
  const bool i420 = fmt == ImageSpec::PixelFormat::I420;

  DataflowInputBinding binding;
  df::TensorSpec spec;
  if (tensor) {
    if (tensor->has_tiled_storage()) {
      *why = "tiled (MLA) tensors are not supported before H1";
      return std::nullopt;
    }
    if (tensor->semantic.encoded.has_value() || tensor->semantic.byte_stream.has_value()) {
      *why = "encoded and byte-stream payloads are not supported before H1";
      return std::nullopt;
    }
    const auto dtype = dataflow_dtype(tensor->dtype);
    if (!dtype) {
      *why = "Float64 tensors have no Dataflow dtype";
      return std::nullopt;
    }
    if (tensor->is_composite() != (nv12 || i420)) {
      *why = tensor->is_composite()
                 ? "a composite tensor other than NV12 or I420 is not supported before H1"
                 : "an NV12 or I420 tensor without plane records is not supported";
      return std::nullopt;
    }
    binding.dtype = tensor->dtype;
    binding.layout = tensor->layout;
    binding.axis_semantics = tensor->axis_semantics;
    binding.semantic = tensor->semantic;
    binding.semantic.preprocess.reset(); // per-frame runtime metadata, not part of the spec
    spec.dtype = *dtype;
    if (nv12 || i420) {
      const int h = tensor->height();
      const int w = tensor->width();
      const auto hd = axis_dim(declared.height, declared.max_height, h, why, "height");
      const auto wd = axis_dim(declared.width, declared.max_width, w, why, "width");
      if (!hd || !wd) {
        if (why->empty())
          *why = "the seed has no width or height";
        return std::nullopt;
      }
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.rank = 2;
    } else {
      const std::size_t rank = tensor->shape.size();
      if (rank == 0U || rank > df::kMaxRank) {
        *why = "a tensor of rank " + std::to_string(rank) + " is not supported (1 to 6)";
        return std::nullopt;
      }
      const AxisRoles roles = axis_roles(rank, tensor->layout);
      spec.shape.rank = static_cast<std::uint8_t>(rank);
      for (std::size_t i = 0; i < rank; ++i) {
        const int axis = static_cast<int>(i);
        const std::int64_t actual = tensor->shape[i];
        std::optional<df::Dim> dim;
        if (axis == roles.height) {
          dim = axis_dim(declared.height, declared.max_height, actual, why, "height");
        } else if (axis == roles.width) {
          dim = axis_dim(declared.width, declared.max_width, actual, why, "width");
        } else if (axis == roles.depth && !fmt) {
          dim = axis_dim(declared.depth, declared.max_depth, actual, why, "depth");
        } else if (actual > 0) {
          dim = df::Dim(actual); // channels of an image format, batch and other axes: fixed
        }
        if (!dim) {
          if (why->empty())
            *why = "axis " + std::to_string(i) + " of the seed is empty";
          return std::nullopt;
        }
        spec.shape.dims[i] = *dim;
      }
      binding.rank = rank;
    }
  } else {
    // Declaration only: an image with its format and both sizes (or bounds) declared.
    if (!fmt || (declared.payload_type != PayloadType::Image &&
                 declared.payload_type != PayloadType::Auto)) {
      return std::nullopt; // bound by the first push
    }
    std::string unused;
    const auto hd = axis_dim(declared.height, declared.max_height, 0, &unused, "height");
    const auto wd = axis_dim(declared.width, declared.max_width, 0, &unused, "width");
    if (!hd || !wd) {
      return std::nullopt;
    }
    spec.dtype = df::DType::UInt8;
    binding.dtype = TensorDType::UInt8;
    binding.semantic.image = ImageSpec{*fmt, ""};
    if (nv12 || i420) {
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.layout = TensorLayout::HW;
      binding.rank = 2;
    } else if (*fmt == ImageSpec::PixelFormat::GRAY8) {
      spec.shape = df::Shape::of({*hd, *wd});
      binding.layout = TensorLayout::HW;
      binding.rank = 2;
    } else {
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.layout = TensorLayout::HWC;
      binding.rank = 3;
    }
  }
  if (fmt && !binding.semantic.image.has_value()) {
    binding.semantic.image = ImageSpec{*fmt, ""};
  } else if (fmt) {
    binding.semantic.image->format = *fmt;
  }
  if (nv12 || i420) {
    binding.planes = nv12 ? DataflowInputBinding::Planes::Nv12 : DataflowInputBinding::Planes::I420;
    spec.dtype = df::DType::UInt8;
    spec.layout = planar_layout(nv12 ? df::PixelFormat::NV12 : df::PixelFormat::I420,
                                spec.shape.dims[0].max, spec.shape.dims[1].max);
  } else {
    spec.layout = df::Dense{};
  }
  binding.spec.payload = spec;
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("the input's spec is not valid: ") + status->message;
    return std::nullopt;
  }
  binding.text = describe_spec(binding.spec);
  return binding;
}

std::string dataflow_binding_mismatch(const DataflowInputBinding& binding, const Tensor& tensor) {
  const auto& spec = std::get<df::TensorSpec>(binding.spec.payload);
  const auto mismatch = [&](const std::string& what) {
    return what + "; the input is bound to " + binding.text +
           " (Dataflow backend: by its declaration, its seed or its first push)";
  };
  if (tensor.dtype != binding.dtype) {
    return mismatch("dtype differs");
  }
  if (binding.semantic.image.has_value() && tensor.semantic.image.has_value() &&
      tensor.semantic.image->format != ImageSpec::PixelFormat::UNKNOWN &&
      tensor.semantic.image->format != binding.semantic.image->format) {
    return mismatch(std::string("image format ") +
                    pipeline_internal::pixel_format_token(tensor.semantic.image->format) +
                    " differs");
  }
  if (binding.planes != DataflowInputBinding::Planes::None) {
    if (!tensor.is_composite()) {
      return mismatch("a dense tensor where planes are bound");
    }
    const std::int64_t h = tensor.height();
    const std::int64_t w = tensor.width();
    if (!spec.shape.dims[0].contains(h) || !spec.shape.dims[1].contains(w)) {
      return mismatch("size " + std::to_string(w) + "x" + std::to_string(h) + " is out of bounds");
    }
    if ((h % 2) != 0 || (w % 2) != 0) {
      return mismatch("odd size " + std::to_string(w) + "x" + std::to_string(h));
    }
    return {};
  }
  if (tensor.is_composite()) {
    return mismatch("a composite tensor where a dense one is bound");
  }
  if (tensor.shape.size() != spec.shape.rank) {
    return mismatch("rank " + std::to_string(tensor.shape.size()) + " differs");
  }
  for (std::size_t i = 0; i < spec.shape.rank; ++i) {
    if (!spec.shape.dims[i].contains(tensor.shape[i])) {
      return mismatch("dimension " + std::to_string(i) + " = " + std::to_string(tensor.shape[i]) +
                      " is out of bounds");
    }
  }
  return {};
}

DataflowNet compile_dataflow_net(const DataflowRunConfig& config,
                                 const DataflowInputBinding& binding, std::uint32_t stream) {
  if (config.link_depth == 0U || config.link_depth > df::kMaxLinkDepth) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kOptionOutOfRange,
        "Graph::build: the Output's max_buffers " + std::to_string(config.link_depth) +
            " is outside the Dataflow link's range 1.." + std::to_string(df::kMaxLinkDepth));
  }
  df::Overflow overflow = df::Overflow::Block;
  if (config.terminal_drop || config.link_policy == OverflowPolicy::KeepLatest) {
    overflow = df::Overflow::DropOldest;
  } else if (config.link_policy == OverflowPolicy::DropIncoming) {
    overflow = df::Overflow::DropIncoming;
  }

  df::Net net;
  const df::OperatorRef<df::Feed> feed = net.add<df::Feed>(
      "input", binding.spec,
      df::Feed::Options{.stream = df::StreamId{stream}, .loans = 1, .max_links = 0});
  const df::OperatorRef<df::Fetch> fetch =
      net.add<df::Fetch>("output", binding.spec, df::Fetch::Options{.retained = 1});
  if (const df::Status status = net.connect(
          feed.out(), fetch.in(),
          df::LinkPolicy{.depth = config.link_depth, .overflow = overflow, .stale_after = {}})) {
    pipeline_internal::throw_dataflow_error(*status, "Graph::build");
  }
  df::PlanReport failure;
  df::Result<df::Plan> plan = std::move(net).compile(df::CompileOptions{
      .force_fine_clock = false, .retention_limit_bytes = 0, .failure_report = &failure});
  if (const df::Error* error = df::failure(plan)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build", failure.mismatch);
  }
  return DataflowNet{std::get<df::Plan>(std::move(plan)), feed.index, fetch.index};
}

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace simaai::neat::runtime

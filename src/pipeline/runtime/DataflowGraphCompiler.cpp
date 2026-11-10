// The Dataflow backend's compiler (H0 plan 3.4, 3.6; H0 step 10, H0c): RunOptions::backend and
// SIMA_NEAT_BACKEND decide the runtime of a Graph::build(). A qualifying graph has public Inputs
// and Outputs only, each Output fed by exactly one Input, after the public boundaries inside it are
// elided as compile_public_graph elides them. Each Input becomes one Feed with one link per Output
// it feeds (a fan-out when there are several) and one Fetch per Output, on a Session of its own, in
// one Run state (DataflowRun.cpp). Everything else (fan-in and combine, operators, device memory)
// stays on GStreamer, or fails loudly when RunBackend::Dataflow was asked for explicitly.

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
#include <functional>
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
// either). One tensor binds a leaf, several a record.
struct Seed {
  TensorList tensors;      // empty: no seed
  std::string unsupported; // a seed this backend cannot carry
};

Seed seed_from_sample(const Sample& sample) {
  Seed seed;
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
    seed.tensors.push_back(*sample.tensor);
    return seed;
  }
  seed.unsupported = "the seed Sample has no tensor";
  return seed;
}

// ---- qualification (plan 3.6) ------------------------------------------------------------------
// The public endpoints of a qualifying graph, and who feeds whom.
struct Qualification {
  std::string reason;                // empty: qualifies
  std::vector<std::size_t> inputs;   // public Input vertices, vertex order
  std::vector<std::size_t> outputs;  // public Output vertices, vertex order
  std::vector<std::size_t> producer; // per output (index into outputs): its input (into inputs)
  std::vector<std::size_t> fan_out;  // per input: how many outputs it feeds
};

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace

// What the compiler reads of a Graph's private composition (Graph::CompositionView), copied out by
// its friend DataflowGraphAccess. Hidden as DataflowGraphAccess is: Core-internal.
struct __attribute__((visibility("hidden"))) DataflowGraphView {
  struct Edge {
    std::size_t from = 0;
    std::size_t to = 0;
    bool implicit = false; // Graph::add's chain (ImplicitLinear); every other edge is explicit
    GraphLinkPolicy policy = GraphLinkPolicy::Default;
    std::string stream_id; // the edge's stream identity, or its link options'
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
    bool user_named = false;
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
    out.edges.push_back({edge.from, edge.to,
                         edge.kind == Graph::CompositionEdgeKind::ImplicitLinear,
                         edge.link_options.policy,
                         edge.stream_id.empty() ? edge.link_options.stream_id : edge.stream_id});
  }
  for (const auto& named : composition.named_fragments) {
    out.named_fragments.push_back({named.start, named.end, named.name, named.user_named});
  }
  return out;
}

#if defined(SIMANEAT_WITH_DATAFLOW)

namespace {

// The public endpoints of the graph and who feeds whom, after compile_public_graph's boundary
// elision (normalize_public_boundaries_for_execution, ExecutionGraphCompiler.cpp:427-650): an
// Input with an explicit upstream connection and an Output with an explicit downstream connection
// are internal boundary declarations (a Branch's, a connected fragment's), not endpoints, and an
// edge into one continues along its outgoing edges. What is left must be public Inputs feeding
// public Outputs, each Output fed by exactly one Input: one Feed per Input, one link and one
// Fetch per Output it feeds.
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
  const std::size_t n = view.vertices.size();
  enum class Kind { None, In, Out };
  std::vector<Kind> kind(n, Kind::None);
  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    const auto& node = view.vertices[vertex];
    if (!node) {
      continue;
    }
    if (dynamic_cast<const Input*>(node.get())) {
      kind[vertex] = Kind::In;
    } else if (dynamic_cast<const Output*>(node.get())) {
      kind[vertex] = Kind::Out;
    } else {
      q.reason = "node " + node_text(*node) + " is not supported before H1";
      return q;
    }
  }
  const auto text = [&](std::size_t vertex) {
    return view.vertices[vertex] ? node_text(*view.vertices[vertex])
                                 : std::string("an empty vertex");
  };
  std::vector<std::size_t> explicit_in(n);
  std::vector<std::size_t> explicit_out(n);
  std::vector<std::vector<std::size_t>> outgoing(n);
  for (std::size_t e = 0; e < view.edges.size(); ++e) {
    const auto& edge = view.edges[e];
    if (edge.from >= n || edge.to >= n) {
      q.reason = "a composition edge references an invalid vertex";
      return q;
    }
    outgoing[edge.from].push_back(e);
    if (!edge.implicit) {
      ++explicit_in[edge.to];
      ++explicit_out[edge.from];
    }
  }
  std::vector<bool> elide(n, false);
  // A public endpoint's index in q.inputs / q.outputs; read only for public endpoints.
  std::vector<std::size_t> input_of(n);
  std::vector<std::size_t> output_of(n);
  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    elide[vertex] = (kind[vertex] == Kind::In && explicit_in[vertex] > 0U) ||
                    (kind[vertex] == Kind::Out && explicit_out[vertex] > 0U);
    if (elide[vertex]) {
      continue;
    }
    if (kind[vertex] == Kind::In) {
      input_of[vertex] = q.inputs.size();
      q.inputs.push_back(vertex);
    } else if (kind[vertex] == Kind::Out) {
      output_of[vertex] = q.outputs.size();
      q.outputs.push_back(vertex);
    }
  }
  if (q.inputs.empty()) {
    q.reason = "a graph with no public Input (a source graph) is not supported before H1";
    return q;
  }
  if (q.outputs.empty()) {
    q.reason = "a graph with no public Output is not supported before H1";
    return q;
  }

  // Every edge leaving a vertex that is not elided, followed through the elided boundaries it
  // enters, to the endpoint it reaches.
  std::vector<std::pair<std::size_t, std::size_t>> links;
  std::vector<bool> visiting(n, false);
  std::function<bool(std::size_t, std::size_t)> follow = [&](std::size_t start, std::size_t e) {
    const auto& edge = view.edges[e];
    if (edge.policy != GraphLinkPolicy::Default) {
      q.reason = "a realtime link policy is not supported before H1";
      return false;
    }
    if (!edge.stream_id.empty()) {
      q.reason = "a link stream_id ('" + edge.stream_id + "') is not supported before H1";
      return false;
    }
    const std::size_t to = edge.to;
    if (!elide[to]) {
      links.emplace_back(start, to);
      return true;
    }
    if (kind[to] == Kind::Out &&
        dynamic_cast<const Output*>(view.vertices[to].get())->options().combine_policy !=
            CombinePolicy::None) {
      q.reason = "an Output CombinePolicy (fan-in) is not supported before H1";
      return false;
    }
    if (visiting[to]) {
      q.reason = "the public boundary declarations form a cycle";
      return false;
    }
    if (outgoing[to].empty()) {
      q.reason = "the internal boundary " + text(to) + " leads to no endpoint";
      return false;
    }
    visiting[to] = true;
    for (const std::size_t next : outgoing[to]) {
      if (!follow(start, next)) {
        return false;
      }
    }
    visiting[to] = false;
    return true;
  };
  for (std::size_t e = 0; e < view.edges.size(); ++e) {
    if (!elide[view.edges[e].from] && !follow(view.edges[e].from, e)) {
      return q;
    }
  }

  constexpr std::size_t kNone = static_cast<std::size_t>(-1);
  q.producer.assign(q.outputs.size(), kNone);
  for (const auto& [from, to] : links) {
    if (kind[from] != Kind::In || kind[to] != Kind::Out) {
      q.reason = "a connection from " + text(from) + " to " + text(to) +
                 " (only Inputs feeding Outputs) is not supported before H1";
      return q;
    }
    if (dynamic_cast<const Output*>(view.vertices[to].get())->options().combine_policy !=
        CombinePolicy::None) {
      q.reason = "an Output CombinePolicy (fan-in) is not supported before H1";
      return q;
    }
    std::size_t& producer = q.producer[output_of[to]];
    if (producer != kNone) {
      q.reason = text(to) + " has more than one producer: fan-in is not supported before H1";
      return q;
    }
    producer = input_of[from];
  }
  q.fan_out.resize(q.inputs.size());
  for (std::size_t o = 0; o < q.outputs.size(); ++o) {
    if (q.producer[o] == kNone) {
      q.reason = text(q.outputs[o]) + " is connected to no Input";
      return q;
    }
    ++q.fan_out[q.producer[o]];
  }
  for (std::size_t i = 0; i < q.inputs.size(); ++i) {
    if (q.fan_out[i] == 0U) {
      q.reason = text(q.inputs[i]) + " is connected to no Output";
      return q;
    }
    if (q.fan_out[i] > df::kMaxFanOut) {
      q.reason = text(q.inputs[i]) + " feeds " + std::to_string(q.fan_out[i]) +
                 " Outputs: the runtime's fan-out is at most " + std::to_string(df::kMaxFanOut);
      return q;
    }
  }
  return q;
}

// The public names of one endpoint vertex, as compile_public_graph names it: its explicit name;
// else, for each named fragment holding it, the fragment's name when it is the fragment's only
// public endpoint of its kind, or that name and its index among them ("classes_1",
// add_fragment_named_candidates, ExecutionGraphCompiler.cpp:2715-2760); kept only when the Graph
// lists it (its prune to Graph::inputs()/outputs()). Sorted. `endpoints` are the public
// endpoints of the vertex's kind.
std::vector<std::string> endpoint_names(const DataflowGraphView& view,
                                        const std::vector<std::size_t>& endpoints,
                                        std::size_t vertex,
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
    for (const bool user_named : {true, false}) {
      for (const auto& named : view.named_fragments) {
        if (named.user_named != user_named || named.name.empty() || vertex < named.start ||
            vertex >= named.end) {
          continue;
        }
        std::size_t index = 0;
        std::size_t count = 0;
        for (const std::size_t endpoint : endpoints) {
          if (endpoint >= named.start && endpoint < named.end) {
            if (endpoint == vertex) {
              index = count;
            }
            ++count;
          }
        }
        names.push_back(count > 1U ? named.name + "_" + std::to_string(index) : named.name);
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

// "" when no name is shared by two endpoints of one kind (DataflowInputConfig or
// DataflowOutputConfig); otherwise compile_public_graph's error (add_named_endpoint,
// ExecutionGraphCompiler.cpp:1376-1392).
template <class Endpoint>
std::string shared_name(const std::vector<Endpoint>& endpoints, const char* kind) {
  std::set<std::string> seen;
  for (const Endpoint& endpoint : endpoints) {
    for (const std::string& name : endpoint.names) {
      if (!seen.insert(name).second) {
        return "Graph endpoint name '" + name + "' is used by more than one " + kind +
               "; choose unique names for multi-input/multi-output graphs";
      }
    }
  }
  return {};
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
  for (const DataflowInputConfig& input : config.inputs) {
    const InputOptions& in = input.input;
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
  }
  for (const DataflowOutputConfig& output : config.outputs) {
    if (output.output.sync)
      ignored("OutputOptions::sync");
  }
}

// Every name of `endpoints`, sorted, once.
template <class Endpoint>
std::vector<std::string> sorted_union(const std::vector<Endpoint>& endpoints) {
  std::vector<std::string> out;
  for (const Endpoint& endpoint : endpoints) {
    out.insert(out.end(), endpoint.names.begin(), endpoint.names.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

const char* policy_text(OverflowPolicy policy) {
  switch (policy) {
  case OverflowPolicy::Block:
    return "Block";
  case OverflowPolicy::KeepLatest:
    return "KeepLatest";
  case OverflowPolicy::DropIncoming:
    return "DropIncoming";
  }
  return "?";
}

// The Run's configuration; `why` set when this graph's options are what Dataflow cannot carry.
DataflowRunConfig derive_config(const Graph& graph, const DataflowGraphView& view,
                                const Qualification& q, const RunOptions& opt, std::string* why) {
  DataflowRunConfig config;
  const std::vector<std::string> listed_inputs = graph.inputs();
  const std::vector<std::string> listed_outputs = graph.outputs();
  config.inputs.resize(q.inputs.size());
  config.outputs.resize(q.outputs.size());
  for (std::size_t i = 0; i < q.inputs.size(); ++i) {
    config.inputs[i].names = endpoint_names(view, q.inputs, q.inputs[i], listed_inputs);
  }
  for (std::size_t o = 0; o < q.outputs.size(); ++o) {
    config.outputs[o].names = endpoint_names(view, q.outputs, q.outputs[o], listed_outputs);
  }
  *why = shared_name(config.inputs, "input");
  if (why->empty()) {
    *why = shared_name(config.outputs, "output");
  }
  if (!why->empty()) {
    return config;
  }
  config.input_names = sorted_union(config.inputs);
  config.output_names = sorted_union(config.outputs);
  // The path GStreamer takes for the same graph: a connected or named graph is a graph Run, a plain
  // linear Run one pipeline segment (RunCore::start, RunCoreGraphStart.cpp:1566-1573, through
  // is_simple_linear_plan, :169-172).
  const bool graph_path =
      !view.linear || !config.input_names.empty() || !config.output_names.empty();

  // RunOptions resolved as GStreamer resolves them for that path, by the same code. A plain linear
  // Run's options are resolve_build_opt's (GraphBuildInput.cpp:1380-1414, via
  // prepare_build_input_context :1421, in RunCore::start_pipeline_segment RunCore.cpp:1661, kept
  // by start_single_pipeline Run.cpp:273): RunPreset fills queue_depth and overflow_policy where
  // they equal their RunOptions{} defaults, so an explicit Block under Realtime is KeepLatest
  // there. A graph Run keeps the caller's RunOptions as given (start_graph_plan
  // RunCoreGraphStart.cpp:1479; its ingress queue is the given queue_depth, RunCore.h:103): the
  // preset does not reach its ingress.
  config.run = graph_path ? opt : session_build_resolve_build_opt(RunMode::Async, opt);
  config.run.backend = RunBackend::Dataflow;
  config.link_policy = config.run.overflow_policy;

  // One link carries both of Core's queues on the way to each Output (Run.h:176-188). On
  // GStreamer a blocking push waits only when the input queue (RunOptions::queue_depth) and the
  // public Output's terminal queue (OutputOptions::max_buffers) are both full, so the link holds
  // their sum:
  //   link depth = queue_depth + max_buffers
  // each after its own default or clamp: queue_depth as resolved above, a value <= 0 (unbounded on
  // GStreamer) counting as 4, reported; max_buffers <= 0 (unbounded) counts as the queue_depth
  // term, reported (the depth is then 2 x queue_depth, 8 by default).
  // The Output's drop flag makes its link lossy for every push, as appsink drop=true is;
  // otherwise RunOptions' overflow policy applies to try_push (KeepLatest is DropOldest). Either
  // acts only when the link holds the whole depth. In a fan-out each Output's link has its own.
  std::uint32_t ingress_depth = 4U;
  if (config.run.queue_depth > 0) {
    ingress_depth = static_cast<std::uint32_t>(config.run.queue_depth);
  } else {
    warn_once(
        "clamp:queue_depth",
        "Dataflow backend: RunOptions::queue_depth=" + std::to_string(config.run.queue_depth) +
            " (unbounded) has no equivalent on this backend; it counts as " +
            std::to_string(ingress_depth) + " in the link depth");
  }
  for (std::size_t o = 0; o < q.outputs.size(); ++o) {
    DataflowOutputConfig& output = config.outputs[o];
    output.output = dynamic_cast<const Output*>(view.vertices[q.outputs[o]].get())->options();
    output.ingress_depth = ingress_depth;
    if (output.output.max_buffers > 0) {
      output.terminal_depth = static_cast<std::uint32_t>(output.output.max_buffers);
    } else {
      output.terminal_depth = ingress_depth;
      warn_once("clamp:max_buffers",
                "Dataflow backend: OutputOptions::max_buffers=" +
                    std::to_string(output.output.max_buffers) +
                    " (unbounded) has no equivalent on this backend; it counts as " +
                    std::to_string(output.terminal_depth) +
                    " (RunOptions::queue_depth, or 4) in the link depth");
    }
    // Each term is at most INT_MAX, so the sum fits; compile_dataflow_net checks the link's range.
    output.link_depth = output.ingress_depth + output.terminal_depth;
    output.terminal_drop = output.output.drop;
    output.input = q.producer[o];
  }
  for (std::size_t i = 0; i < q.inputs.size(); ++i) {
    DataflowInputConfig& input = config.inputs[i];
    input.input = *view.inputs[q.inputs[i]];
    for (std::size_t o = 0; o < q.outputs.size(); ++o) {
      if (q.producer[o] == i) {
        input.outputs.push_back(o); // output (vertex) order
      }
    }
    // A declared shape allows Run::alloc, whose slot the application holds while a push takes
    // another: the pool is sized for both (PORTS.md 5.4).
    input.feed_loans = input.input.shape.empty() ? 1U : 2U;
    // GStreamer stamps an empty stream_id with the input's public name, or "default" when it has
    // none (RunPush.cpp:225-264), on its graph path; on a plain linear Run its ingress meta stamps
    // "0" (InputStreamUtil.cpp:2703, 3720).
    input.default_stream =
        graph_path ? (input.names.empty() ? "default" : input.names.front()) : "0";
    // A blocking push into a fan-out waits for room on the links of the Outputs without
    // OutputOptions::drop and evicts on the others. The runtime's waiting push waits on every link
    // whatever its policy (Session::push_waiting), its policy push only on Block links
    // (Session::push): with both kinds of Output under a try_push policy other than Block, neither
    // does that.
    bool lossy = false;
    bool lossless = false;
    for (const std::size_t o : input.outputs) {
      (config.outputs[o].terminal_drop ? lossy : lossless) = true;
    }
    if (lossy && lossless && config.link_policy != OverflowPolicy::Block) {
      *why = "Input " + join(input.names) +
             " feeds Outputs with and without OutputOptions::drop under OverflowPolicy::" +
             std::string(policy_text(config.link_policy)) +
             ": a blocking push would wait for room on some of its links and evict on the "
             "others, which the runtime's push cannot do yet (it waits on every link, or on "
             "the Block links only)";
      return config;
    }
  }
  config.label = view.name;
  return config;
}

// The [NEAT] line of a Run that takes Dataflow: every input and the links of its Outputs.
std::string describe_choice(const DataflowRunConfig& config,
                            const std::vector<std::optional<DataflowInputBinding>>& bindings) {
  std::string out;
  for (std::size_t i = 0; i < config.inputs.size(); ++i) {
    const DataflowInputConfig& input = config.inputs[i];
    out += (i == 0 ? "" : "; ") + std::string("input ") + join(input.names) +
           (bindings[i] ? " " + bindings[i]->text : " bound by its first push") + " -> ";
    for (std::size_t k = 0; k < input.outputs.size(); ++k) {
      const DataflowOutputConfig& output = config.outputs[input.outputs[k]];
      out += (k == 0 ? "" : ", ") + std::string("output ") + join(output.names) + " link depth " +
             std::to_string(output.link_depth) + " (queue_depth " +
             std::to_string(output.ingress_depth) + " + max_buffers " +
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
  if (!seed.tensors.empty() && q.inputs.size() != 1U) {
    // GStreamer's own rule (connected_default_input_segment_or_throw, GraphBuildInput.cpp:193).
    return decline(want, where, error_codes::kInputShape,
                   "a seed for a graph of " + std::to_string(q.inputs.size()) +
                       " public Inputs: graph has no unambiguous default input; build without a "
                       "seed");
  }
  std::string why;
  DataflowRunConfig config = derive_config(graph, view, q, opt, &why);
  if (!why.empty()) {
    return decline(want, where, error_codes::kPipelineShape, why);
  }
  // Each input's spec, when a seed or the declaration gives it now; an unsupported declaration or
  // seed is decided here, before a Run exists.
  std::vector<std::optional<DataflowInputBinding>> bindings;
  for (const DataflowInputConfig& input : config.inputs) {
    bindings.push_back(
        seed.tensors.size() > 1U
            ? bind_dataflow_record(input.input, seed.tensors, &why)
            : bind_dataflow_input(input.input,
                                  seed.tensors.empty() ? nullptr : &seed.tensors.front(), &why));
    if (!why.empty()) {
      return decline(want, where, error_codes::kInputShape, why);
    }
  }
  warn_ignored_options(config);
  trace_choice(where, "dataflow", describe_choice(config, bindings));

  auto core = std::make_shared<RunCore>();
  initialize_run_identity(*core);
  core->opt = config.run;
  core->mode = RunMode::Async;
  core->dataflow_ = make_dataflow_run_state(*core, std::move(config), seed.tensors);
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
    s.tensors = seed;
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
    InputOptions input =
        q.reason.empty() && q.inputs.size() == 1U ? *view.inputs[q.inputs.front()] : InputOptions{};
    input.memory_policy = InputMemoryPolicy::SystemMemory;
    s.tensors.push_back(tensor_from_cv_mat(seed.front(), input, "Graph::build(inputs)"));
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
  // A record's description is one TensorSpec per field: up to kMaxFields of them.
  std::vector<char> buffer(spec.is_record() ? df::kDescribeMax * (df::kMaxFields + 1U)
                                            : df::kDescribeMax);
  df::Writer writer{std::span<char>(buffer)};
  df::describe(spec, writer);
  return std::string(writer.view());
}

// InputOptions::shape (Graph::input(name, shape, dtype)): the spec is the declaration, whatever
// the seed or the first push (validate_public_input checks a push against it). A dynamic axis is
// Dim{1, max_shape[i]}; without a bound it cannot be sized, so the Run declines.
std::optional<DataflowInputBinding> bind_declared_shape(const InputOptions& declared,
                                                        std::string* why) {
  const auto dtype = dataflow_dtype(declared.dtype);
  if (!dtype) {
    *why = "the declared dtype Float64 has no Dataflow dtype";
    return std::nullopt;
  }
  if (declared.shape.size() > df::kMaxRank) {
    *why = "a declared shape of rank " + std::to_string(declared.shape.size()) +
           " is not supported (1 to 6)";
    return std::nullopt;
  }
  df::TensorSpec spec;
  spec.dtype = *dtype;
  spec.layout = df::Dense{};
  spec.shape.rank = static_cast<std::uint8_t>(declared.shape.size());
  for (std::size_t i = 0; i < declared.shape.size(); ++i) {
    const std::int64_t size = declared.shape[i];
    const std::int64_t bound = declared.max_shape.empty() ? -1 : declared.max_shape[i];
    if (size > 0) {
      spec.shape.dims[i] = df::Dim(size);
    } else if (bound > 0) {
      spec.shape.dims[i] = df::Dim(1, bound);
    } else {
      *why = "axis " + std::to_string(i) +
             " of the declared shape is dynamic without a bound; the Dataflow backend sizes its "
             "input pool at build, so declare one (Graph::input's max_shape)";
      return std::nullopt;
    }
  }
  DataflowInputBinding binding;
  binding.dtype = declared.dtype;
  binding.rank = declared.shape.size();
  binding.spec.payload = spec;
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("the declared shape is not valid: ") + status->message;
    return std::nullopt;
  }
  binding.text = describe_spec(binding.spec);
  return binding;
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
  if (!declared.shape.empty()) {
    return bind_declared_shape(declared, why);
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
  if (binding.is_record()) {
    return "one tensor where a record of " + std::to_string(binding.fields.size()) +
           " is bound; the input is bound to " + binding.text +
           " (Dataflow backend: by its seed or its first push)";
  }
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

std::optional<DataflowInputBinding> bind_dataflow_record(const InputOptions& declared,
                                                         std::span<const Tensor> fields,
                                                         std::string* why) {
  why->clear();
  if (!declared.shape.empty()) {
    *why = "an item of " + std::to_string(fields.size()) +
           " tensors where one tensor of a declared shape is expected";
    return std::nullopt;
  }
  if (fields.size() > df::kMaxFields) {
    *why = "a record of " + std::to_string(fields.size()) + " tensors is not supported (at most " +
           std::to_string(df::kMaxFields) + ")";
    return std::nullopt;
  }
  // Each field binds as an undeclared tensor of its own: the InputOptions describe one tensor (an
  // image's format and size), so only their memory and pool settings apply to a field.
  InputOptions field_declared;
  field_declared.memory_policy = declared.memory_policy;
  field_declared.use_simaai_pool = declared.use_simaai_pool;
  field_declared.pool_max_buffers = declared.pool_max_buffers;
  DataflowInputBinding binding;
  df::RecordSchema schema;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    std::optional<DataflowInputBinding> field =
        bind_dataflow_input(field_declared, &fields[i], why);
    if (!field) {
      *why = "field " + std::to_string(i) +
             " of the record: " + (why->empty() ? std::string("cannot be bound") : *why);
      return std::nullopt;
    }
    if (field->planes != DataflowInputBinding::Planes::None) {
      *why = "field " + std::to_string(i) +
             " of the record is an NV12 or I420 image; planar record fields are not supported "
             "before H1";
      return std::nullopt;
    }
    // Dense fields, fixed by the tensor: a field's own declaration does not exist before H1.
    df::TensorSpec spec = std::get<df::TensorSpec>(field->spec.payload);
    for (std::size_t d = 0; d < spec.shape.rank; ++d) {
      spec.shape.dims[d] = df::Dim(fields[i].shape[d]);
    }
    schema.fields.push_back(
        df::Field{df::FieldId{static_cast<std::uint16_t>(i)}, std::to_string(i), spec});
    DataflowFieldBinding meta;
    meta.dtype = field->dtype;
    meta.layout = field->layout;
    meta.axis_semantics = std::move(field->axis_semantics);
    meta.semantic = std::move(field->semantic);
    meta.rank = field->rank;
    binding.fields.push_back(std::move(meta));
  }
  binding.spec.payload = std::move(schema);
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("the record's spec is not valid: ") + status->message;
    return std::nullopt;
  }
  const auto& bound = std::get<df::RecordSchema>(binding.spec.payload);
  for (std::size_t i = 0; i < binding.fields.size(); ++i) {
    binding.fields[i].offset = df::field_offset(bound, i);
    binding.fields[i].extent_offset = df::field_extent_offset(bound, i);
  }
  binding.text = describe_spec(binding.spec);
  return binding;
}

std::string dataflow_record_mismatch(const DataflowInputBinding& binding,
                                     std::span<const Tensor> fields) {
  const auto mismatch = [&](const std::string& what) {
    return what + "; the input is bound to the record " + binding.text +
           " (Dataflow backend: by its seed or its first push)";
  };
  if (!binding.is_record()) {
    return "an item of " + std::to_string(fields.size()) +
           " tensors where one tensor is bound; the input is bound to " + binding.text +
           " (Dataflow backend: by its declaration, its seed or its first push)";
  }
  if (fields.size() != binding.fields.size()) {
    return mismatch("an item of " + std::to_string(fields.size()) + " tensors");
  }
  const auto& schema = std::get<df::RecordSchema>(binding.spec.payload);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const Tensor& tensor = fields[i];
    const df::TensorSpec& spec = schema.fields[i].spec;
    const std::string field = "field " + std::to_string(i) + ": ";
    if (tensor.dtype != binding.fields[i].dtype) {
      return mismatch(field + "dtype differs");
    }
    if (tensor.is_composite() || tensor.has_tiled_storage()) {
      return mismatch(field + "a composite or tiled tensor where a dense one is bound");
    }
    if (tensor.shape.size() != spec.shape.rank) {
      return mismatch(field + "rank " + std::to_string(tensor.shape.size()) + " differs");
    }
    for (std::size_t d = 0; d < spec.shape.rank; ++d) {
      if (!spec.shape.dims[d].contains(tensor.shape[d])) {
        return mismatch(field + "dimension " + std::to_string(d) + " = " +
                        std::to_string(tensor.shape[d]) + " is out of bounds");
      }
    }
  }
  return {};
}

DataflowNet compile_dataflow_net(const DataflowRunConfig& config, std::size_t input,
                                 const DataflowInputBinding& binding, std::uint32_t stream) {
  const DataflowInputConfig& in = config.inputs[input];
  df::Net net;
  const df::OperatorRef<df::Feed> feed = net.add<df::Feed>(
      "input", binding.spec,
      df::Feed::Options{.stream = df::StreamId{stream}, .loans = in.feed_loans, .max_links = 0});
  std::vector<std::uint32_t> fetches;
  for (std::size_t k = 0; k < in.outputs.size(); ++k) {
    const DataflowOutputConfig& output = config.outputs[in.outputs[k]];
    if (output.link_depth == 0U || output.link_depth > df::kMaxLinkDepth) {
      pipeline_internal::error_util::throw_session_error(
          error_codes::kOptionOutOfRange,
          "Graph::build: the link depth " + std::to_string(output.link_depth) +
              " (RunOptions::queue_depth " + std::to_string(output.ingress_depth) +
              " + OutputOptions::max_buffers " + std::to_string(output.terminal_depth) +
              ") is outside the Dataflow link's range 1.." + std::to_string(df::kMaxLinkDepth));
    }
    df::Overflow overflow = df::Overflow::Block;
    if (output.terminal_drop || config.link_policy == OverflowPolicy::KeepLatest) {
      overflow = df::Overflow::DropOldest;
    } else if (config.link_policy == OverflowPolicy::DropIncoming) {
      overflow = df::Overflow::DropIncoming;
    }
    // Operator names are the Plan's own (Plan::fetch(name)); Core addresses its Fetches by index.
    const df::OperatorRef<df::Fetch> fetch =
        net.add<df::Fetch>(k == 0U ? std::string("output") : "output" + std::to_string(k),
                           binding.spec, df::Fetch::Options{.retained = 1});
    if (const df::Status status = net.connect(
            feed.out(), fetch.in(),
            df::LinkPolicy{.depth = output.link_depth, .overflow = overflow, .stale_after = {}})) {
      pipeline_internal::throw_dataflow_error(*status, "Graph::build");
    }
    fetches.push_back(fetch.index);
  }
  df::PlanReport failure;
  df::Result<df::Plan> plan = std::move(net).compile(df::CompileOptions{
      .force_fine_clock = false, .retention_limit_bytes = 0, .failure_report = &failure});
  if (const df::Error* error = df::failure(plan)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build", failure.mismatch);
  }
  return DataflowNet{std::get<df::Plan>(std::move(plan)), feed.index, std::move(fetches)};
}

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace simaai::neat::runtime

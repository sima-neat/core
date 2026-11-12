// The Dataflow backend's graph compiler (H0 plan 3.4, 3.6; H0 step 10, H0c): whether a graph
// qualifies for Dataflow and, when it does, its Run configuration. A qualifying graph has public
// Inputs and Outputs only, each Output fed by exactly one Input, after the public boundaries
// inside it are elided as compile_public_graph elides them. Each Input becomes one Feed with one
// link per Output it feeds (a fan-out when there are several) and one Fetch per Output, on a
// Session of its own (DataflowInputPort.cpp), in one Run state (DataflowRun.cpp). Everything else
// (fan-in and combine, operators, device memory) is declined (DataflowBackendSelect.cpp). Here:
// qualification, endpoint names, and the mapping of RunOptions and OutputOptions to link depths
// and policies.

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/PublicInputContract.h"
#include "pipeline/runtime/RunCore.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/internal/DataflowRuntime.h"
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::runtime {

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

namespace df = pipeline_internal::df;

std::string node_text(const Node& node) {
  const std::string label = node.user_label();
  return label.empty() ? node.kind() : node.kind() + " '" + label + "'";
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
    dataflow_warn_once(
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
      dataflow_warn_once("clamp:max_buffers",
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
      *why = "Input " + dataflow_names_text(input.names) +
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

} // namespace

// The options Dataflow does not read (plan 3.4), when set away from their defaults: one warning
// per process and option.
void warn_ignored_dataflow_options(const DataflowRunConfig& config) {
  const InputOptions defaults{};
  const auto ignored = [](const char* option) {
    dataflow_warn_once(std::string("ignored:") + option,
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

InputOptions dataflow_seed_input_options(const Graph& graph) {
  const DataflowGraphView view = DataflowGraphAccess::view(graph);
  const Qualification q = qualify(view);
  return q.reason.empty() && q.inputs.size() == 1U ? *view.inputs[q.inputs.front()]
                                                   : InputOptions{};
}

std::optional<DataflowRunConfig> plan_dataflow_run(const Graph& graph, const RunOptions& opt,
                                                   const DataflowSeed& seed, DataflowDecline* why) {
  const auto decline = [why](const char* code, std::string reason) {
    *why = DataflowDecline{code, std::move(reason)};
    return std::nullopt;
  };
  const DataflowGraphView view = DataflowGraphAccess::view(graph);
  const Qualification q = qualify(view);
  if (!q.reason.empty()) {
    return decline(error_codes::kPipelineShape, q.reason);
  }
  if (!seed.unsupported.empty()) {
    return decline(error_codes::kInputShape, seed.unsupported);
  }
  if (!seed.tensors.empty() && q.inputs.size() != 1U) {
    // GStreamer's own rule (connected_default_input_segment_or_throw, GraphBuildInput.cpp:193).
    return decline(error_codes::kInputShape,
                   "a seed for a graph of " + std::to_string(q.inputs.size()) +
                       " public Inputs: graph has no unambiguous default input; build without a "
                       "seed");
  }
  std::string reason;
  DataflowRunConfig config = derive_config(graph, view, q, opt, &reason);
  if (!reason.empty()) {
    return decline(error_codes::kPipelineShape, reason);
  }
  return config;
}

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace simaai::neat::runtime

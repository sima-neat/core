// The Dataflow backend's graph compiler: whether a graph qualifies for Dataflow and, when it does,
// its Run configuration. A qualifying graph has public Inputs and Outputs only, each Output fed by
// exactly one Input, after the public boundaries inside it are elided as compile_public_graph
// elides them; or it is a model route (one Input, the stage nodes of a Model in a chain, one
// Output), which becomes Feed -> one operator per stage -> Fetch when the route planner carries
// its stages (DataflowModelRoute.h). Each Input becomes one Feed with one link per Output it feeds
// (a fan-out when there are several) and one Fetch per Output, on a Session of its own
// (DataflowInputPort.cpp), in one Run state (DataflowRun.cpp). Everything else (fan-in and
// combine, other nodes) is not carried: the compiler says who and what (DataflowNotCarried), and
// the build throws it (DataflowBackendSelect.cpp). Here: qualification, endpoint
// names, the mapping of RunOptions and OutputOptions to link depths and policies, and each
// Input's memory (InputOptions::memory_policy resolved as GStreamer's build resolves it).

#include "pipeline/runtime/DataflowGraphCompiler.h"

#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/CredentialRedaction.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/ModelRouteOrigin.h"
#include "pipeline/internal/PublicInputContract.h"
#include "pipeline/runtime/RunCore.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/runtime/DataflowModelRoute.h"
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
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
  // A Model's own build of the Graph (ModelRouteScope): every vertex between its Input and its
  // Output is a stage of that Model.
  std::shared_ptr<const pipeline_internal::ModelRouteOrigin> origin;
  // The Graph's options, as its build reads them, except the element-name prefix and suffix: the
  // ones the Graph's creator set (empty when unset; the constructor's numbered suffix names
  // GStreamer's elements only).
  GraphOptions options;
  bool linear = true;
  std::string name;
  std::vector<std::shared_ptr<Node>> vertices;
  std::vector<std::string> runtime_stages;         // kinds of runtime-stage vertices
  std::vector<std::optional<InputOptions>> inputs; // per vertex: a public Input's resolved options
  std::vector<bool> model_stage; // per vertex: a node of a Model's fragment (its provenance)
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
  out.options = composition.options;
  out.options.element_name_prefix = graph.composition_->creator_names.prefix;
  out.options.element_name_suffix = graph.composition_->creator_names.suffix;
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
      // The options compile_public_graph gives this Input.
      out.inputs[vertex] = pipeline_internal::public_input_options(composition, vertex);
    }
  }
  out.model_stage.assign(composition.vertices.size(), false);
  out.origin = pipeline_internal::ModelRouteScope::of(graph);
  if (out.origin) {
    for (std::size_t vertex = 0; vertex < composition.vertices.size(); ++vertex) {
      const Node* node = composition.vertices[vertex].get();
      out.model_stage[vertex] =
          node && !dynamic_cast<const Input*>(node) && !dynamic_cast<const Output*>(node);
    }
  }
  for (const auto& fragment : composition.fragments) {
    // Model::preprocess(), inference(), postprocess() and graph() attach the model's identity to
    // the vertices they add (ModelAccess::build_stage_graph_fragment).
    if (fragment.provenance.model_id.empty()) {
      continue;
    }
    for (std::size_t vertex = fragment.graph_start;
         vertex < fragment.graph_end && vertex < out.model_stage.size(); ++vertex) {
      out.model_stage[vertex] = true;
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

const VerboseOptions& DataflowGraphAccess::verbose(const Graph& graph) {
  return graph.opt_.verbose;
}

bool DataflowGraphAccess::connected(const Graph& graph) {
  return graph.composition_ && !graph.composition_->is_linear();
}

std::shared_ptr<const pipeline_internal::ModelRouteOrigin>
DataflowGraphAccess::model_origin(const Graph& graph) {
  return pipeline_internal::ModelRouteScope::of(graph);
}

#if defined(SIMANEAT_WITH_DATAFLOW)

namespace {

namespace df = pipeline_internal::df;
using pipeline_internal::DataflowNotCarried;

// A node as a refusal names it: its kind and its label. A source's label is its URL: the
// credentials in it never reach a refusal (`user:password@` becomes `<redacted>@`, as
// GStreamer's errors print a source).
std::string node_text(const Node& node) {
  const std::string label = pipeline_internal::redact_gst_credentials(node.user_label());
  return label.empty() ? node.kind() : node.kind() + " '" + label + "'";
}

// The graph as a subject, for a fact that is no one node's.
constexpr const char* kThisGraph = "this graph";

// ---- qualification -----------------------------------------------------------------------------
// The public endpoints of a qualifying graph, and who feeds whom.
struct Qualification {
  DataflowNotCarried declined;       // empty: qualifies
  std::vector<std::size_t> inputs;   // public Input vertices, vertex order
  std::vector<std::size_t> outputs;  // public Output vertices, vertex order
  std::vector<std::size_t> producer; // per output (index into outputs): its input (into inputs)
  std::vector<std::size_t> fan_out;  // per input: how many outputs it feeds
  // A model route: the model's stage vertices in chain order, between the one Input and the
  // one Output; empty otherwise.
  std::vector<std::size_t> stages;
};

// The public endpoints of the graph and who feeds whom, after compile_public_graph's boundary
// elision (normalize_public_boundaries_for_execution): an
// Input with an explicit upstream connection and an Output with an explicit downstream connection
// are internal boundary declarations (a Branch's, a connected fragment's), not endpoints, and an
// edge into one continues along its outgoing edges. What is left must be public Inputs feeding
// public Outputs, each Output fed by exactly one Input: one Feed per Input, one link and one
// Fetch per Output it feeds. One other shape qualifies, in a build that has the MLA operator:
// one Input feeding a chain of a Model's stage nodes (Model::inference() alone, or
// Model::preprocess(), inference(), postprocess()) feeding one Output. Whether those nodes are
// stages this backend runs is the planner's question (derive_config).
Qualification qualify(const DataflowGraphView& view) {
  static constexpr const char* kRuns =
      "the backend runs Inputs, Outputs and the stages of a Model between them";
  Qualification q;
  const auto decline = [&q](std::string subject, std::string fact) {
    q.declined.subject = std::move(subject);
    q.declined.fact = std::move(fact);
  };
  if (view.model_route) {
    if (!view.origin) {
      decline(kThisGraph, "its Model processes its own input (Model::graph() with an input route)");
    } else if (view.origin->inputs > 1U) {
      decline("model " + view.origin->model,
              "it reads " + std::to_string(view.origin->inputs) + " inputs");
    } else {
      decline("model " + view.origin->model,
              "its input tensor goes to the MLA as it is pushed, with no stage in front of the "
              "MLA");
    }
    return q;
  }
  if (!view.runtime_stages.empty()) {
    decline("runtime stage " + view.runtime_stages.front(),
            std::string("it is a runtime stage: ") + kRuns);
    return q;
  }
  const std::size_t n = view.vertices.size();
  enum class Kind { None, In, Out, Fragment };
  std::vector<Kind> kind(n, Kind::None);
  std::vector<std::size_t> fragments;
  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    const auto& node = view.vertices[vertex];
    if (!node) {
      continue;
    }
    if (dynamic_cast<const Input*>(node.get())) {
      kind[vertex] = Kind::In;
    } else if (const auto* output = dynamic_cast<const Output*>(node.get())) {
      if (output->options().combine_policy != CombinePolicy::None) {
        decline("node " + node_text(*node), "it has a CombinePolicy (fan-in)");
        return q;
      }
      kind[vertex] = Kind::Out;
    } else if (dataflow_has_model_route() &&
               (node->kind() == "ModelFragment" || view.model_stage[vertex])) {
      kind[vertex] = Kind::Fragment;
      fragments.push_back(vertex);
    } else if (node->kind() == "SimaBoxDecode") {
      decline("node " + node_text(*node),
              "it is a standalone BoxDecode node: BoxDecode runs as the model's own stage, "
              "Model::postprocess() behind Model::inference()");
      return q;
    } else {
      decline("node " + node_text(*node), kRuns);
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
      decline(kThisGraph, "a composition edge references an invalid vertex");
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
    decline(kThisGraph, "it has no public Input (a source graph)");
    return q;
  }
  if (q.outputs.empty()) {
    decline(kThisGraph, "it has no public Output");
    return q;
  }

  // Every edge leaving a vertex that is not elided, followed through the elided boundaries it
  // enters, to the endpoint it reaches.
  std::vector<std::pair<std::size_t, std::size_t>> links;
  std::vector<bool> visiting(n, false);
  std::function<bool(std::size_t, std::size_t)> follow = [&](std::size_t start, std::size_t e) {
    const auto& edge = view.edges[e];
    if (edge.policy != GraphLinkPolicy::Default) {
      decline(kThisGraph, "a link of it has a policy other than GraphLinkPolicy::Default");
      return false;
    }
    if (!edge.stream_id.empty()) {
      decline(kThisGraph, "a link of it has a stream_id ('" + edge.stream_id + "')");
      return false;
    }
    const std::size_t to = edge.to;
    if (!elide[to]) {
      links.emplace_back(start, to);
      return true;
    }
    if (visiting[to]) {
      decline(kThisGraph, "its public boundary declarations form a cycle");
      return false;
    }
    if (outgoing[to].empty()) {
      decline("node " + text(to), "it is an internal boundary that leads to no endpoint");
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
  if (!fragments.empty()) {
    // The model route: exactly Input -> stage node -> ... -> stage node -> Output, every stage
    // node once.
    bool chain =
        q.inputs.size() == 1U && q.outputs.size() == 1U && links.size() == fragments.size() + 1U;
    std::size_t at = chain ? q.inputs[0] : 0U;
    while (chain && q.stages.size() < fragments.size()) {
      std::size_t next = kNone;
      std::size_t found = 0;
      for (const auto& [from, to] : links) {
        if (from == at) {
          next = to;
          ++found;
        }
      }
      chain = found == 1U && kind[next] == Kind::Fragment &&
              std::find(q.stages.begin(), q.stages.end(), next) == q.stages.end();
      if (chain) {
        q.stages.push_back(next);
        at = next;
      }
    }
    chain = chain && std::count(links.begin(), links.end(), std::pair{at, q.outputs[0]}) == 1;
    if (!chain) {
      q.stages.clear();
      decline("node " + text(fragments.front()),
              "it is a stage of a Model outside one chain between one Input and one Output");
      return q;
    }
    q.producer[0] = 0;
    q.fan_out.assign(1, 1);
    return q;
  }
  for (const auto& [from, to] : links) {
    if (kind[from] != Kind::In || kind[to] != Kind::Out) {
      decline("the connection from " + text(from) + " to " + text(to),
              "it is not an Input feeding an Output");
      return q;
    }
    std::size_t& producer = q.producer[output_of[to]];
    if (producer != kNone) {
      decline("node " + text(to), "it has more than one producer (fan-in)");
      return q;
    }
    producer = input_of[from];
  }
  q.fan_out.resize(q.inputs.size());
  for (std::size_t o = 0; o < q.outputs.size(); ++o) {
    if (q.producer[o] == kNone) {
      decline("node " + text(q.outputs[o]), "it is connected to no Input");
      return q;
    }
    ++q.fan_out[q.producer[o]];
  }
  for (std::size_t i = 0; i < q.inputs.size(); ++i) {
    if (q.fan_out[i] == 0U) {
      decline("node " + text(q.inputs[i]), "it is connected to no Output");
      return q;
    }
    if (q.fan_out[i] > df::kMaxFanOut) {
      decline("node " + text(q.inputs[i]), "it feeds " + std::to_string(q.fan_out[i]) +
                                               " Outputs: an Input feeds at most " +
                                               std::to_string(df::kMaxFanOut));
      return q;
    }
  }
  return q;
}

// The public names of one endpoint vertex, as compile_public_graph names it: its explicit name;
// else, for each named fragment holding it, the fragment's name when it is the fragment's only
// public endpoint of its kind, or that name and its index among them ("classes_1",
// add_fragment_named_candidates); kept only when the Graph
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
// DataflowOutputConfig); otherwise compile_public_graph's error (add_named_endpoint).
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

// The Run's configuration; `why` set when this graph's options, or its model's route, are what
// Dataflow does not carry. Throws what GStreamer's build throws for endpoint names two endpoints
// share.
DataflowRunConfig derive_config(const Graph& graph, const DataflowGraphView& view,
                                const Qualification& q, const RunOptions& opt,
                                std::span<const Tensor> seed, DataflowNotCarried* why) {
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
  std::string shared = shared_name(config.inputs, "input");
  if (shared.empty()) {
    shared = shared_name(config.outputs, "output");
  }
  if (!shared.empty()) {
    throw std::runtime_error(shared);
  }
  config.input_names = sorted_union(config.inputs);
  config.output_names = sorted_union(config.outputs);
  // The path GStreamer takes for the same graph: a connected or named graph is a graph Run, a plain
  // linear Run one pipeline segment (RunCore::start, through is_simple_linear_plan).
  const bool graph_path =
      !view.linear || !config.input_names.empty() || !config.output_names.empty();
  config.graph_run = graph_path;

  // RunOptions resolved as GStreamer resolves them for either path, by the same code,
  // resolve_build_opt: a plain linear Run's (prepare_build_input_context, in
  // RunCore::start_pipeline_segment, kept by start_single_pipeline), and a graph Run's ingress
  // (graph_runtime_options_from_run_options: its queue the resolved queue_depth, its policy the
  // resolved overflow_policy). RunPreset fills queue_depth where it equals its RunOptions{}
  // default and overflow_policy where it is OverflowPolicy::Auto, so an explicit Block under
  // Realtime stays Block, and Realtime with Auto is KeepLatest at the preset's depth.
  config.run = session_build_resolve_build_opt(RunMode::Async, opt);
  config.link_policy = config.run.overflow_policy;

  // One link stands in for both of Core's queues on the way to each Output (RunOptions):
  // GStreamer's input queue (RunOptions::queue_depth) and the public Output's terminal queue
  // (OutputOptions::max_buffers). It is as deep as the deeper of the two:
  //   link depth = max(queue_depth, max_buffers)
  // not their sum: with a pusher faster than the puller the link stays full, so the latency and
  // the tensors the application must keep alive both scale with the depth, and the sum doubled
  // them whenever the two were set equal. A queue GStreamer leaves unbounded (queue_depth or
  // max_buffers <= 0) has no link to stand for it: not carried.
  // The Output's drop flag makes its link lossy for every push, as appsink drop=true is;
  // otherwise RunOptions' overflow policy applies to try_push (KeepLatest is DropOldest). Either
  // acts only when the link holds the whole depth. In a fan-out each Output's link has its own.
  const auto unbounded = [why](std::string option, int value) {
    why->subject = "option " + std::move(option) + " = " + std::to_string(value);
    why->fact = "it leaves a queue unbounded, and every link of this backend holds a bounded "
                "number of items";
    why->code = error_codes::kOptionOutOfRange;
  };
  if (config.run.queue_depth <= 0) {
    unbounded("RunOptions::queue_depth", config.run.queue_depth);
    return config;
  }
  const auto ingress_depth = static_cast<std::uint32_t>(config.run.queue_depth);
  for (std::size_t o = 0; o < q.outputs.size(); ++o) {
    DataflowOutputConfig& output = config.outputs[o];
    output.output = dynamic_cast<const Output*>(view.vertices[q.outputs[o]].get())->options();
    if (output.output.max_buffers <= 0) {
      unbounded("OutputOptions::max_buffers", output.output.max_buffers);
      return config;
    }
    if (output.output.sync) {
      why->subject = "option OutputOptions::sync";
      why->fact = "it paces the Output by GStreamer's pipeline clock, and this backend's Outputs "
                  "hand each item over when it is ready";
      why->code = error_codes::kOptionOutOfRange;
      return config;
    }
    output.ingress_depth = ingress_depth;
    output.terminal_depth = static_cast<std::uint32_t>(output.output.max_buffers);
    output.link_depth = std::max(output.ingress_depth, output.terminal_depth);
    output.terminal_drop = output.output.drop;
    output.input = q.producer[o];
  }
  for (std::size_t i = 0; i < q.inputs.size(); ++i) {
    DataflowInputConfig& input = config.inputs[i];
    input.input = *view.inputs[q.inputs[i]];
    // A model route's Input carries the Model's own options (Model::input_appsrc_options), not
    // the caller's.
    if (q.stages.empty() && !input.input.caps_override.empty()) {
      // Caps that do not parse are the application's mistake, refused as GStreamer refuses it.
      gst_init_once();
      session_build_validate_caps_override(input.input, "Graph::build(input)");
      why->subject = "option InputOptions::caps_override of Input " +
                     dataflow_names_text(config.inputs[i].names);
      why->fact = "it is a GStreamer caps string, and this backend's inputs take the spec of what "
                  "is pushed";
      why->code = error_codes::kOptionOutOfRange;
      return config;
    }
    if (view.origin && input.input.payload_type == PayloadType::Image) {
      // A Model's own build gives its Input the size of the seed's frame (GStreamer's first caps)
      // and the model's capacity as the bound: each frame brings its own size up to the bound,
      // as on GStreamer.
      input.input.width = 0;
      input.input.height = 0;
    }
    // The Input and what it feeds, as the nodes of GStreamer's segment for it.
    std::vector<std::shared_ptr<Node>> route{view.vertices[q.inputs[i]]};
    for (const std::size_t stage : q.stages) {
      route.push_back(view.vertices[stage]);
    }
    for (std::size_t o = 0; o < q.outputs.size(); ++o) {
      if (q.producer[o] == i) {
        input.outputs.push_back(o); // output (vertex) order
        route.push_back(view.vertices[q.outputs[o]]);
      }
    }
    // Where the input's pool lives, by the function GStreamer's build asks
    // (GraphBuildInput.cpp, ExecutionGraphCompiler.cpp resolve_pipeline_input_memory): an explicit
    // memory_policy as given (Ev74, Dms0: a device pool); use_simaai_pool = false and Auto in front
    // of an Output: system memory, a host pool. Kept in the input's options, as GStreamer keeps it
    // in its appsrc options: from here on memory_policy is never Auto.
    input.input.memory_policy =
        pipeline_internal::resolve_input_memory(input.input, route).allocation;
    if (!q.stages.empty()) {
      // The model's stages, from the contracts GStreamer's build compiles and lowers for the
      // same nodes under the same Graph options. The input's pool is where GStreamer puts it:
      // Dms0 in front of a model fragment (Auto), or as declared.
      input.model = plan_dataflow_model_route(route, view.options, input.input, seed, why);
      if (!input.model) {
        return config;
      }
      if (input.model->ev74_input) {
        // In front of a kernel on the EV74 the pool is linux,cma whatever was declared: the EV74
        // reaches no other memory. A pushed tensor of the DMS heap is copied into a slot.
        input.input.memory_policy = InputMemoryPolicy::Ev74;
      }
      if (config.outputs[input.outputs.front()].terminal_drop) {
        why->subject = "option OutputOptions::drop behind a model";
        why->fact = "it makes the route's last link drop frames, which Core does not count";
        why->code = error_codes::kOptionOutOfRange;
        return config;
      }
    }
    // A declared shape allows Run::alloc, whose slot the application holds while a push takes
    // another: the pool is sized for both.
    input.feed_loans = input.input.shape.empty() ? 1U : 2U;
    // GStreamer stamps an empty stream_id with the input's public name, or "default" when it has
    // none, on its graph path; on a plain linear Run its ingress meta stamps "0".
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
      why->subject = "Input " + dataflow_names_text(input.names);
      why->fact = "it feeds Outputs with and without OutputOptions::drop under OverflowPolicy::" +
                  std::string(policy_text(config.link_policy)) +
                  ": a blocking push would wait for room on some of its links and evict on the "
                  "others";
      why->code = error_codes::kOptionOutOfRange;
      return config;
    }
  }
  config.label = view.name;
  return config;
}

} // namespace

InputOptions dataflow_seed_input_options(const Graph& graph) {
  const DataflowGraphView view = DataflowGraphAccess::view(graph);
  const Qualification q = qualify(view);
  return !q.declined && q.inputs.size() == 1U ? *view.inputs[q.inputs.front()] : InputOptions{};
}

std::optional<DataflowRunConfig> plan_dataflow_run(const Graph& graph, const RunOptions& opt,
                                                   const DataflowSeed& seed, const char* where,
                                                   DataflowNotCarried* why) {
  const DataflowGraphView view = DataflowGraphAccess::view(graph);
  if (view.linear && (!seed.tensors.empty() || !seed.unsupported.empty())) {
    // A seeded build of a linear graph without an Input, or with a source, is the application's
    // mistake: GStreamer's build refuses it first, and so does this one, with its text.
    enforce_push_run_mode(view.vertices, "Graph::build(input)");
  }
  Qualification q = qualify(view);
  if (q.declined) {
    *why = std::move(q.declined);
    return std::nullopt;
  }
  if (!seed.unsupported.empty()) {
    *why = DataflowNotCarried{error_codes::kInputShape, "the seed", seed.unsupported};
    return std::nullopt;
  }
  if (!seed.tensors.empty() && q.inputs.size() != 1U) {
    // GStreamer's own rule and text (connected_default_input_segment_or_throw).
    throw std::runtime_error(std::string(where ? where : "Graph::build(inputs)") +
                             ": graph has no unambiguous default input; build without a seed or "
                             "make the connected graph topology unambiguous");
  }
  DataflowNotCarried declined;
  DataflowRunConfig config = derive_config(graph, view, q, opt, seed.tensors, &declined);
  if (declined) {
    *why = std::move(declined);
    return std::nullopt;
  }
  return config;
}

#endif // SIMANEAT_WITH_DATAFLOW

} // namespace simaai::neat::runtime

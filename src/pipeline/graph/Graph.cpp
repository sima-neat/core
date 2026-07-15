// src/pipeline/Graph.cpp

#include "pipeline/Graph.h"
#include "GraphDetail.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/UxLogging.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

namespace {
std::atomic<std::uint64_t> g_pipeline_instance{0};
std::atomic<std::uint64_t> g_graph_instance{0};

std::uint64_t next_graph_id() {
  return g_graph_instance.fetch_add(1, std::memory_order_relaxed) + 1U;
}

std::string trim_endpoint_name(std::string_view name) {
  std::size_t first = 0;
  while (first < name.size() && std::isspace(static_cast<unsigned char>(name[first]))) {
    ++first;
  }
  std::size_t last = name.size();
  while (last > first && std::isspace(static_cast<unsigned char>(name[last - 1U]))) {
    --last;
  }
  return std::string(name.substr(first, last - first));
}

std::string endpoint_token(std::string_view text, std::string_view fallback) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc)) {
      out.push_back(static_cast<char>(std::tolower(uc)));
    } else if (c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else if (!out.empty() && out.back() != '_') {
      out.push_back('_');
    }
  }
  while (!out.empty() && out.front() == '_') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  if (out.empty()) {
    out = std::string(fallback);
  }
  if (!out.empty() && std::isdigit(static_cast<unsigned char>(out.front()))) {
    out.insert(out.begin(), '_');
  }
  return out;
}

template <typename Nodes>
std::string infer_endpoint_name_from_nodes(const Nodes& nodes, std::string_view fallback) {
  for (const auto& node : nodes) {
    if (!node) {
      continue;
    }
    const std::string kind = node->kind();
    const std::string label = trim_endpoint_name(node->user_label());
    if (!label.empty() && !(kind == "Input" && label == "mysrc")) {
      return endpoint_token(label, fallback);
    }
  }
  if (nodes.size() == 1U && nodes.front()) {
    const std::string kind = nodes.front()->kind();
    if (kind == "Input") {
      return "input";
    }
    if (kind == "Output") {
      return "output";
    }
  }
  for (const auto& node : nodes) {
    if (node) {
      const std::string kind = trim_endpoint_name(node->kind());
      if (!kind.empty()) {
        return endpoint_token(kind, fallback);
      }
    }
  }
  return std::string(fallback);
}

const Input* as_input_endpoint(const std::shared_ptr<Node>& node) {
  return dynamic_cast<const Input*>(node.get());
}

const Output* as_output_endpoint(const std::shared_ptr<Node>& node) {
  return dynamic_cast<const Output*>(node.get());
}

bool is_boundary_endpoint_node(const std::shared_ptr<Node>& node) {
  return as_input_endpoint(node) != nullptr || as_output_endpoint(node) != nullptr;
}

std::string explicit_endpoint_name(const std::shared_ptr<Node>& node) {
  if (const auto* input = as_input_endpoint(node)) {
    return trim_endpoint_name(input->endpoint_name());
  }
  if (const auto* output = as_output_endpoint(node)) {
    return trim_endpoint_name(output->endpoint_name());
  }
  return {};
}

std::string join_strings(const std::vector<std::string>& values) {
  if (values.empty()) {
    return "<none>";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ", ";
    }
    oss << values[i];
  }
  return oss.str();
}

void append_unique_endpoint_name(std::vector<std::string>* out, std::string name) {
  if (!out) {
    return;
  }
  name = trim_endpoint_name(name);
  if (name.empty()) {
    return;
  }
  if (std::find(out->begin(), out->end(), name) != out->end()) {
    return;
  }
  out->push_back(std::move(name));
}

std::string basename_token_from_path(const std::string& path, std::string_view fallback) {
  std::size_t pos = path.find_last_of("/\\");
  std::string base = (pos == std::string::npos) ? path : path.substr(pos + 1U);
  for (const std::string& suffix :
       {std::string(".tar.gz"), std::string(".tgz"), std::string(".mpk"), std::string(".tar")}) {
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
      base.resize(base.size() - suffix.size());
      break;
    }
  }
  return endpoint_token(base, fallback);
}

std::string model_import_key(const Model& model, const Graph& route_fragment) {
  std::string key = internal::ModelAccess::model_id(model);
  if (key.empty()) {
    key = internal::ModelAccess::source_path(model);
  }
  if (key.empty()) {
    key = "unnamed";
  }

  (void)route_fragment;
  std::ostringstream oss;
  // Use the loaded Model instance as the primary identity. This guarantees
  // repeated `app.connect(input, model); app.connect(model, output);` calls
  // reuse the same imported route, while different Model objects from the same
  // archive remain distinct. That is deliberately conservative: it avoids the
  // old bug where two Models with the same archive path but different
  // construction options could be silently coalesced.
  oss << "model@" << static_cast<const void*>(&model) << ":" << key;
  return oss.str();
}

std::string model_endpoint_name(const Model& model) {
  const std::string path = internal::ModelAccess::source_path(model);
  return basename_token_from_path(path, "model");
}

template <typename Composition>
std::vector<std::string> available_endpoint_names(const Composition& composition, bool inputs) {
  std::vector<std::string> names;
  for (const auto& node : composition.vertices) {
    const bool role_matches =
        inputs ? as_input_endpoint(node) != nullptr : as_output_endpoint(node) != nullptr;
    if (!role_matches) {
      continue;
    }
    std::string name = explicit_endpoint_name(node);
    if (name.empty()) {
      name = node ? std::string("<unnamed ") + node->kind() + ">" : std::string("<null>");
    }
    names.push_back(std::move(name));
  }
  return names;
}

template <typename Composition>
std::size_t resolve_boundary_endpoint_or_throw(const Composition& composition,
                                               std::string_view raw_name, const char* direction,
                                               const char* context) {
  const std::string name = trim_endpoint_name(raw_name);
  if (name.empty()) {
    throw std::runtime_error(std::string(context ? context : "Graph::connect") +
                             ": endpoint name must not be empty");
  }

  std::vector<std::size_t> matches;
  for (std::size_t i = 0; i < composition.vertices.size(); ++i) {
    if (!is_boundary_endpoint_node(composition.vertices[i])) {
      continue;
    }
    if (explicit_endpoint_name(composition.vertices[i]) == name) {
      matches.push_back(i);
    }
  }

  if (matches.size() == 1U) {
    return matches.front();
  }

  std::ostringstream oss;
  oss << (context ? context : "Graph::connect") << ": ";
  if (matches.empty()) {
    oss << "missing " << direction << " endpoint '" << name << "'";
  } else {
    oss << "endpoint '" << name << "' is ambiguous for " << direction << " resolution";
  }
  oss << "; available inputs: " << join_strings(available_endpoint_names(composition, true))
      << "; available outputs: " << join_strings(available_endpoint_names(composition, false));
  throw std::runtime_error(oss.str());
}

template <typename Edge> bool edge_is_implicit_linear(const Edge& edge) {
  using Kind = std::decay_t<decltype(edge.kind)>;
  return edge.kind == Kind::ImplicitLinear;
}

template <typename Composition>
void promote_endpoint_mode_or_throw(Composition& composition, std::string_view raw_from,
                                    std::string_view raw_to, const char* context) {
  const std::string from = trim_endpoint_name(raw_from);
  const std::string to = trim_endpoint_name(raw_to);
  composition.endpoint_mode = true;

  auto name_involved = [&](const std::shared_ptr<Node>& node) {
    const std::string name = explicit_endpoint_name(node);
    return !name.empty() && (name == from || name == to);
  };

  using Edge = typename std::decay_t<decltype(composition.edges)>::value_type;
  std::vector<Edge> kept;
  kept.reserve(composition.edges.size());

  for (const auto& edge : composition.edges) {
    if (!edge_is_implicit_linear(edge)) {
      kept.push_back(edge);
      continue;
    }
    if (edge.from >= composition.vertices.size() || edge.to >= composition.vertices.size()) {
      throw std::runtime_error(
          std::string(context ? context : "Graph::connect") +
          ": implicit edge references invalid vertex during endpoint promotion");
    }
    const bool from_boundary = is_boundary_endpoint_node(composition.vertices[edge.from]);
    const bool to_boundary = is_boundary_endpoint_node(composition.vertices[edge.to]);
    const bool involved = name_involved(composition.vertices[edge.from]) ||
                          name_involved(composition.vertices[edge.to]);
    if (!involved) {
      kept.push_back(edge);
      continue;
    }
    if (from_boundary && to_boundary) {
      // Accidental declaration-chain edge created by add(); drop it.
      continue;
    }
    throw std::runtime_error(
        std::string(context ? context : "Graph::connect") +
        ": endpoint promotion would require pruning an implicit edge touching a non-boundary "
        "node; keep the linear path or split it into explicit Graph fragments");
  }

  composition.edges = std::move(kept);
  composition.recompute_unique_tail();
}

bool stop_trace_enabled() {
  return pipeline_internal::env_bool("SIMA_STOP_TRACE", false);
}
} // namespace

Graph::Graph(const GraphOptions& opt)
    : composition_(std::make_unique<CompositionGraph>()), opt_(opt) {
  // Fold the jargon-free advanced_execution surface into the legacy execution fields before
  // anything consumes them. No-op when unset (default), so existing behavior is unchanged.
  opt_.resolve_advanced_execution();
  graph_id_ = next_graph_id();
  verbose_guard_ = pipeline_internal::ux::acquire_runtime_verbosity(opt_.verbose);
  if (opt_.element_name_suffix.empty()) {
    const std::uint64_t id = g_pipeline_instance.fetch_add(1) + 1;
    opt_.element_name_suffix = "_" + std::to_string(id);
  }
}

Graph::Graph(std::string name, const GraphOptions& opt) : Graph(opt) {
  endpoint_name_ = trim_endpoint_name(name);
}

Graph::~Graph() noexcept {
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Graph::~Graph begin\n");
  }
  invalidate_built_();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] Graph::~Graph end\n");
  }
}

// Single source of truth for invalidating the built pipeline + cached runner.
// All resource teardown happens via RAII inside BuiltState; this function
// only releases ownership of the unique_ptrs. Safe to call when there is no
// built state. Never throws.
void Graph::invalidate_built_() noexcept {
  built_.reset();
  run_cache_.reset();
}

Graph::Graph(Graph&& other) noexcept {
  // `*this` is freshly default-constructed: built_ and run_cache_ are null, so
  // adopting other's pointers cannot leak any resource of our own.
  composition_ = std::move(other.composition_);
  groups_ = std::move(other.groups_);
  last_pipeline_ = std::move(other.last_pipeline_);
  guard_ = std::move(other.guard_);
  verbose_guard_ = std::move(other.verbose_guard_);
  opt_ = other.opt_;
  endpoint_name_ = std::move(other.endpoint_name_);
  tensor_cb_ = std::move(other.tensor_cb_);
  nodes_version_.store(other.nodes_version_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  built_ = std::move(other.built_);
  run_cache_ = std::move(other.run_cache_);
  built_version_ = other.built_version_;
  input_route_processor_ = std::move(other.input_route_processor_);
  graph_id_ = other.graph_id_;
  other.graph_id_ = next_graph_id();
}

Graph& Graph::operator=(Graph&& other) noexcept {
  if (this != &other) {
    // `built_ = std::move(other.built_);` correctly tears down our previous
    // pipeline (if any) via RAII inside BuiltState before adopting `other`'s
    // pointer — std::unique_ptr's move-assignment destroys the held object
    // first and the BuiltState destructor stops + unrefs the pipeline/sink.
    // Same for `run_cache_`.
    composition_ = std::move(other.composition_);
    groups_ = std::move(other.groups_);
    last_pipeline_ = std::move(other.last_pipeline_);
    guard_ = std::move(other.guard_);
    verbose_guard_ = std::move(other.verbose_guard_);
    opt_ = other.opt_;
    endpoint_name_ = std::move(other.endpoint_name_);
    tensor_cb_ = std::move(other.tensor_cb_);
    nodes_version_.store(other.nodes_version_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    built_ = std::move(other.built_);
    run_cache_ = std::move(other.run_cache_);
    built_version_ = other.built_version_;
    input_route_processor_ = std::move(other.input_route_processor_);
    graph_id_ = other.graph_id_;
    other.graph_id_ = next_graph_id();
  }
  return *this;
}

Graph& Graph::set_name(std::string name) {
  endpoint_name_ = trim_endpoint_name(name);
  mark_composition_changed();
  return *this;
}

const std::string& Graph::name() const noexcept {
  return endpoint_name_;
}

std::vector<std::string> Graph::inputs() const {
  std::vector<std::string> out;
  if (!composition_) {
    return out;
  }
  for (std::size_t vertex = 0; vertex < composition_->vertices.size(); ++vertex) {
    if (const auto* input = as_input_endpoint(composition_->vertices[vertex])) {
      const bool connected = std::any_of(
          composition_->edges.begin(), composition_->edges.end(), [&](const auto& edge) {
            return edge.kind == CompositionEdgeKind::PublicEndpoint && edge.to == vertex;
          });
      if (!connected) {
        append_unique_endpoint_name(&out, input->endpoint_name());
      }
    }
  }
  for (const auto& named : composition_->named_fragments) {
    if (named.name.empty() || named.end <= named.start ||
        named.end > composition_->vertices.size()) {
      continue;
    }
    bool all_inputs = true;
    std::vector<std::size_t> public_inputs;
    for (std::size_t vertex = named.start; vertex < named.end; ++vertex) {
      if (as_input_endpoint(composition_->vertices[vertex]) == nullptr) {
        all_inputs = false;
        break;
      }
      const bool connected = std::any_of(
          composition_->edges.begin(), composition_->edges.end(), [&](const auto& edge) {
            return edge.kind == CompositionEdgeKind::PublicEndpoint && edge.to == vertex;
          });
      if (!connected) {
        public_inputs.push_back(vertex);
      }
    }
    if (!all_inputs || public_inputs.empty()) {
      continue;
    }
    if (public_inputs.size() == 1U) {
      append_unique_endpoint_name(&out, named.name);
    } else {
      for (std::size_t i = 0; i < public_inputs.size(); ++i) {
        append_unique_endpoint_name(&out, named.name + "_" + std::to_string(i));
      }
    }
  }
  auto fragment_input_connected = [&](const runtime::FragmentPlan& fragment,
                                      const std::string& name) {
    return std::any_of(composition_->edges.begin(), composition_->edges.end(),
                       [&](const auto& edge) {
                         return edge.kind == CompositionEdgeKind::PublicEndpoint &&
                                edge.endpoint.has_value() && edge.to >= fragment.graph_start &&
                                edge.to < fragment.graph_end && edge.endpoint->to_endpoint == name;
                       });
  };
  for (const auto& fragment : composition_->fragments) {
    if (!fragment.boundary_hints.has_value()) {
      continue;
    }
    for (const auto& name : fragment.boundary_hints->ingress_endpoint_names) {
      if (!fragment_input_connected(fragment, name)) {
        append_unique_endpoint_name(&out, name);
      }
    }
  }
  return out;
}

std::vector<std::string> Graph::outputs() const {
  std::vector<std::string> out;
  if (!composition_) {
    return out;
  }
  auto has_downstream = [&](std::size_t vertex) {
    return std::any_of(
        composition_->edges.begin(), composition_->edges.end(), [&](const auto& edge) {
          return edge.kind == CompositionEdgeKind::PublicEndpoint && edge.from == vertex;
        });
  };
  for (std::size_t vertex = 0; vertex < composition_->vertices.size(); ++vertex) {
    if (const auto* output = as_output_endpoint(composition_->vertices[vertex])) {
      if (!has_downstream(vertex)) {
        append_unique_endpoint_name(&out, output->endpoint_name());
      }
    }
  }
  for (const auto& named : composition_->named_fragments) {
    if (named.name.empty() || named.end <= named.start ||
        named.end > composition_->vertices.size()) {
      continue;
    }
    bool all_outputs = true;
    std::vector<std::size_t> public_outputs;
    for (std::size_t vertex = named.start; vertex < named.end; ++vertex) {
      if (as_output_endpoint(composition_->vertices[vertex]) == nullptr) {
        all_outputs = false;
        break;
      }
      if (!has_downstream(vertex)) {
        public_outputs.push_back(vertex);
      }
    }
    if (!all_outputs || public_outputs.empty()) {
      continue;
    }
    if (public_outputs.size() == 1U) {
      append_unique_endpoint_name(&out, named.name);
    } else {
      for (std::size_t i = 0; i < public_outputs.size(); ++i) {
        append_unique_endpoint_name(&out, named.name + "_" + std::to_string(i));
      }
    }
  }
  auto fragment_output_connected = [&](const runtime::FragmentPlan& fragment,
                                       const std::string& name) {
    return std::any_of(
        composition_->edges.begin(), composition_->edges.end(), [&](const auto& edge) {
          return edge.kind == CompositionEdgeKind::PublicEndpoint && edge.endpoint.has_value() &&
                 edge.from >= fragment.graph_start && edge.from < fragment.graph_end &&
                 edge.endpoint->from_endpoint == name;
        });
  };
  for (const auto& fragment : composition_->fragments) {
    if (!fragment.boundary_hints.has_value()) {
      continue;
    }
    for (const auto& name : fragment.boundary_hints->egress_endpoint_names) {
      if (!fragment_output_connected(fragment, name)) {
        append_unique_endpoint_name(&out, name);
      }
    }
  }
  return out;
}

void Graph::mark_composition_changed() {
  nodes_version_.fetch_add(1, std::memory_order_relaxed);
  invalidate_built_();
}

std::vector<std::shared_ptr<Node>> Graph::linear_nodes_snapshot(const char* where) const {
  if (!composition_) {
    return {};
  }
  return composition_->linear_nodes_or_throw(where);
}

Graph::CompositionView Graph::composition_view_for_internal_compile() const {
  if (!composition_) {
    throw std::runtime_error("Graph: cannot compile a moved-from composition");
  }
  const bool linear = composition_->is_linear();
  return CompositionView{
      .linear_nodes = linear ? composition_->linear_nodes_or_throw("Graph::compile")
                             : std::vector<std::shared_ptr<Node>>{},
      .vertices = composition_->pipeline_vertices_snapshot(),
      .runtime_vertices = composition_->runtime_vertices_snapshot(),
      .edges =
          std::span<const CompositionEdge>(composition_->edges.data(), composition_->edges.size()),
      .groups = std::span<const GroupMeta>(groups_.data(), groups_.size()),
      .fragments = std::span<const runtime::FragmentPlan>(composition_->fragments.data(),
                                                          composition_->fragments.size()),
      .named_fragments = std::span<const NamedFragment>(composition_->named_fragments.data(),
                                                        composition_->named_fragments.size()),
      .options = opt_,
      .graph_name = endpoint_name_,
      .graph_user_named = !endpoint_name_.empty(),
      .graph_id = graph_id_,
      .graph_version = nodes_version_.load(std::memory_order_relaxed),
      .linear = linear,
  };
}

namespace {

runtime::FragmentPlan rebase_fragment_plan(runtime::FragmentPlan fragment, std::size_t offset) {
  fragment.graph_start += offset;
  fragment.graph_end += offset;
  fragment.provenance.graph_start += offset;
  fragment.provenance.graph_end += offset;
  return fragment;
}

bool in_half_open(std::size_t value, std::size_t start, std::size_t end) {
  return value >= start && value < end;
}

template <typename Composition>
std::size_t unique_default_source_vertex_or_throw(const Composition& composition, std::size_t start,
                                                  std::size_t end, const char* where) {
  if (end <= start || end > composition.vertices.size()) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot select source endpoint for empty Graph fragment");
  }
  std::vector<bool> has_internal_out(end - start, false);
  for (const auto& edge : composition.edges) {
    if (in_half_open(edge.from, start, end) && in_half_open(edge.to, start, end)) {
      has_internal_out[edge.from - start] = true;
    }
  }
  std::size_t candidate = static_cast<std::size_t>(-1);
  for (std::size_t i = 0; i < has_internal_out.size(); ++i) {
    if (has_internal_out[i]) {
      continue;
    }
    if (candidate != static_cast<std::size_t>(-1)) {
      throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                               ": source Graph has multiple output endpoints; name the endpoint "
                               "or use an explicit Branch/Combine fragment");
    }
    candidate = start + i;
  }
  if (candidate == static_cast<std::size_t>(-1)) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": source Graph has no output endpoint");
  }
  return candidate;
}

template <typename Composition>
std::size_t unique_default_sink_vertex_or_throw(const Composition& composition, std::size_t start,
                                                std::size_t end, const char* where) {
  if (end <= start || end > composition.vertices.size()) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot select destination endpoint for empty Graph fragment");
  }
  std::vector<bool> has_internal_in(end - start, false);
  for (const auto& edge : composition.edges) {
    if (in_half_open(edge.from, start, end) && in_half_open(edge.to, start, end)) {
      has_internal_in[edge.to - start] = true;
    }
  }
  std::size_t candidate = static_cast<std::size_t>(-1);
  for (std::size_t i = 0; i < has_internal_in.size(); ++i) {
    if (has_internal_in[i]) {
      continue;
    }
    if (candidate != static_cast<std::size_t>(-1)) {
      throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                               ": destination Graph has multiple input endpoints; name the "
                               "endpoint or use an explicit Branch/Combine fragment");
    }
    candidate = start + i;
  }
  if (candidate == static_cast<std::size_t>(-1)) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": destination Graph has no input endpoint");
  }
  return candidate;
}

struct FragmentEndpointCandidate {
  std::size_t vertex = static_cast<std::size_t>(-1);
  std::string name;
  bool explicit_name = false;
};

struct FragmentEndpointMatch {
  std::size_t from = static_cast<std::size_t>(-1);
  std::size_t to = static_cast<std::size_t>(-1);
  std::string from_name;
  std::string to_name;
};

void add_unique_fragment_endpoint_candidate(std::vector<FragmentEndpointCandidate>* out,
                                            std::size_t vertex, std::string name) {
  if (!out || name.empty()) {
    return;
  }
  const auto dup = std::find_if(out->begin(), out->end(), [&](const auto& existing) {
    return existing.vertex == vertex && existing.name == name;
  });
  if (dup != out->end()) {
    return;
  }
  out->push_back(FragmentEndpointCandidate{
      .vertex = vertex,
      .name = std::move(name),
      .explicit_name = true,
  });
}

template <typename Composition>
void append_fragment_boundary_hint_candidates(const Composition& composition, std::size_t start,
                                              std::size_t end, bool source,
                                              std::vector<FragmentEndpointCandidate>* out,
                                              const char* where) {
  if (!out || end <= start || end > composition.vertices.size()) {
    return;
  }
  for (const auto& fragment : composition.fragments) {
    if (!fragment.boundary_hints.has_value() || fragment.graph_start < start ||
        fragment.graph_end > end || fragment.graph_end <= fragment.graph_start) {
      continue;
    }
    const auto& names = source ? fragment.boundary_hints->egress_endpoint_names
                               : fragment.boundary_hints->ingress_endpoint_names;
    if (names.empty()) {
      continue;
    }
    std::size_t vertex = static_cast<std::size_t>(-1);
    try {
      vertex = source ? unique_default_source_vertex_or_throw(composition, fragment.graph_start,
                                                              fragment.graph_end, where)
                      : unique_default_sink_vertex_or_throw(composition, fragment.graph_start,
                                                            fragment.graph_end, where);
    } catch (const std::exception&) {
      continue;
    }
    for (const auto& name : names) {
      add_unique_fragment_endpoint_candidate(out, vertex, name);
    }
  }
}

template <typename Composition>
std::vector<FragmentEndpointCandidate>
collect_source_endpoint_candidates(const Composition& composition, std::size_t start,
                                   std::size_t end, std::string_view graph_name) {
  (void)graph_name;
  if (end <= start || end > composition.vertices.size()) {
    return {};
  }

  std::vector<FragmentEndpointCandidate> out;
  append_fragment_boundary_hint_candidates(composition, start, end, true, &out, "Graph::connect");
  const bool has_fragment_boundary_hints = !out.empty();

  std::vector<bool> has_internal_out(end - start, false);
  for (const auto& edge : composition.edges) {
    if (in_half_open(edge.from, start, end) && in_half_open(edge.to, start, end)) {
      has_internal_out[edge.from - start] = true;
    }
  }

  for (std::size_t i = 0; i < has_internal_out.size(); ++i) {
    if (has_internal_out[i]) {
      continue;
    }
    const std::size_t vertex = start + i;
    std::string name = explicit_endpoint_name(composition.vertices[vertex]);
    const bool explicit_name = !name.empty();
    if (explicit_name) {
      add_unique_fragment_endpoint_candidate(&out, vertex, std::move(name));
      continue;
    }
    if (has_fragment_boundary_hints) {
      continue;
    }
    if (name.empty() && composition.vertices[vertex]) {
      name =
          endpoint_token(composition.vertices[vertex]->user_label(), "v" + std::to_string(vertex));
      if (name == "v" + std::to_string(vertex)) {
        name = endpoint_token(composition.vertices[vertex]->kind(), name);
      }
    }
    if (name.empty()) {
      name = "v" + std::to_string(vertex);
    }
    out.push_back(FragmentEndpointCandidate{
        .vertex = vertex,
        .name = std::move(name),
        .explicit_name = explicit_name,
    });
  }
  return out;
}

template <typename Composition>
std::vector<FragmentEndpointCandidate>
collect_destination_endpoint_candidates(const Composition& composition, std::size_t start,
                                        std::size_t end, std::string_view graph_name) {
  (void)graph_name;
  if (end <= start || end > composition.vertices.size()) {
    return {};
  }

  std::vector<FragmentEndpointCandidate> out;
  append_fragment_boundary_hint_candidates(composition, start, end, false, &out, "Graph::connect");
  const bool has_fragment_boundary_hints = !out.empty();

  std::vector<bool> has_internal_in(end - start, false);
  for (const auto& edge : composition.edges) {
    if (in_half_open(edge.from, start, end) && in_half_open(edge.to, start, end)) {
      has_internal_in[edge.to - start] = true;
    }
  }

  for (std::size_t i = 0; i < has_internal_in.size(); ++i) {
    if (has_internal_in[i]) {
      continue;
    }
    const std::size_t vertex = start + i;
    std::string name = explicit_endpoint_name(composition.vertices[vertex]);
    const bool explicit_name = !name.empty();
    if (explicit_name) {
      add_unique_fragment_endpoint_candidate(&out, vertex, std::move(name));
      continue;
    }
    if (has_fragment_boundary_hints) {
      continue;
    }
    if (name.empty() && composition.vertices[vertex]) {
      name =
          endpoint_token(composition.vertices[vertex]->user_label(), "v" + std::to_string(vertex));
      if (name == "v" + std::to_string(vertex)) {
        name = endpoint_token(composition.vertices[vertex]->kind(), name);
      }
    }
    if (name.empty()) {
      name = "v" + std::to_string(vertex);
    }
    out.push_back(FragmentEndpointCandidate{
        .vertex = vertex,
        .name = std::move(name),
        .explicit_name = explicit_name,
    });
  }
  return out;
}

std::string join_candidate_names(const std::vector<FragmentEndpointCandidate>& candidates) {
  if (candidates.empty()) {
    return "<none>";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i != 0U) {
      oss << ", ";
    }
    oss << candidates[i].name;
  }
  return oss.str();
}

const FragmentEndpointCandidate*
single_candidate_named(const std::vector<FragmentEndpointCandidate>& candidates,
                       std::string_view name) {
  if (name.empty()) {
    return nullptr;
  }
  const FragmentEndpointCandidate* match = nullptr;
  for (const auto& candidate : candidates) {
    if (candidate.name != name) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &candidate;
  }
  return match;
}

std::optional<FragmentEndpointMatch>
unique_same_name_match(const std::vector<FragmentEndpointCandidate>& sources,
                       const std::vector<FragmentEndpointCandidate>& destinations) {
  std::optional<FragmentEndpointMatch> out;
  for (const auto& source : sources) {
    if (source.name.empty()) {
      continue;
    }
    for (const auto& destination : destinations) {
      if (source.name != destination.name) {
        continue;
      }
      if (out.has_value()) {
        return std::nullopt;
      }
      out = FragmentEndpointMatch{.from = source.vertex,
                                  .to = destination.vertex,
                                  .from_name = source.name,
                                  .to_name = destination.name};
    }
  }
  return out;
}

const FragmentEndpointCandidate*
single_vertex_alias_candidate(const std::vector<FragmentEndpointCandidate>& candidates) {
  if (candidates.empty()) {
    return nullptr;
  }
  const std::size_t vertex = candidates.front().vertex;
  for (const auto& candidate : candidates) {
    if (candidate.vertex != vertex) {
      return nullptr;
    }
  }
  return &candidates.front();
}

FragmentEndpointMatch resolve_fragment_endpoint_match_or_throw(
    const std::vector<FragmentEndpointCandidate>& sources,
    const std::vector<FragmentEndpointCandidate>& destinations, std::string_view source_graph_name,
    std::string_view destination_graph_name, const char* where) {
  (void)source_graph_name;
  (void)destination_graph_name;
  const std::string context = where ? where : "Graph::connect";
  if (sources.empty()) {
    throw std::runtime_error(context + ": source Graph has no output endpoint");
  }
  if (destinations.empty()) {
    throw std::runtime_error(context + ": destination Graph has no input endpoint");
  }

  if (sources.size() == 1U && destinations.size() == 1U) {
    return FragmentEndpointMatch{.from = sources.front().vertex,
                                 .to = destinations.front().vertex,
                                 .from_name = sources.front().name,
                                 .to_name = destinations.front().name};
  }

  if (sources.size() == 1U) {
    if (const auto* destination = single_candidate_named(destinations, sources.front().name)) {
      return FragmentEndpointMatch{.from = sources.front().vertex,
                                   .to = destination->vertex,
                                   .from_name = sources.front().name,
                                   .to_name = destination->name};
    }
  }

  if (destinations.size() == 1U) {
    if (const auto* source = single_candidate_named(sources, destinations.front().name)) {
      return FragmentEndpointMatch{.from = source->vertex,
                                   .to = destinations.front().vertex,
                                   .from_name = source->name,
                                   .to_name = destinations.front().name};
    }
  }

  if (auto same_name = unique_same_name_match(sources, destinations)) {
    return *same_name;
  }

  if (destinations.size() == 1U) {
    if (const auto* source = single_vertex_alias_candidate(sources)) {
      return FragmentEndpointMatch{.from = source->vertex,
                                   .to = destinations.front().vertex,
                                   .from_name = source->name,
                                   .to_name = destinations.front().name};
    }
  }

  std::ostringstream oss;
  oss << context << ": cannot infer endpoints between Graph fragments"
      << "; source outputs: " << join_candidate_names(sources)
      << "; destination inputs: " << join_candidate_names(destinations)
      << "; fix by declaring matching Input(\"name\") / Output(\"name\") endpoints";
  throw std::runtime_error(oss.str());
}

} // namespace

std::pair<std::size_t, std::size_t> Graph::append_linear_fragment_(const Graph& fragment,
                                                                   const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!fragment.composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::add") +
                             ": cannot append a moved-from Graph fragment");
  }
  if (composition_ && !composition_->is_linear()) {
    throw std::runtime_error("Graph::add after branching is ambiguous; use connect()");
  }
  const auto nodes = fragment.composition_->linear_nodes_or_throw(where);
  const auto start = composition_->size();
  for (const auto& node : nodes) {
    composition_->append_vertex(node);
  }
  const auto end = composition_->size();
  for (const auto& group : fragment.groups_) {
    if (group.end <= group.start)
      continue;
    groups_.push_back(GroupMeta{
        .start = start + group.start,
        .end = start + group.end,
        .caps_behavior = group.caps_behavior,
        .label = group.label,
    });
  }
  for (const auto& meta : fragment.composition_->fragments) {
    composition_->fragments.push_back(rebase_fragment_plan(meta, start));
  }
  for (const auto& named : fragment.composition_->named_fragments) {
    if (named.end <= named.start)
      continue;
    composition_->named_fragments.push_back(NamedFragment{
        .start = start + named.start,
        .end = start + named.end,
        .name = named.name,
        .user_named = named.user_named,
    });
  }
  if (fragment.input_route_processor_) {
    if (input_route_processor_ && input_route_processor_ != fragment.input_route_processor_) {
      throw std::runtime_error(std::string(where ? where : "Graph::add") +
                               ": cannot merge multiple input route processors; use explicit "
                               "Sample routing or connect() fragments at the top level");
    }
    input_route_processor_ = fragment.input_route_processor_;
  }
  return {start, end};
}

std::pair<std::size_t, std::size_t> Graph::import_composition_fragment_(const Graph& fragment,
                                                                        const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!fragment.composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import a moved-from Graph fragment");
  }

  const auto& source = *fragment.composition_;
  const std::size_t start = composition_->vertices.size();
  composition_->vertices.insert(composition_->vertices.end(), source.vertices.begin(),
                                source.vertices.end());
  const std::size_t end = composition_->vertices.size();

  for (const auto& edge : source.edges) {
    if (edge.from >= source.vertices.size() || edge.to >= source.vertices.size()) {
      throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                               ": source Graph contains an invalid composition edge");
    }
    CompositionEdge rebased = edge;
    rebased.from += start;
    rebased.to += start;
    composition_->edges.push_back(std::move(rebased));
  }
  composition_->endpoint_mode = composition_->endpoint_mode || source.endpoint_mode;

  for (const auto& group : fragment.groups_) {
    if (group.end <= group.start) {
      continue;
    }
    groups_.push_back(GroupMeta{
        .start = start + group.start,
        .end = start + group.end,
        .caps_behavior = group.caps_behavior,
        .label = group.label,
    });
  }
  for (const auto& meta : source.fragments) {
    composition_->fragments.push_back(rebase_fragment_plan(meta, start));
  }
  for (const auto& named : source.named_fragments) {
    if (named.end <= named.start) {
      continue;
    }
    composition_->named_fragments.push_back(NamedFragment{
        .start = start + named.start,
        .end = start + named.end,
        .name = named.name,
        .user_named = named.user_named,
    });
  }

  if (end > start) {
    const bool user_named = !fragment.endpoint_name_.empty();
    const std::string name =
        user_named
            ? fragment.endpoint_name_
            : infer_endpoint_name_from_nodes(source.vertices, "fragment" + std::to_string(start));
    const bool duplicate_wrapper =
        std::any_of(composition_->named_fragments.begin(), composition_->named_fragments.end(),
                    [&](const NamedFragment& named) {
                      return named.start == start && named.end == end && named.name == name &&
                             named.user_named == user_named;
                    });
    if (!duplicate_wrapper) {
      composition_->named_fragments.push_back(NamedFragment{
          .start = start,
          .end = end,
          .name = name,
          .user_named = user_named,
      });
    }
  }

  if (fragment.input_route_processor_) {
    if (input_route_processor_ && input_route_processor_ != fragment.input_route_processor_) {
      throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                               ": cannot merge multiple input route processors; use explicit "
                               "Sample routing or split model ingress at the top level");
    }
    input_route_processor_ = fragment.input_route_processor_;
  }

  composition_->recompute_unique_tail();
  return {start, end};
}

bool Graph::is_output_collection_fragment_(const Graph& fragment) const {
  if (!fragment.composition_ || fragment.composition_->vertices.size() <= 1U) {
    return false;
  }
  return std::all_of(fragment.composition_->vertices.begin(), fragment.composition_->vertices.end(),
                     [](const std::shared_ptr<Node>& node) {
                       return dynamic_cast<const Output*>(node.get()) != nullptr;
                     });
}

std::pair<std::size_t, std::size_t> Graph::import_output_collection_fragment_(const Graph& fragment,
                                                                              const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!fragment.composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import a moved-from Graph fragment");
  }
  if (!is_output_collection_fragment_(fragment)) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": expected a Graph containing only Output nodes");
  }

  const auto& nodes = fragment.composition_->vertices;
  const std::size_t start = composition_->vertices.size();
  composition_->vertices.insert(composition_->vertices.end(), nodes.begin(), nodes.end());
  const std::size_t end = composition_->vertices.size();
  if (end > start) {
    composition_->tail = end - 1U;
  }

  for (std::size_t i = start; i < end; ++i) {
    const auto& node = composition_->vertices[i];
    groups_.push_back(GroupMeta{
        .start = i,
        .end = i + 1U,
        .caps_behavior = node ? node->caps_behavior() : NodeCapsBehavior::Dynamic,
        .label = "",
    });
  }

  const bool user_named = !fragment.endpoint_name_.empty();
  const std::string name =
      user_named ? fragment.endpoint_name_
                 : infer_endpoint_name_from_nodes(nodes, "fragment" + std::to_string(start));
  composition_->named_fragments.push_back(NamedFragment{
      .start = start,
      .end = end,
      .name = name,
      .user_named = user_named,
  });
  return {start, end};
}

std::pair<std::size_t, std::size_t>
Graph::import_or_reuse_composition_fragment_(const Graph& fragment, const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (&fragment == this) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import this Graph into itself");
  }
  if (!fragment.composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import a moved-from Graph fragment");
  }

  const std::uint64_t source_version = fragment.nodes_version_.load(std::memory_order_relaxed);
  auto existing = composition_->imported_fragments.find(fragment.graph_id_);
  if (existing != composition_->imported_fragments.end()) {
    if (existing->second.source_version != source_version) {
      throw std::runtime_error(
          std::string(where ? where : "Graph::connect") +
          ": source Graph was modified after it was imported into this Graph; create a separate "
          "Graph fragment if you intended a second copy");
    }
    return {existing->second.start, existing->second.end};
  }

  const auto [start, end] = import_composition_fragment_(fragment, where);
  composition_->imported_fragments.emplace(
      fragment.graph_id_, CompositionGraph::ImportedFragment{
                              .source_version = source_version, .start = start, .end = end});
  return {start, end};
}

std::pair<std::size_t, std::size_t>
Graph::import_or_reuse_output_collection_fragment_(const Graph& fragment, const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (&fragment == this) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import this Graph into itself");
  }
  if (!fragment.composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot import a moved-from Graph fragment");
  }

  const std::uint64_t source_version = fragment.nodes_version_.load(std::memory_order_relaxed);
  auto existing = composition_->imported_fragments.find(fragment.graph_id_);
  if (existing != composition_->imported_fragments.end()) {
    if (existing->second.source_version != source_version) {
      throw std::runtime_error(
          std::string(where ? where : "Graph::connect") +
          ": source Graph was modified after it was imported into this Graph; create a separate "
          "Graph fragment if you intended a second copy");
    }
    return {existing->second.start, existing->second.end};
  }

  const auto [start, end] = import_output_collection_fragment_(fragment, where);
  composition_->imported_fragments.emplace(
      fragment.graph_id_, CompositionGraph::ImportedFragment{
                              .source_version = source_version, .start = start, .end = end});
  return {start, end};
}

std::pair<std::size_t, std::size_t>
Graph::import_or_reuse_node_fragment_(std::shared_ptr<Node> node, const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!node) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot connect a null Node");
  }

  const Node* key = node.get();
  auto existing = composition_->imported_nodes.find(key);
  if (existing != composition_->imported_nodes.end()) {
    return {existing->second.start, existing->second.end};
  }

  const std::size_t start = composition_->vertices.size();
  composition_->vertices.emplace_back(std::move(node));
  const std::size_t end = composition_->vertices.size();

  const auto& inserted = composition_->vertices[start];
  groups_.push_back(GroupMeta{
      .start = start,
      .end = end,
      .caps_behavior = inserted ? inserted->caps_behavior() : NodeCapsBehavior::Dynamic,
      .label = "",
  });

  std::string name = explicit_endpoint_name(inserted);
  if (name.empty() && inserted) {
    name = endpoint_token(inserted->user_label(), "v" + std::to_string(start));
    if (name == "v" + std::to_string(start)) {
      name = endpoint_token(inserted->kind(), name);
    }
  }
  composition_->named_fragments.push_back(NamedFragment{
      .start = start,
      .end = end,
      .name = name,
      .user_named = !explicit_endpoint_name(inserted).empty(),
  });

  composition_->imported_nodes.emplace(
      key, CompositionGraph::ImportedFragment{.source_version = 0, .start = start, .end = end});
  composition_->recompute_unique_tail();
  return {start, end};
}

std::pair<std::size_t, std::size_t> Graph::import_or_reuse_model_fragment_(const Model& model,
                                                                           const char* where) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }

  Graph fragment = model.graph();
  if (fragment.name().empty()) {
    fragment.set_name(model_endpoint_name(model));
  }
  const std::string key = model_import_key(model, fragment);
  auto existing = composition_->imported_models.find(key);
  if (existing != composition_->imported_models.end()) {
    return {existing->second.start, existing->second.end};
  }

  const auto [start, end] = import_composition_fragment_(fragment, where);
  composition_->imported_models.emplace(
      key, CompositionGraph::ImportedFragment{.source_version = 0, .start = start, .end = end});
  return {start, end};
}

void Graph::connect_imported_ranges_(std::pair<std::size_t, std::size_t> from_range,
                                     std::string_view from_name,
                                     std::pair<std::size_t, std::size_t> to_range,
                                     std::string_view to_name, const char* where) {
  if (!composition_) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot connect without a composition");
  }
  const auto [from_start, from_end] = from_range;
  const auto [to_start, to_end] = to_range;
  if (from_end <= from_start || to_end <= to_start) {
    throw std::runtime_error(std::string(where ? where : "Graph::connect") +
                             ": cannot connect empty fragments");
  }

  const auto source_candidates =
      collect_source_endpoint_candidates(*composition_, from_start, from_end, from_name);
  const auto destination_candidates =
      collect_destination_endpoint_candidates(*composition_, to_start, to_end, to_name);
  const FragmentEndpointMatch match = resolve_fragment_endpoint_match_or_throw(
      source_candidates, destination_candidates, from_name, to_name, where);
  composition_->connect_endpoint(match.from, match.to, match.from_name, match.to_name);
}

void Graph::attach_fragment_boundary_hints_(std::size_t start, std::size_t end,
                                            runtime::FragmentBoundaryHints hints) {
  runtime::Provenance provenance;
  attach_fragment_boundary_hints_(start, end, std::move(hints), std::move(provenance));
}

void Graph::attach_fragment_boundary_hints_(std::size_t start, std::size_t end,
                                            runtime::FragmentBoundaryHints hints,
                                            runtime::Provenance provenance) {
  if (!composition_) {
    throw std::runtime_error("Graph::attach_fragment_boundary_hints: moved-from Graph");
  }
  if (end < start || end > composition_->size()) {
    throw std::runtime_error("Graph::attach_fragment_boundary_hints: fragment range out of range");
  }
  if (end == start) {
    return;
  }
  runtime::FragmentPlan fragment;
  fragment.graph_start = start;
  fragment.graph_end = end;
  fragment.boundary_hints = std::move(hints);
  fragment.provenance = std::move(provenance);
  fragment.provenance.graph_start = start;
  fragment.provenance.graph_end = end;
  if (fragment.boundary_hints->input_route_processor) {
    if (input_route_processor_ &&
        input_route_processor_ != fragment.boundary_hints->input_route_processor) {
      throw std::runtime_error(
          "Graph::attach_fragment_boundary_hints: cannot merge multiple input route processors");
    }
    input_route_processor_ = fragment.boundary_hints->input_route_processor;
  }
  composition_->fragments.push_back(std::move(fragment));
}

std::size_t Graph::append_pipeline_vertex_for_internal_graph_(std::shared_ptr<Node> node) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!node) {
    throw std::runtime_error("Graph::append_pipeline_vertex_for_internal_graph_: node is null");
  }

  const std::size_t id = composition_->vertices.size();
  const NodeCapsBehavior behavior = node->caps_behavior();
  composition_->vertices.emplace_back(std::move(node));
  composition_->tail = CompositionGraph::kInvalid;
  groups_.push_back(GroupMeta{
      .start = id,
      .end = id + 1U,
      .caps_behavior = behavior,
      .label = "",
  });
  mark_composition_changed();
  return id;
}

std::size_t
Graph::append_runtime_vertex_for_internal_graph_(std::shared_ptr<simaai::neat::graph::Node> node) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (!node) {
    throw std::runtime_error("Graph::append_runtime_vertex_for_internal_graph_: node is null");
  }
  const std::size_t id = composition_->vertices.size();
  composition_->vertices.push_back(CompositionGraph::CompositionVertex::runtime(std::move(node)));
  composition_->tail = CompositionGraph::kInvalid;
  mark_composition_changed();
  return id;
}

void Graph::connect_runtime_port_for_internal_graph_(std::size_t from, std::string_view from_port,
                                                     std::size_t to, std::string_view to_port) {
  if (!composition_) {
    throw std::runtime_error("Graph::connect_runtime_port_for_internal_graph_: moved-from Graph");
  }
  composition_->connect_runtime_port(from, to, std::string(from_port), std::string(to_port));
  mark_composition_changed();
}

Graph& Graph::add(std::shared_ptr<Node> node) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  const NodeCapsBehavior behavior = node ? node->caps_behavior() : NodeCapsBehavior::Dynamic;
  const auto [start, end] = composition_->append_node(std::move(node));
  if (end > start) {
    groups_.push_back({start, end, behavior, ""});
  }
  mark_composition_changed();
  return *this;
}

Graph& Graph::add(const Graph& fragment) {
  if (!fragment.composition_) {
    throw std::runtime_error("Graph::add(Graph): cannot append a moved-from Graph fragment");
  }
  if (fragment.composition_->is_linear()) {
    append_linear_fragment_(fragment, "Graph::add(Graph)");
  } else {
    if (composition_ && !composition_->empty()) {
      throw std::runtime_error(
          "Graph::add(Graph): cannot append a connected/runtime Graph fragment to a non-empty "
          "Graph; use connect() to compose topology explicitly");
    }
    import_composition_fragment_(fragment, "Graph::add(Graph)");
  }
  mark_composition_changed();
  return *this;
}

Graph& Graph::add(Graph&& fragment) {
  return add(static_cast<const Graph&>(fragment));
}

Graph& Graph::connect(std::string_view from_endpoint, std::string_view to_endpoint) {
  if (!composition_) {
    composition_ = std::make_unique<CompositionGraph>();
  }
  if (composition_->empty()) {
    throw std::runtime_error("Graph::connect: cannot resolve endpoints in an empty Graph");
  }

  const std::string from_name = trim_endpoint_name(from_endpoint);
  const std::string to_name = trim_endpoint_name(to_endpoint);
  if (from_name.empty() || to_name.empty()) {
    throw std::runtime_error("Graph::connect: endpoint name must not be empty");
  }

  const auto from =
      resolve_boundary_endpoint_or_throw(*composition_, from_name, "source", "Graph::connect");
  const auto to =
      resolve_boundary_endpoint_or_throw(*composition_, to_name, "destination", "Graph::connect");
  promote_endpoint_mode_or_throw(*composition_, from_name, to_name, "Graph::connect");
  composition_->connect_endpoint(from, to, from_name, to_name);
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Graph& from, const Graph& to) {
  return connect(from, to, GraphLinkOptions{});
}

Graph& Graph::connect(const Graph& from, const Graph& to, const GraphLinkOptions& options) {
  return connect_with_link_options_(from, to, options);
}

Graph& Graph::connect_with_link_options_(const Graph& from, const Graph& to,
                                         const GraphLinkOptions& options) {
  const auto [from_start, from_end] =
      import_or_reuse_composition_fragment_(from, "Graph::connect(from)");
  if (from_end <= from_start) {
    throw std::runtime_error("Graph::connect: cannot connect empty Graph fragments");
  }
  const auto source_candidates =
      collect_source_endpoint_candidates(*composition_, from_start, from_end, from.endpoint_name_);
  if (source_candidates.empty()) {
    throw std::runtime_error("Graph::connect: source Graph has no output endpoint");
  }

  if (is_output_collection_fragment_(to)) {
    const auto [to_start, to_end] =
        import_or_reuse_output_collection_fragment_(to, "Graph::connect(to)");
    if (to_end <= to_start) {
      throw std::runtime_error("Graph::connect: cannot connect empty Graph fragments");
    }
    if (source_candidates.size() != 1U) {
      throw std::runtime_error(
          "Graph::connect: source Graph has multiple output endpoints; cannot connect to an "
          "output collection without an explicit Branch fragment");
    }
    const std::size_t from_vertex = source_candidates.front().vertex;
    for (std::size_t output = to_start; output < to_end; ++output) {
      std::string to_name = explicit_endpoint_name(composition_->vertices[output]);
      if (to_name.empty()) {
        to_name = to.endpoint_name_.empty()
                      ? "output_" + std::to_string(output - to_start)
                      : to.endpoint_name_ + "_" + std::to_string(output - to_start);
      }
      composition_->connect_endpoint(from_vertex, output, source_candidates.front().name,
                                     std::move(to_name), options);
    }
  } else {
    const auto [to_start, to_end] = import_or_reuse_composition_fragment_(to, "Graph::connect(to)");
    if (to_end <= to_start) {
      throw std::runtime_error("Graph::connect: cannot connect empty Graph fragments");
    }
    const auto destination_candidates =
        collect_destination_endpoint_candidates(*composition_, to_start, to_end, to.endpoint_name_);
    const FragmentEndpointMatch match = resolve_fragment_endpoint_match_or_throw(
        source_candidates, destination_candidates, from.endpoint_name_, to.endpoint_name_,
        "Graph::connect");
    composition_->connect_endpoint(match.from, match.to, match.from_name, match.to_name, options);
  }
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(std::shared_ptr<Node> from, std::shared_ptr<Node> to) {
  const auto from_range = import_or_reuse_node_fragment_(std::move(from), "Graph::connect(from)");
  const auto to_range = import_or_reuse_node_fragment_(std::move(to), "Graph::connect(to)");
  connect_imported_ranges_(from_range, {}, to_range, {}, "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Graph& from, std::shared_ptr<Node> to) {
  const auto from_range = import_or_reuse_composition_fragment_(from, "Graph::connect(from)");
  const auto to_range = import_or_reuse_node_fragment_(std::move(to), "Graph::connect(to)");
  connect_imported_ranges_(from_range, from.endpoint_name_, to_range, {}, "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(std::shared_ptr<Node> from, const Graph& to) {
  const auto from_range = import_or_reuse_node_fragment_(std::move(from), "Graph::connect(from)");
  const auto to_range = is_output_collection_fragment_(to)
                            ? import_or_reuse_output_collection_fragment_(to, "Graph::connect(to)")
                            : import_or_reuse_composition_fragment_(to, "Graph::connect(to)");
  connect_imported_ranges_(from_range, {}, to_range, to.endpoint_name_, "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Model& from, const Model& to) {
  const auto from_range = import_or_reuse_model_fragment_(from, "Graph::connect(from)");
  const auto to_range = import_or_reuse_model_fragment_(to, "Graph::connect(to)");
  connect_imported_ranges_(from_range, model_endpoint_name(from), to_range, model_endpoint_name(to),
                           "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Model& from, const Graph& to) {
  const auto from_range = import_or_reuse_model_fragment_(from, "Graph::connect(from)");
  const auto to_range = is_output_collection_fragment_(to)
                            ? import_or_reuse_output_collection_fragment_(to, "Graph::connect(to)")
                            : import_or_reuse_composition_fragment_(to, "Graph::connect(to)");
  connect_imported_ranges_(from_range, model_endpoint_name(from), to_range, to.endpoint_name_,
                           "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Graph& from, const Model& to) {
  const auto from_range = import_or_reuse_composition_fragment_(from, "Graph::connect(from)");
  const auto to_range = import_or_reuse_model_fragment_(to, "Graph::connect(to)");
  connect_imported_ranges_(from_range, from.endpoint_name_, to_range, model_endpoint_name(to),
                           "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(const Model& from, std::shared_ptr<Node> to) {
  const auto from_range = import_or_reuse_model_fragment_(from, "Graph::connect(from)");
  const auto to_range = import_or_reuse_node_fragment_(std::move(to), "Graph::connect(to)");
  connect_imported_ranges_(from_range, model_endpoint_name(from), to_range, {}, "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::connect(std::shared_ptr<Node> from, const Model& to) {
  const auto from_range = import_or_reuse_node_fragment_(std::move(from), "Graph::connect(from)");
  const auto to_range = import_or_reuse_model_fragment_(to, "Graph::connect(to)");
  connect_imported_ranges_(from_range, {}, to_range, model_endpoint_name(to), "Graph::connect");
  mark_composition_changed();
  return *this;
}

Graph& Graph::custom(std::string fragment) {
  return add(nodes::Custom(std::move(fragment)));
}

Graph& Graph::custom(std::string fragment, InputRole role) {
  return add(nodes::Custom(std::move(fragment), role));
}

void Graph::set_guard(std::shared_ptr<void> guard) {
  guard_ = std::move(guard);
  invalidate_built_();
}

void Graph::set_tensor_callback(TensorCallback cb) {
  tensor_cb_ = std::move(cb);
}

Graph& Graph::add_output_tensor(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  if (o.format.empty())
    o.format = FormatTag::RGB;
  if (o.dtype != TensorDType::UInt8) {
    throw std::runtime_error("add_output_tensor: only UInt8 is supported for now");
  }

  // Normalize to a CPU-friendly raw-video tensor path.
  add(nodes::VideoConvert());
  add(nodes::VideoScale());

  // Force SystemMemory to keep CPU-accessible tensors for future bindings.
  add(nodes::CapsRaw(o.format.str(), o.target_width, o.target_height, o.target_fps,
                     simaai::neat::CapsMemory::SystemMemory));
  add(nodes::Output());
  return *this;
}

} // namespace simaai::neat

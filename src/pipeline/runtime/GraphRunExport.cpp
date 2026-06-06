#include "pipeline/RunExport.h"

#include "RunInternal.h"

#include "builder/OutputSpec.h"
#include "model/internal/ModelRouteRetarget.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/Tess.h"
#include "pipeline/GraphMetrics.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/PipelineSegmentRuntime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

namespace simaai::neat {
namespace {

using json = nlohmann::ordered_json;

std::string node_ref(graph::NodeId id) {
  return id == graph::kInvalidNode ? std::string("invalid") : ("n" + std::to_string(id));
}

std::string port_name(const runtime::ExecutionGraphPlan& plan, graph::PortId port) {
  if (port == graph::kInvalidPort) {
    return {};
  }
  return port < plan.port_names.size() ? plan.port_names[port] : ("port" + std::to_string(port));
}

const char* endpoint_kind_name(runtime::Endpoint::Kind kind) {
  switch (kind) {
  case runtime::Endpoint::Kind::PipelineInput:
    return "pipeline_input";
  case runtime::Endpoint::Kind::PipelineOutput:
    return "pipeline_output";
  case runtime::Endpoint::Kind::StageInput:
    return "stage_input";
  case runtime::Endpoint::Kind::StageOutput:
    return "stage_output";
  case runtime::Endpoint::Kind::GraphSink:
    return "graph_sink";
  }
  return "unknown";
}

json endpoint_to_json(const runtime::ExecutionGraphPlan& plan, const runtime::Endpoint& endpoint) {
  json j;
  j["kind"] = endpoint_kind_name(endpoint.kind);
  j["node"] = node_ref(endpoint.node);
  const std::string port = port_name(plan, endpoint.port);
  j["port"] = port.empty() ? json(nullptr) : json(port);
  if (endpoint.segment != static_cast<std::size_t>(-1)) {
    j["segment"] = endpoint.segment;
  }
  return j;
}

json power_summary_json(const PowerSummary& summary) {
  return json::parse(power_summary_to_json(summary, 0));
}

json latency_summary_json(const NodeLatencySummary& summary) {
  return {
      {"samples", summary.samples},
      {"total_ms", summary.total_ms},
      {"avg_ms", summary.avg_ms},
      {"min_ms", summary.min_ms},
      {"max_ms", summary.max_ms},
      {"min_max_available", summary.min_max_available},
  };
}

json latency_summary_json(std::uint64_t samples, double total_ms, double avg_ms, double min_ms,
                          double max_ms, bool min_max_available = true) {
  return {
      {"samples", samples},
      {"total_ms", total_ms},
      {"avg_ms", avg_ms},
      {"min_ms", min_ms},
      {"max_ms", max_ms},
      {"min_max_available", min_max_available},
  };
}

json plugin_metric_to_json(const MeasurePluginLatency& p, const std::string& mapping_error = {}) {
  json out;
  out["name"] = p.name.empty() ? json(nullptr) : json(p.name);
  out["backend"] = p.backend;
  out["phase"] = p.phase;
  out["kernel_name"] = p.kernel_name;
  out["stage_name"] = p.stage_name;
  out["gst_element_name"] = p.gst_element_name.empty() ? json(nullptr) : json(p.gst_element_name);
  out["run_id_hash"] = p.run_id_hash == 0 ? json(nullptr) : json(p.run_id_hash);
  out["pipeline_segment_id"] =
      p.pipeline_segment_id >= 0 ? json(p.pipeline_segment_id) : json(nullptr);
  out["runtime_node_id"] = p.runtime_node_id >= 0 ? json(p.runtime_node_id) : json(nullptr);
  out["public_node_id"] = p.public_node_id >= 0 ? json(p.public_node_id) : json(nullptr);
  std::vector<std::string> public_node_ids = p.public_node_ids;
  if (p.public_node_id >= 0) {
    const std::string public_id = "p" + std::to_string(p.public_node_id);
    if (std::find(public_node_ids.begin(), public_node_ids.end(), public_id) ==
        public_node_ids.end()) {
      public_node_ids.push_back(public_id);
    }
  }
  out["public_node_ids"] = std::move(public_node_ids);
  out["physical_input_index"] = p.physical_input_index;
  out["output_slot"] = p.output_slot;
  out["calls"] = p.calls;
  out["latency_ms"] =
      latency_summary_json(p.calls, p.total_ms, p.avg_ms, p.min_ms, p.max_ms, p.calls > 0);
  if (!mapping_error.empty()) {
    out["mapping_error"] = mapping_error;
  }
  return out;
}

bool node_metric_has_samples(const GraphNodeMetrics& node) {
  if (node.latency.samples > 0 || node.latency.total_ms > 0.0) {
    return true;
  }
  for (const GraphElementMetrics& element : node.elements) {
    if (element.latency.samples > 0 || element.latency.total_ms > 0.0) {
      return true;
    }
  }
  return false;
}

json node_metrics_to_json(const std::vector<GraphNodeMetrics>& nodes,
                          const std::string& aggregation,
                          const std::string& latency_semantics,
                          bool include_empty_node_metrics,
                          bool include_plugin_metrics) {
  json node_metrics = json::array();
  for (const GraphNodeMetrics& node : nodes) {
    if (!include_empty_node_metrics && !node_metric_has_samples(node)) {
      continue;
    }
    json n;
    n["node_id"] = node.node_id.empty() ? json(nullptr) : json(node.node_id);
    n["runtime_node"] = node.node_id.empty() ? json(nullptr) : json(node.node_id);
    n["runtime_node_id"] =
        node.runtime_node_id == graph::kInvalidNode
            ? json(nullptr)
            : json(static_cast<std::uint64_t>(node.runtime_node_id));
    n["public_node_ids"] = node.public_node_ids;
    n["pipeline_segment_id"] =
        node.pipeline_segment_id == static_cast<std::size_t>(-1)
            ? json(nullptr)
            : json(node.pipeline_segment_id);
    n["kind"] = node.kind.empty() ? json(nullptr) : json(node.kind);
    n["label"] = node.label.empty() ? json(nullptr) : json(node.label);
    n["element_names"] = node.element_names;
    n["latency_semantics"] = latency_semantics;
    n["aggregation"] = aggregation;
    n["latency_ms"] = latency_summary_json(node.latency);
    if (include_plugin_metrics) {
      n["plugins"] = json::array();
    }
    json elements = json::array();
    for (const GraphElementMetrics& element : node.elements) {
      elements.push_back({
          {"name", element.name},
          {"latency_ms", latency_summary_json(element.latency)},
      });
    }
    n["elements"] = std::move(elements);
    node_metrics.push_back(std::move(n));
  }
  return node_metrics;
}

json output_spec_to_json(const OutputSpec& spec) {
  json j;
  j["media_type"] = spec.media_type;
  j["format"] = spec.format;
  j["width"] = spec.width;
  j["height"] = spec.height;
  j["depth"] = spec.depth;
  if (spec.fps_num > 0 && spec.fps_den > 0) {
    j["fps"] = static_cast<double>(spec.fps_num) / static_cast<double>(spec.fps_den);
  } else {
    j["fps"] = nullptr;
  }
  j["memory"] = spec.memory;
  j["dtype"] = spec.dtype.empty() ? json(nullptr) : json(spec.dtype);
  j["layout"] = spec.layout.empty() ? json(nullptr) : json(spec.layout);
  j["byte_size"] = spec.byte_size;
  j["note"] = spec.note;
  return j;
}

json named_endpoints_to_json(const runtime::ExecutionGraphPlan& plan,
                             const std::unordered_map<std::string, runtime::Endpoint>& endpoints) {
  json arr = json::array();
  for (const auto& [name, endpoint] : endpoints) {
    json item = endpoint_to_json(plan, endpoint);
    item["name"] = name;
    arr.push_back(std::move(item));
  }
  return arr;
}

bool compiler_generated_runtime_node(const std::string& kind, const std::string& label) {
  return kind.find("FanOut") != std::string::npos || kind.find("JoinBundle") != std::string::npos ||
         label.rfind("fanout", 0) == 0 || label.rfind("combine_", 0) == 0;
}

std::string iso8601_utc(std::chrono::system_clock::time_point tp) {
  if (tp.time_since_epoch().count() == 0) {
    return {};
  }
  const std::time_t t = std::chrono::system_clock::to_time_t(tp);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32] = {};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string hostname_safe() {
#if defined(__linux__)
  char buf[HOST_NAME_MAX + 1] = {};
  if (gethostname(buf, HOST_NAME_MAX) == 0) {
    return buf;
  }
#endif
  return {};
}

std::vector<std::string> argv_safe() {
  std::vector<std::string> out;
#if defined(__linux__)
  std::ifstream in("/proc/self/cmdline", std::ios::binary);
  if (!in) {
    return out;
  }
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::size_t start = 0;
  for (std::size_t i = 0; i <= data.size(); ++i) {
    if (i == data.size() || data[i] == '\0') {
      if (i > start) {
        out.emplace_back(data.data() + start, i - start);
      }
      start = i + 1U;
    }
  }
#endif
  return out;
}

int pid_safe() {
#if defined(__linux__)
  return static_cast<int>(getpid());
#else
  return 0;
#endif
}

const char* model_stage_role_name(internal::ModelLineageStageRole role) {
  switch (role) {
  case internal::ModelLineageStageRole::Preprocess:
    return "preprocess";
  case internal::ModelLineageStageRole::Infer:
    return "infer";
  case internal::ModelLineageStageRole::ManualPost:
    return "manual_post";
  }
  return "unknown";
}

const internal::ModelLineageBinding* model_lineage_for_node(const std::shared_ptr<Node>& node) {
  if (!node) {
    return nullptr;
  }
  if (auto* pre = dynamic_cast<Preproc*>(node.get())) {
    return pre->options().model_lineage.get();
  }
  if (auto* quant = dynamic_cast<Quant*>(node.get())) {
    return quant->options().model_lineage.get();
  }
  if (auto* tess = dynamic_cast<Tess*>(node.get())) {
    return tess->options().model_lineage.get();
  }
  if (auto* quanttess = dynamic_cast<QuantTess*>(node.get())) {
    return quanttess->options().model_lineage.get();
  }
  if (auto* cast = dynamic_cast<Cast*>(node.get())) {
    return cast->options().model_lineage.get();
  }
  if (auto* casttess = dynamic_cast<CastTess*>(node.get())) {
    return casttess->options().model_lineage.get();
  }
  if (auto* box = dynamic_cast<SimaBoxDecode*>(node.get())) {
    return box->model_lineage_binding_internal().get();
  }
  if (auto* provider = dynamic_cast<const internal::ModelLineageProvider*>(node.get())) {
    return provider->model_lineage_binding();
  }
  return nullptr;
}

json model_lineage_to_json(const internal::ModelLineageBinding& binding) {
  json j;
  j["id"] = binding.lineage_key;
  j["source_path"] = binding.source_path.empty() ? json(nullptr) : json(binding.source_path);
  j["stage_role"] = model_stage_role_name(binding.stage_role);
  j["requested_post"] = internal::requested_post_route_name(binding.requested_post);
  j["requester_kind"] =
      binding.requester_kind.empty() ? json(nullptr) : json(binding.requester_kind);
  if (!binding.source_path.empty()) {
    const std::filesystem::path path(binding.source_path);
    j["name"] = path.filename().string();
  }
  return j;
}

json model_provenance_to_json(const runtime::Provenance& provenance) {
  json j;
  j["id"] = provenance.model_id.empty() ? json(nullptr) : json(provenance.model_id);
  j["source_path"] =
      provenance.model_source_path.empty() ? json(nullptr) : json(provenance.model_source_path);
  j["stage_role"] =
      provenance.model_stage_role.empty() ? json(nullptr) : json(provenance.model_stage_role);
  if (!provenance.model_source_path.empty()) {
    const std::filesystem::path path(provenance.model_source_path);
    j["name"] = path.filename().string();
  }
  return j;
}

const runtime::FragmentPlan*
model_fragment_for_public_index(const runtime::ExecutionGraphPlan& plan, std::size_t public_index) {
  for (const auto& fragment : plan.fragments) {
    if (fragment.provenance.model_source_path.empty()) {
      continue;
    }
    if (public_index >= fragment.graph_start && public_index < fragment.graph_end) {
      return &fragment;
    }
  }
  return nullptr;
}

const runtime::FragmentPlan*
model_fragment_for_public_range(const runtime::ExecutionGraphPlan& plan, std::size_t start,
                                std::size_t end) {
  const runtime::FragmentPlan* best = nullptr;
  std::size_t best_overlap = 0;
  for (const auto& fragment : plan.fragments) {
    if (fragment.provenance.model_source_path.empty()) {
      continue;
    }
    if (start >= end || start >= fragment.graph_end || fragment.graph_start >= end) {
      continue;
    }
    const std::size_t overlap =
        std::min(end, fragment.graph_end) - std::max(start, fragment.graph_start);
    if (overlap > best_overlap) {
      best = &fragment;
      best_overlap = overlap;
    }
  }
  return best;
}

const runtime::FragmentPlan*
model_fragment_for_segment(const runtime::ExecutionGraphPlan& plan,
                           const runtime::PipelineSegmentPlan& segment) {
  std::size_t start = std::numeric_limits<std::size_t>::max();
  std::size_t end = 0;
  for (const auto& provenance : segment.provenance) {
    start = std::min(start, provenance.graph_start);
    end = std::max(end, provenance.graph_end);
  }
  if (start == std::numeric_limits<std::size_t>::max()) {
    return nullptr;
  }
  return model_fragment_for_public_range(plan, start, end);
}

json source_to_json(const std::shared_ptr<Node>& node) {
  if (!node) {
    return nullptr;
  }
  if (auto* input = dynamic_cast<Input*>(node.get())) {
    json j;
    j["kind"] = "app_push";
    j["uri"] = nullptr;
    j["endpoint"] = input->endpoint_name().empty() ? json(nullptr) : json(input->endpoint_name());
    j["details"] = {
        {"memory_policy", static_cast<int>(input->options().memory_policy)},
        {"payload_type", static_cast<int>(input->options().payload_type)},
    };
    return j;
  }
  if (auto* file = dynamic_cast<FileInput*>(node.get())) {
    json j;
    j["kind"] = "file";
    j["uri"] = std::string("file://") + file->path();
    j["path"] = file->path();
    return j;
  }
  if (auto* still = dynamic_cast<StillImageInput*>(node.get())) {
    json j;
    j["kind"] = "still_image";
    j["uri"] = std::string("file://") + still->image_path();
    j["path"] = still->image_path();
    j["details"] = {
        {"content_width", still->content_w()},
        {"content_height", still->content_h()},
        {"encoded_width", still->enc_w()},
        {"encoded_height", still->enc_h()},
        {"fps", still->fps()},
    };
    return j;
  }
  if (auto* rtsp = dynamic_cast<RTSPInput*>(node.get())) {
    json j;
    j["kind"] = "rtsp";
    j["uri"] = rtsp->url();
    j["details"] = {
        {"latency_ms", rtsp->latency_ms()},
        {"tcp", rtsp->tcp()},
        {"drop_on_latency", rtsp->drop_on_latency()},
        {"buffer_mode", rtsp->buffer_mode()},
    };
    return j;
  }
  return nullptr;
}

json sink_to_json(const std::shared_ptr<Node>& node) {
  if (!node) {
    return nullptr;
  }
  if (auto* output = dynamic_cast<Output*>(node.get())) {
    json j;
    j["kind"] = "appsink";
    j["uri"] = nullptr;
    j["endpoint"] = output->endpoint_name().empty() ? json(nullptr) : json(output->endpoint_name());
    j["details"] = {
        {"max_buffers", output->options().max_buffers},
        {"drop", output->options().drop},
        {"sync", output->options().sync},
    };
    return j;
  }
  if (auto* udp = dynamic_cast<UdpOutput*>(node.get())) {
    const UdpOutputOptions& opt = udp->options();
    json j;
    j["kind"] = "udp";
    j["uri"] = "udp://" + opt.host + ":" + std::to_string(opt.port);
    j["details"] = {
        {"host", opt.host},
        {"port", opt.port},
        {"sync", opt.sync},
        {"async", opt.async},
    };
    return j;
  }
  return nullptr;
}

void attach_node_identity_blocks(json& n, const std::shared_ptr<Node>& node) {
  if (json source = source_to_json(node); !source.is_null()) {
    n["source"] = std::move(source);
  }
  if (json sink = sink_to_json(node); !sink.is_null()) {
    n["sink"] = std::move(sink);
  }
  if (const auto* lineage = model_lineage_for_node(node)) {
    n["model"] = model_lineage_to_json(*lineage);
  }
}

json public_view_to_json(const runtime::ExecutionGraphPlan& plan) {
  json view;
  json nodes = json::array();
  for (const auto& node : plan.public_nodes) {
    json n;
    n["id"] = "p" + std::to_string(node.id);
    n["index"] = node.id;
    n["kind"] = node.kind;
    n["label"] = node.label;
    n["endpoint_name"] = node.endpoint_name.empty() ? json(nullptr) : json(node.endpoint_name);
    n["input_endpoint"] = node.input_endpoint;
    n["output_endpoint"] = node.output_endpoint;
    n["runtime_node"] = node_ref(node.runtime_node);
    attach_node_identity_blocks(n, node.public_node);
    if (n.find("model") == n.end()) {
      if (const auto* fragment = model_fragment_for_public_index(plan, node.id)) {
        n["model"] = model_provenance_to_json(fragment->provenance);
      }
    }
    nodes.push_back(std::move(n));
  }
  view["nodes"] = std::move(nodes);

  json edges = json::array();
  for (const auto& edge : plan.public_edges) {
    json e;
    e["id"] = "pe" + std::to_string(edge.id);
    e["index"] = edge.id;
    e["from"] = "p" + std::to_string(edge.from);
    e["to"] = "p" + std::to_string(edge.to);
    e["kind"] = edge.kind;
    e["from_endpoint"] = edge.from_endpoint.empty() ? json(nullptr) : json(edge.from_endpoint);
    e["to_endpoint"] = edge.to_endpoint.empty() ? json(nullptr) : json(edge.to_endpoint);
    e["runtime_from"] = node_ref(edge.runtime_from);
    e["runtime_to"] = node_ref(edge.runtime_to);
    json runtime_edges = json::array();
    for (const std::size_t runtime_edge : edge.runtime_edge_indices) {
      runtime_edges.push_back("e" + std::to_string(runtime_edge));
    }
    e["runtime_edges"] = std::move(runtime_edges);
    edges.push_back(std::move(e));
  }
  view["edges"] = std::move(edges);
  return view;
}

json graph_topology_to_json(const runtime::RunCore& core) {
  json graph;
  if (!core.graph_execution_) {
    graph["mode"] = "linear";
    graph["nodes"] = json::array();
    graph["edges"] = json::array();
    graph["named_inputs"] = json::array();
    graph["named_outputs"] = json::array();
    return graph;
  }

  const runtime::ExecutionGraphRuntime& execution = core.graph_execution();
  const runtime::ExecutionGraphPlan& plan = execution.plan;
  graph["mode"] = plan.linear_compat ? "linear" : "connected";
  graph["public_graph_id"] = plan.public_graph_id;
  graph["public_graph_version"] = plan.public_graph_version;
  graph["named_inputs"] = named_endpoints_to_json(plan, plan.named_inputs);
  graph["named_outputs"] = named_endpoints_to_json(plan, plan.named_outputs);
  graph["public_view"] = public_view_to_json(plan);

  json nodes = json::array();
  for (const auto& segment : plan.pipeline_segments) {
    const runtime::FragmentPlan* segment_model = model_fragment_for_segment(plan, segment);
    for (std::size_t local = 0; local < segment.node_ids.size(); ++local) {
      const graph::NodeId id = segment.node_ids[local];
      json n;
      n["id"] = node_ref(id);
      n["stable_id"] = "segment_" + std::to_string(segment.id) + ".n" +
                       std::to_string(static_cast<std::size_t>(id));
      n["backend"] = "pipeline";
      n["segment"] = segment.id;
      n["segment_local_index"] = local;
      if (id < plan.node_labels.size()) {
        n["label"] = plan.node_labels[id];
      }
      if (local < segment.nodes.size() && segment.nodes[local]) {
        n["kind"] = segment.nodes[local]->kind();
        n["user_label"] = segment.nodes[local]->user_label();
        attach_node_identity_blocks(n, segment.nodes[local]);
      } else {
        n["kind"] = "PipelineNode";
      }
      if (segment_model != nullptr && n.find("model") == n.end()) {
        n["model"] = model_provenance_to_json(segment_model->provenance);
      }
      n["compiler_generated"] = false;
      nodes.push_back(std::move(n));
    }
  }
  for (const auto& stage : plan.stage_nodes) {
    json n;
    n["id"] = node_ref(stage.node_id);
    n["stable_id"] = "stage.n" + std::to_string(static_cast<std::size_t>(stage.node_id));
    n["backend"] = "stage";
    n["kind"] = stage.node ? stage.node->kind() : std::string("StageNode");
    n["label"] = stage.node_id < plan.node_labels.size() ? plan.node_labels[stage.node_id] : "";
    n["user_label"] = stage.node ? stage.node->user_label() : "";
    n["compiler_generated"] = compiler_generated_runtime_node(n["kind"].get<std::string>(),
                                                              n["label"].get<std::string>());
    nodes.push_back(std::move(n));
  }
  graph["nodes"] = std::move(nodes);

  json edges = json::array();
  for (std::size_t i = 0; i < plan.edges.size(); ++i) {
    const auto& edge = plan.edges[i];
    json e;
    e["id"] = "e" + std::to_string(i);
    e["from"] = node_ref(edge.from);
    e["to"] = node_ref(edge.to);
    const std::string from_port = port_name(plan, edge.from_port);
    const std::string to_port = port_name(plan, edge.to_port);
    e["from_port"] = from_port.empty() ? json(nullptr) : json(from_port);
    e["to_port"] = to_port.empty() ? json(nullptr) : json(to_port);
    e["spec_complete"] = edge.spec_complete;
    e["spec"] = output_spec_to_json(edge.spec);
    edges.push_back(std::move(e));
  }
  if (edges.empty() && plan.named_inputs.size() == 1U && plan.named_outputs.size() == 1U) {
    const auto& input = *plan.named_inputs.begin();
    const auto& output = *plan.named_outputs.begin();
    json e;
    e["id"] = "pe0";
    e["kind"] = "public_endpoint";
    e["from"] = node_ref(input.second.node);
    e["to"] = node_ref(output.second.node);
    e["from_port"] = input.first;
    e["to_port"] = output.first;
    e["spec_complete"] = false;
    edges.push_back(std::move(e));
  }
  graph["edges"] = std::move(edges);

  json segments = json::array();
  for (std::size_t i = 0; i < execution.pipelines.size(); ++i) {
    const auto& pipeline = execution.pipelines[i];
    if (!pipeline) {
      continue;
    }
    json s;
    s["index"] = i;
    s["id"] = pipeline->seg.id;
    s["built"] = pipeline->transport.built.load();
    s["has_input"] = pipeline->transport.has_input;
    s["has_output"] = pipeline->transport.has_output;
    s["last_pipeline"] = pipeline->last_pipeline;
    segments.push_back(std::move(s));
  }
  graph["pipeline_segments"] = std::move(segments);

  graph["lowered_view"] = {
      {"nodes", graph["nodes"]},
      {"edges", graph["edges"]},
      {"pipeline_segments", graph["pipeline_segments"]},
  };
  return graph;
}

json run_metrics_to_json(const Run& run, const runtime::RunCore& core,
                         const RunExportOptions& opt) {
  json j;
  j["identity"] = {
      {"uuid", core.run_id},
      {"created_at", iso8601_utc(core.created_wall_at).empty()
                         ? json(nullptr)
                         : json(iso8601_utc(core.created_wall_at))},
      {"closed_at", iso8601_utc(core.closed_wall_at).empty()
                        ? json(nullptr)
                        : json(iso8601_utc(core.closed_wall_at))},
      {"hostname", hostname_safe()},
      {"pid", pid_safe()},
      {"argv", argv_safe()},
  };
  j["stats_enabled"] = true;
  j["executed"] = core.outputs_pulled.load(std::memory_order_relaxed) > 0 ||
                  core.inputs_pushed.load(std::memory_order_relaxed) > 0;
  const RunStats stats = run.stats();
  j["stats"] = {
      {"inputs_enqueued", stats.inputs_enqueued},
      {"inputs_dropped", stats.inputs_dropped},
      {"inputs_pushed", stats.inputs_pushed},
      {"outputs_ready", stats.outputs_ready},
      {"outputs_pulled", stats.outputs_pulled},
      {"outputs_dropped", stats.outputs_dropped},
      {"latency_ms",
       {{"avg", stats.avg_latency_ms},
        {"min", stats.min_latency_ms},
        {"max", stats.max_latency_ms}}},
  };
  const RuntimeMetrics metrics =
      run.metrics(RuntimeMetricsOptions{.include_power = opt.include_power});
  j["elapsed_seconds"] = metrics.elapsed_seconds;
  j["throughput_fps"] = metrics.throughput_fps;
  j["input_names"] = core.input_names();
  j["output_names"] = core.output_names();
  j["last_error"] = core.last_error();
  if (opt.include_power && metrics.power.enabled) {
    json p;
    p["enabled"] = metrics.power.enabled;
    p["samples"] = metrics.power.samples;
    p["duration_seconds"] = metrics.power.duration_seconds;
    p["energy_joules"] = metrics.power.energy_joules;
    p["total_watts"] = {{"avg", metrics.power.total_avg_watts},
                        {"min", metrics.power.total_min_watts},
                        {"max", metrics.power.total_max_watts}};
    j["power"] = std::move(p);
  }

  const GraphMetricsReport graph_report = build_graph_metrics_report_run_lifetime(
      run, RuntimeMetricsOptions{.include_power = opt.include_power});
  json graph_metrics;
  graph_metrics["measurement_scope"] = graph_report.aggregation;
  graph_metrics["aggregation"] = graph_report.aggregation;
  graph_metrics["latency_semantics"] = graph_report.latency_semantics;
  graph_metrics["throughput_counting"] = graph_report.throughput_counting;
  graph_metrics["elapsed_seconds"] = graph_report.graph_metrics.elapsed_seconds;
  graph_metrics["outputs_pulled"] = graph_report.graph_metrics.counters.outputs_pulled;
  graph_metrics["throughput_fps"] = graph_report.graph_metrics.throughput_fps;
  graph_metrics["counters"] = {
      {"inputs_enqueued", graph_report.graph_metrics.counters.inputs_enqueued},
      {"inputs_dropped", graph_report.graph_metrics.counters.inputs_dropped},
      {"inputs_pushed", graph_report.graph_metrics.counters.inputs_pushed},
      {"outputs_ready", graph_report.graph_metrics.counters.outputs_ready},
      {"outputs_pulled", graph_report.graph_metrics.counters.outputs_pulled},
      {"outputs_dropped", graph_report.graph_metrics.counters.outputs_dropped},
  };
  if (opt.include_power && graph_report.graph_metrics.power.enabled) {
    graph_metrics["power"] = power_summary_json(graph_report.graph_metrics.power);
  }
  j["graph_metrics"] = std::move(graph_metrics);

  if (opt.include_node_metrics) {
    j["node_metrics"] =
        node_metrics_to_json(graph_report.node_metrics, graph_report.aggregation,
                             graph_report.latency_semantics, opt.include_empty_node_metrics,
                             opt.include_plugin_metrics);
  }
  if (opt.include_plugin_metrics) {
    j["plugin_metrics_unattributed"] = json::array();
  }
  return j;
}

bool json_nullable_int_equals(const json& value, std::int32_t expected) {
  if (expected < 0) {
    return false;
  }
  if (value.is_number_unsigned()) {
    return value.get<std::uint64_t>() == static_cast<std::uint64_t>(expected);
  }
  if (value.is_number_integer()) {
    return value.get<std::int64_t>() == static_cast<std::int64_t>(expected);
  }
  return false;
}

bool json_array_contains_string(const json& value, const std::string& needle) {
  if (!value.is_array()) {
    return false;
  }
  return std::any_of(value.begin(), value.end(), [&](const json& item) {
    return item.is_string() && item.get<std::string>() == needle;
  });
}

bool node_metric_segment_matches(const json& node, std::int32_t segment_id) {
  return segment_id < 0 || json_nullable_int_equals(node.value("pipeline_segment_id", json(nullptr)),
                                                    segment_id);
}

bool node_metric_has_element(const json& node, const std::string& element_name) {
  if (element_name.empty() || !node.contains("element_names") ||
      !node.at("element_names").is_array()) {
    return false;
  }
  return json_array_contains_string(node.at("element_names"), element_name);
}

std::optional<std::size_t> unique_node_match_from_indices(const std::vector<std::size_t>& matches,
                                                          const std::string& ambiguity_error,
                                                          const std::string& missing_error,
                                                          std::string* mapping_error) {
  if (matches.size() == 1U) {
    return matches.front();
  }
  if (mapping_error) {
    *mapping_error = matches.empty() ? missing_error : ambiguity_error;
  }
  return std::nullopt;
}

std::string plugin_preferred_element_name(const MeasurePluginLatency& plugin) {
  return !plugin.gst_element_name.empty() ? plugin.gst_element_name : plugin.stage_name;
}

std::optional<std::size_t> find_plugin_node_metric(const json& node_metrics,
                                                   const MeasurePluginLatency& plugin,
                                                   std::string* mapping_error) {
  if (!node_metrics.is_array() || node_metrics.empty()) {
    if (mapping_error) {
      *mapping_error = "unattributed: node metrics unavailable";
    }
    return std::nullopt;
  }

  if (plugin.runtime_node_id >= 0) {
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < node_metrics.size(); ++i) {
      const json& node = node_metrics.at(i);
      if (json_nullable_int_equals(node.value("runtime_node_id", json(nullptr)),
                                   plugin.runtime_node_id) &&
          node_metric_segment_matches(node, plugin.pipeline_segment_id)) {
        matches.push_back(i);
      }
    }
    return unique_node_match_from_indices(
        matches, "unattributed: profiler runtime_node_id matched multiple node metrics",
        "unattributed: profiler runtime_node_id did not match a node metric", mapping_error);
  }

  if (plugin.public_node_id >= 0) {
    const std::string public_id = "p" + std::to_string(plugin.public_node_id);
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < node_metrics.size(); ++i) {
      const json& node = node_metrics.at(i);
      if (json_array_contains_string(node.value("public_node_ids", json::array()), public_id)) {
        matches.push_back(i);
      }
    }
    return unique_node_match_from_indices(
        matches, "unattributed: profiler public_node_id matched multiple node metrics",
        "unattributed: profiler public_node_id did not match a node metric", mapping_error);
  }

  if (!plugin.public_node_ids.empty()) {
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < node_metrics.size(); ++i) {
      const json& node = node_metrics.at(i);
      for (const std::string& public_id : plugin.public_node_ids) {
        if (json_array_contains_string(node.value("public_node_ids", json::array()), public_id)) {
          matches.push_back(i);
          break;
        }
      }
    }
    return unique_node_match_from_indices(
        matches, "unattributed: profiler public_node_ids matched multiple node metrics",
        "unattributed: profiler public_node_ids did not match a node metric", mapping_error);
  }

  const std::string element_name = plugin_preferred_element_name(plugin);
  if (element_name.empty()) {
    if (mapping_error) {
      *mapping_error = "unattributed: profiler event has no stage_name or gst_element_name";
    }
    return std::nullopt;
  }

  if (plugin.pipeline_segment_id >= 0) {
    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < node_metrics.size(); ++i) {
      const json& node = node_metrics.at(i);
      if (node_metric_segment_matches(node, plugin.pipeline_segment_id) &&
          node_metric_has_element(node, element_name)) {
        matches.push_back(i);
      }
    }
    return unique_node_match_from_indices(
        matches, "unattributed: profiler element matched multiple node metrics in segment",
        "unattributed: profiler element did not match a graph element in segment", mapping_error);
  }

  std::vector<std::size_t> matches;
  for (std::size_t i = 0; i < node_metrics.size(); ++i) {
    if (node_metric_has_element(node_metrics.at(i), element_name)) {
      matches.push_back(i);
    }
  }
  return unique_node_match_from_indices(
      matches, "unattributed: profiler element matched multiple node metrics",
      "unattributed: profiler element did not match a graph element", mapping_error);
}

json merge_public_node_ids(json plugin_ids, const json& node_ids) {
  if (!plugin_ids.is_array()) {
    plugin_ids = json::array();
  }
  if (!node_ids.is_array()) {
    return plugin_ids;
  }
  for (const json& node_id : node_ids) {
    if (node_id.is_string() && !json_array_contains_string(plugin_ids, node_id.get<std::string>())) {
      plugin_ids.push_back(node_id);
    }
  }
  return plugin_ids;
}

void inherit_plugin_node_identity(json& plugin, const json& node) {
  if (plugin.value("pipeline_segment_id", json(nullptr)).is_null()) {
    plugin["pipeline_segment_id"] = node.value("pipeline_segment_id", json(nullptr));
  }
  if (plugin.value("runtime_node_id", json(nullptr)).is_null()) {
    plugin["runtime_node_id"] = node.value("runtime_node_id", json(nullptr));
  }
  plugin["public_node_ids"] =
      merge_public_node_ids(plugin.value("public_node_ids", json::array()),
                            node.value("public_node_ids", json::array()));
}

} // namespace

std::string run_to_json(const Run& run, const RunExportOptions& opt, std::string* err) {
  try {
    const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
    if (!core) {
      if (err) {
        *err = "run_to_json: Run has no runtime core";
      }
      return {};
    }

    json root;
    root["schema"] = "sima.neat.graph_run";
    root["schema_version"] = 1;
    root["producer"] = {
        {"name", "neat"},
#ifdef SIMANEAT_VERSION_STRING
        {"version", SIMANEAT_VERSION_STRING},
#else
        {"version", "unknown"},
#endif
#ifdef SIMANEAT_GIT_SHA
        {"git", SIMANEAT_GIT_SHA},
#else
        {"git", "unknown"},
#endif
    };
    root["label"] = opt.label.empty() ? "run" : opt.label;
    json metadata = json::object();
    for (const auto& [key, value] : opt.metadata) {
      metadata[key] = value;
    }
    root["metadata"] = std::move(metadata);
    root["graph"] = graph_topology_to_json(*core);
    root["run"] = opt.include_metrics ? run_metrics_to_json(run, *core, opt) : json::object();
    return root.dump(opt.indent);
  } catch (const std::exception& e) {
    if (err) {
      *err = std::string("run_to_json failed: ") + e.what();
    }
    return {};
  }
}

std::string run_to_json(const Run& run, const MeasureReport& report, const RunExportOptions& opt,
                        std::string* err) {
  std::string base_err;
  const std::string base = run_to_json(run, opt, &base_err);
  if (base.empty()) {
    if (err) {
      *err = base_err;
    }
    return {};
  }

  try {
    json root = json::parse(base);
    json& run_json = root["run"];
    run_json["graph_metrics"] = {
        {"measurement_scope", "measured_window"},
        {"aggregation", "measured_window"},
        {"latency_semantics", "sum_element_residency_delta"},
        {"throughput_counting", "all_pulled_outputs"},
        {"elapsed_seconds", report.elapsed_s},
        {"outputs_pulled", report.outputs_pulled},
        {"throughput_fps", report.throughput_batches_per_s},
        {"counters",
         {{"inputs_enqueued", report.inputs_pushed},
          {"inputs_dropped", report.inputs_dropped},
          {"inputs_pushed", report.inputs_pushed},
          {"outputs_ready", report.outputs_pulled},
          {"outputs_pulled", report.outputs_pulled},
          {"outputs_dropped", report.outputs_dropped}}},
    };
    if (opt.include_power && report.power.enabled) {
      run_json["graph_metrics"]["power"] = power_summary_json(report.power);
    }
    if (opt.include_node_metrics && !report.node_metrics.empty()) {
      run_json["node_metrics"] =
          node_metrics_to_json(report.node_metrics, "measured_window",
                               "sum_element_residency_delta",
                               opt.include_empty_node_metrics,
                               opt.include_plugin_metrics);
    } else if (!opt.include_node_metrics) {
      run_json.erase("node_metrics");
    }

    if (opt.include_plugin_metrics) {
      json plugins = json::array();
      for (const MeasurePluginLatency& p : report.plugin_latency) {
        std::string mapping_error;
        const std::optional<std::size_t> node_index =
            run_json.contains("node_metrics") && run_json.at("node_metrics").is_array()
                ? find_plugin_node_metric(run_json.at("node_metrics"), p, &mapping_error)
                : std::nullopt;
        if (node_index.has_value()) {
          json plugin = plugin_metric_to_json(p);
          json& node = run_json["node_metrics"].at(*node_index);
          inherit_plugin_node_identity(plugin, node);
          if (!node.contains("plugins") || !node["plugins"].is_array()) {
            node["plugins"] = json::array();
          }
          node["plugins"].push_back(std::move(plugin));
        } else {
          if (mapping_error.empty() && !run_json.contains("node_metrics")) {
            mapping_error = "unattributed: node metrics excluded from export";
          }
          plugins.push_back(plugin_metric_to_json(p, mapping_error));
        }
      }
      run_json["plugin_metrics_unattributed"] = std::move(plugins);
    } else {
      run_json.erase("plugin_metrics_unattributed");
    }
    run_json["measurement"] = {
        {"warmup_iterations", report.warmup_iterations},
        {"outputs", report.outputs},
        {"throughput_batches_per_s", report.throughput_batches_per_s},
        {"throughput_inferences_per_s", report.throughput_inferences_per_s},
        {"end_to_end",
         {{"count", report.end_to_end.count},
          {"avg_ms", report.end_to_end.avg_ms},
          {"p50_ms", report.end_to_end.p50_ms},
          {"p90_ms", report.end_to_end.p90_ms},
          {"p95_ms", report.end_to_end.p95_ms},
          {"p99_ms", report.end_to_end.p99_ms},
          {"max_ms", report.end_to_end.max_ms}}},
        {"frame_gap",
         {{"count", report.frame_gap.count},
          {"avg_ms", report.frame_gap.avg_ms},
          {"p50_ms", report.frame_gap.p50_ms},
          {"p90_ms", report.frame_gap.p90_ms},
          {"p95_ms", report.frame_gap.p95_ms},
          {"p99_ms", report.frame_gap.p99_ms},
          {"max_ms", report.frame_gap.max_ms}}},
    };
    return root.dump(opt.indent);
  } catch (const std::exception& e) {
    if (err) {
      *err = std::string("run_to_json measured report failed: ") + e.what();
    }
    return {};
  }
}

bool save_run_json(const Run& run, const std::string& path, const RunExportOptions& opt,
                   std::string* err) {
  const std::string body = run_to_json(run, opt, err);
  if (body.empty()) {
    return false;
  }

  const std::filesystem::path target(path);
  const std::filesystem::path tmp = target.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      if (err) {
        *err = "save_run_json: cannot open " + tmp.string();
      }
      return false;
    }
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!out) {
      if (err) {
        *err = "save_run_json: write failed for " + tmp.string();
      }
      return false;
    }
  }

  std::error_code ec;
  std::filesystem::rename(tmp, target, ec);
  if (ec) {
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(tmp, target, ec);
  }
  if (ec) {
    if (err) {
      *err = "save_run_json: rename failed: " + ec.message();
    }
    return false;
  }
  return true;
}

bool save_run_json(const Run& run, const MeasureReport& report, const std::string& path,
                   const RunExportOptions& opt, std::string* err) {
  const std::string body = run_to_json(run, report, opt, err);
  if (body.empty()) {
    return false;
  }

  const std::filesystem::path target(path);
  const std::filesystem::path tmp = target.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      if (err) {
        *err = "save_run_json: cannot open " + tmp.string();
      }
      return false;
    }
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!out) {
      if (err) {
        *err = "save_run_json: write failed for " + tmp.string();
      }
      return false;
    }
  }

  std::error_code ec;
  std::filesystem::rename(tmp, target, ec);
  if (ec) {
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(tmp, target, ec);
  }
  if (ec) {
    if (err) {
      *err = "save_run_json: rename failed: " + ec.message();
    }
    return false;
  }
  return true;
}

} // namespace simaai::neat

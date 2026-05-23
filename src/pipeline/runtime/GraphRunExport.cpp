#include "pipeline/GraphRunExport.h"

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
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/PipelineSegmentRuntime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>

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
                         const GraphRunExportOptions& opt) {
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
  return j;
}

} // namespace

std::string graph_run_to_json(const Run& run, const GraphRunExportOptions& opt, std::string* err) {
  try {
    const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
    if (!core) {
      if (err) {
        *err = "graph_run_to_json: Run has no runtime core";
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
      *err = std::string("graph_run_to_json failed: ") + e.what();
    }
    return {};
  }
}

bool save_graph_run_json(const Run& run, const std::string& path, const GraphRunExportOptions& opt,
                         std::string* err) {
  const std::string body = graph_run_to_json(run, opt, err);
  if (body.empty()) {
    return false;
  }

  const std::filesystem::path target(path);
  const std::filesystem::path tmp = target.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      if (err) {
        *err = "save_graph_run_json: cannot open " + tmp.string();
      }
      return false;
    }
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!out) {
      if (err) {
        *err = "save_graph_run_json: write failed for " + tmp.string();
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
      *err = "save_graph_run_json: rename failed: " + ec.message();
    }
    return false;
  }
  return true;
}

} // namespace simaai::neat

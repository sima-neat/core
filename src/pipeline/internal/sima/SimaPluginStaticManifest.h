#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

enum class QuantGranularity : std::uint8_t {
  PerTensor = 0,
  PerAxis = 1,
};

struct QuantStaticSpec {
  QuantGranularity granularity = QuantGranularity::PerTensor;
  int axis = -1;
  std::vector<double> scales;
  std::vector<std::int64_t> zero_points;
};

struct TensorStaticSpec {
  int tensor_index = -1;
  std::vector<std::int64_t> shape;
  std::string dtype;
  std::string layout;
  int max_w = 0;
  int max_h = 0;
  int max_stride = 0;
  std::string semantic_tag;
};

struct ResolutionTrace {
  std::string field;
  std::string source_used;
  std::vector<std::string> fallback_chain;
  bool conflict = false;
};

struct StageSinkRoute {
  int sink_pad_index = -1;
  bool required = true;
  std::string src_stage_id;
  int src_output_slot = 0;
  int tensor_index = -1;
  std::string cm_input_name;
  std::string source_segment_name;
};

struct StageOutputRoute {
  int output_slot = 0;
  std::string cm_output_name;
  std::string segment_name;
};

struct StageStaticSpec {
  std::string element_name;
  std::string logical_stage_id;
  std::string plugin_kind;
  std::string kernel_kind;
  std::vector<TensorStaticSpec> inputs;
  std::vector<TensorStaticSpec> outputs;
  std::vector<int> sink_pad_tensor_index_map;
  std::vector<StageSinkRoute> sink_routes;
  std::vector<StageOutputRoute> output_order;
  std::vector<QuantStaticSpec> output_quant;
  nlohmann::json runtime_defaults = nlohmann::json::object();
  std::vector<ResolutionTrace> resolution_trace;
};

struct SimaPluginStaticManifest {
  int manifest_version = 1;
  std::string session_id;
  std::string model_id;
  std::vector<StageStaticSpec> stages;
};

struct PipelineElementSpec {
  std::size_t element_index = 0;
  std::string plugin;
  std::string element_name;
  std::string stage_id;
  std::string config_path;
  std::optional<std::string> decode_type_property;
  std::optional<double> detection_threshold_property;
  std::optional<double> nms_iou_threshold_property;
  std::optional<int> topk_property;
  std::string fragment;
};

struct ManifestBuildDiagnostics {
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

std::vector<PipelineElementSpec> parse_pipeline_elements(const std::string& pipeline_string);

nlohmann::json to_json(const QuantStaticSpec& spec);
nlohmann::json to_json(const TensorStaticSpec& spec);
nlohmann::json to_json(const ResolutionTrace& trace);
nlohmann::json to_json(const StageSinkRoute& route);
nlohmann::json to_json(const StageOutputRoute& route);
nlohmann::json to_json(const StageStaticSpec& spec);
nlohmann::json to_json(const SimaPluginStaticManifest& manifest);

std::string serialize_manifest_json(const SimaPluginStaticManifest& manifest);

std::optional<SimaPluginStaticManifest> parse_manifest_json(const std::string& manifest_json,
                                                            std::string* error_message = nullptr);

bool attach_manifest_context(GstElement* pipeline, const SimaPluginStaticManifest& manifest,
                             std::string* error_message = nullptr);

bool extract_manifest_json_from_context(const GstContext* context, std::string& manifest_json_out,
                                        guint* manifest_version_out = nullptr);

} // namespace simaai::neat::pipeline_internal::sima

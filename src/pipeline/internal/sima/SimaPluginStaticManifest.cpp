#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include "gst/SimaPluginStaticManifestAbi.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim_copy_local(const std::string& s) {
  return trim_copy(s);
}

std::vector<std::string> split_pipeline_segments(const std::string& pipeline_string) {
  std::vector<std::string> out;
  if (pipeline_string.empty())
    return out;

  bool in_single_quote = false;
  bool in_double_quote = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i < pipeline_string.size(); ++i) {
    const char c = pipeline_string[i];
    if (c == '\'' && !in_double_quote) {
      in_single_quote = !in_single_quote;
      continue;
    }
    if (c == '"' && !in_single_quote) {
      in_double_quote = !in_double_quote;
      continue;
    }
    if (c == '!' && !in_single_quote && !in_double_quote) {
      const std::string seg = trim_copy_local(pipeline_string.substr(start, i - start));
      if (!seg.empty()) {
        out.push_back(seg);
      }
      start = i + 1;
    }
  }
  const std::string tail = trim_copy_local(pipeline_string.substr(start));
  if (!tail.empty()) {
    out.push_back(tail);
  }
  return out;
}

std::string parse_first_token(const std::string& seg) {
  std::size_t i = 0;
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  std::size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])))
    ++j;
  return (j > i) ? seg.substr(i, j - i) : std::string{};
}

std::optional<std::string> parse_property_value(const std::string& seg, const char* key) {
  if (!key || !*key)
    return std::nullopt;
  const std::string needle = std::string(key) + "=";
  std::size_t pos = seg.find(needle);
  if (pos == std::string::npos)
    return std::nullopt;
  std::size_t i = pos + needle.size();
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  if (i >= seg.size())
    return std::nullopt;

  if (seg[i] == '"') {
    const std::size_t j = seg.find('"', i + 1);
    if (j == std::string::npos || j <= i + 1)
      return std::nullopt;
    return seg.substr(i + 1, j - i - 1);
  }

  std::size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])) && seg[j] != '!')
    ++j;
  if (j <= i)
    return std::nullopt;
  return seg.substr(i, j - i);
}

std::optional<double> parse_double_property(const std::string& seg, const char* key) {
  const auto raw = parse_property_value(seg, key);
  if (!raw.has_value() || raw->empty())
    return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(raw->c_str(), &end);
  if (errno != 0 || end == raw->c_str() || (end && *end != '\0'))
    return std::nullopt;
  return value;
}

std::optional<int> parse_int_property(const std::string& seg, const char* key) {
  const auto raw = parse_property_value(seg, key);
  if (!raw.has_value() || raw->empty())
    return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(raw->c_str(), &end, 10);
  if (errno != 0 || end == raw->c_str() || (end && *end != '\0'))
    return std::nullopt;
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    return std::nullopt;
  return static_cast<int>(value);
}

const char* quant_granularity_name(QuantGranularity granularity) {
  switch (granularity) {
  case QuantGranularity::PerTensor:
    return "per_tensor";
  case QuantGranularity::PerAxis:
    return "per_axis";
  }
  return "per_tensor";
}

std::optional<QuantGranularity> quant_granularity_from_json(const nlohmann::json& j) {
  if (!j.is_string())
    return std::nullopt;
  const std::string name = lower_copy(j.get<std::string>());
  if (name == "per_tensor")
    return QuantGranularity::PerTensor;
  if (name == "per_axis")
    return QuantGranularity::PerAxis;
  return std::nullopt;
}

template <typename T> void read_number_vector(const nlohmann::json& j, std::vector<T>& out) {
  out.clear();
  if (j.is_array()) {
    for (const auto& item : j) {
      if (item.is_number_integer()) {
        out.push_back(static_cast<T>(item.get<std::int64_t>()));
      } else if (item.is_number()) {
        out.push_back(static_cast<T>(item.get<double>()));
      }
    }
    return;
  }
  if (j.is_number_integer()) {
    out.push_back(static_cast<T>(j.get<std::int64_t>()));
  } else if (j.is_number()) {
    out.push_back(static_cast<T>(j.get<double>()));
  }
}

template <typename T>
void read_number_vector_key(const nlohmann::json& obj, const char* key, std::vector<T>& out) {
  if (!obj.contains(key)) {
    out.clear();
    return;
  }
  read_number_vector(obj.at(key), out);
}

bool read_string_key(const nlohmann::json& obj, const char* key, std::string& out) {
  out.clear();
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_string()) {
    out = value.get<std::string>();
    return true;
  }
  return false;
}

bool read_int_key(const nlohmann::json& obj, const char* key, int& out) {
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_number_integer()) {
    out = value.get<int>();
    return true;
  }
  if (value.is_number()) {
    out = static_cast<int>(value.get<double>());
    return true;
  }
  return false;
}

bool read_bool_key(const nlohmann::json& obj, const char* key, bool& out) {
  if (!obj.contains(key))
    return false;
  const auto& value = obj.at(key);
  if (value.is_boolean()) {
    out = value.get<bool>();
    return true;
  }
  if (value.is_number_integer()) {
    out = value.get<int>() != 0;
    return true;
  }
  return false;
}

} // namespace

std::vector<PipelineElementSpec> parse_pipeline_elements(const std::string& pipeline_string) {
  std::vector<PipelineElementSpec> out;
  const std::vector<std::string> segments = split_pipeline_segments(pipeline_string);
  out.reserve(segments.size());

  for (std::size_t i = 0; i < segments.size(); ++i) {
    const std::string& seg = segments[i];
    const std::string plugin = parse_first_token(seg);
    if (plugin.empty())
      continue;

    PipelineElementSpec spec;
    spec.element_index = i;
    spec.plugin = lower_copy(plugin);
    spec.fragment = seg;
    if (const auto name = parse_property_value(seg, "name"); name.has_value())
      spec.element_name = *name;
    if (const auto stage_id = parse_property_value(seg, "stage-id"); stage_id.has_value()) {
      spec.stage_id = *stage_id;
    } else if (const auto stage_id_snake = parse_property_value(seg, "stage_id");
               stage_id_snake.has_value()) {
      spec.stage_id = *stage_id_snake;
    }
    if (const auto cfg = parse_property_value(seg, "config"); cfg.has_value())
      spec.config_path = *cfg;
    if (const auto decode = parse_property_value(seg, "decode-type"); decode.has_value()) {
      spec.decode_type_property = *decode;
    } else if (const auto decode_snake = parse_property_value(seg, "decode_type");
               decode_snake.has_value()) {
      spec.decode_type_property = *decode_snake;
    }
    if (const auto det = parse_double_property(seg, "detection-threshold"); det.has_value()) {
      spec.detection_threshold_property = *det;
    } else if (const auto det_snake = parse_double_property(seg, "detection_threshold");
               det_snake.has_value()) {
      spec.detection_threshold_property = *det_snake;
    }
    if (const auto nms = parse_double_property(seg, "nms-iou-threshold"); nms.has_value()) {
      spec.nms_iou_threshold_property = *nms;
    } else if (const auto nms_snake = parse_double_property(seg, "nms_iou_threshold");
               nms_snake.has_value()) {
      spec.nms_iou_threshold_property = *nms_snake;
    }
    if (const auto topk = parse_int_property(seg, "topk"); topk.has_value())
      spec.topk_property = *topk;

    out.push_back(std::move(spec));
  }

  return out;
}

nlohmann::json to_json(const QuantStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["granularity"] = quant_granularity_name(spec.granularity);
  j["axis"] = spec.axis;
  j["scales"] = spec.scales;
  j["zero_points"] = spec.zero_points;
  return j;
}

nlohmann::json to_json(const TensorStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["tensor_index"] = spec.tensor_index;
  j["shape"] = spec.shape;
  j["dtype"] = spec.dtype;
  j["layout"] = spec.layout;
  j["max_w"] = spec.max_w;
  j["max_h"] = spec.max_h;
  j["max_stride"] = spec.max_stride;
  j["semantic_tag"] = spec.semantic_tag;
  return j;
}

nlohmann::json to_json(const ResolutionTrace& trace) {
  nlohmann::json j = nlohmann::json::object();
  j["field"] = trace.field;
  j["source_used"] = trace.source_used;
  j["fallback_chain"] = trace.fallback_chain;
  j["conflict"] = trace.conflict;
  return j;
}

nlohmann::json to_json(const StageSinkRoute& route) {
  nlohmann::json j = nlohmann::json::object();
  j["sink_pad_index"] = route.sink_pad_index;
  j["required"] = route.required;
  j["src_stage_id"] = route.src_stage_id;
  j["src_output_slot"] = route.src_output_slot;
  j["tensor_index"] = route.tensor_index;
  j["cm_input_name"] = route.cm_input_name;
  j["source_segment_name"] = route.source_segment_name;
  return j;
}

nlohmann::json to_json(const StageOutputRoute& route) {
  nlohmann::json j = nlohmann::json::object();
  j["output_slot"] = route.output_slot;
  j["cm_output_name"] = route.cm_output_name;
  j["segment_name"] = route.segment_name;
  return j;
}

nlohmann::json to_json(const StageStaticSpec& spec) {
  nlohmann::json j = nlohmann::json::object();
  j["element_name"] = spec.element_name;
  j["logical_stage_id"] = spec.logical_stage_id;
  j["plugin_kind"] = spec.plugin_kind;
  j["kernel_kind"] = spec.kernel_kind;

  nlohmann::json in = nlohmann::json::array();
  for (const auto& tensor : spec.inputs)
    in.push_back(to_json(tensor));
  j["inputs"] = std::move(in);

  nlohmann::json out = nlohmann::json::array();
  for (const auto& tensor : spec.outputs)
    out.push_back(to_json(tensor));
  j["outputs"] = std::move(out);
  j["sink_pad_tensor_index_map"] = spec.sink_pad_tensor_index_map;

  nlohmann::json sink_routes = nlohmann::json::array();
  for (const auto& route : spec.sink_routes)
    sink_routes.push_back(to_json(route));
  j["sink_routes"] = std::move(sink_routes);

  nlohmann::json output_routes = nlohmann::json::array();
  for (const auto& route : spec.output_order)
    output_routes.push_back(to_json(route));
  j["output_order"] = std::move(output_routes);

  nlohmann::json q = nlohmann::json::array();
  for (const auto& quant : spec.output_quant)
    q.push_back(to_json(quant));
  j["output_quant"] = std::move(q);
  j["runtime_defaults"] = spec.runtime_defaults;

  nlohmann::json trace = nlohmann::json::array();
  for (const auto& t : spec.resolution_trace)
    trace.push_back(to_json(t));
  j["resolution_trace"] = std::move(trace);

  return j;
}

nlohmann::json to_json(const SimaPluginStaticManifest& manifest) {
  nlohmann::json j = nlohmann::json::object();
  j["manifest_version"] = manifest.manifest_version;
  j["session_id"] = manifest.session_id;
  j["model_id"] = manifest.model_id;
  j["stages"] = nlohmann::json::array();
  for (const auto& stage : manifest.stages) {
    j["stages"].push_back(to_json(stage));
  }
  return j;
}

std::string serialize_manifest_json(const SimaPluginStaticManifest& manifest) {
  return to_json(manifest).dump();
}

std::optional<SimaPluginStaticManifest> parse_manifest_json(const std::string& manifest_json,
                                                            std::string* error_message) {
  if (error_message)
    error_message->clear();

  nlohmann::json root;
  try {
    root = nlohmann::json::parse(manifest_json);
  } catch (const std::exception& e) {
    if (error_message) {
      *error_message = std::string("manifest parse error: ") + e.what();
    }
    return std::nullopt;
  }

  if (!root.is_object()) {
    if (error_message)
      *error_message = "manifest root must be a JSON object";
    return std::nullopt;
  }

  SimaPluginStaticManifest manifest;
  read_int_key(root, "manifest_version", manifest.manifest_version);
  read_string_key(root, "session_id", manifest.session_id);
  read_string_key(root, "model_id", manifest.model_id);

  if (!root.contains("stages") || !root["stages"].is_array()) {
    if (error_message)
      *error_message = "manifest missing stages[]";
    return std::nullopt;
  }

  for (const auto& stage_j : root["stages"]) {
    if (!stage_j.is_object())
      continue;
    StageStaticSpec stage;
    read_string_key(stage_j, "element_name", stage.element_name);
    read_string_key(stage_j, "logical_stage_id", stage.logical_stage_id);
    read_string_key(stage_j, "plugin_kind", stage.plugin_kind);
    read_string_key(stage_j, "kernel_kind", stage.kernel_kind);

    if (stage_j.contains("inputs") && stage_j["inputs"].is_array()) {
      for (const auto& in_j : stage_j["inputs"]) {
        if (!in_j.is_object())
          continue;
        TensorStaticSpec tensor;
        read_int_key(in_j, "tensor_index", tensor.tensor_index);
        read_number_vector_key<std::int64_t>(in_j, "shape", tensor.shape);
        read_string_key(in_j, "dtype", tensor.dtype);
        read_string_key(in_j, "layout", tensor.layout);
        read_int_key(in_j, "max_w", tensor.max_w);
        read_int_key(in_j, "max_h", tensor.max_h);
        read_int_key(in_j, "max_stride", tensor.max_stride);
        read_string_key(in_j, "semantic_tag", tensor.semantic_tag);
        stage.inputs.push_back(std::move(tensor));
      }
    }

    if (stage_j.contains("outputs") && stage_j["outputs"].is_array()) {
      for (const auto& out_j : stage_j["outputs"]) {
        if (!out_j.is_object())
          continue;
        TensorStaticSpec tensor;
        read_int_key(out_j, "tensor_index", tensor.tensor_index);
        read_number_vector_key<std::int64_t>(out_j, "shape", tensor.shape);
        read_string_key(out_j, "dtype", tensor.dtype);
        read_string_key(out_j, "layout", tensor.layout);
        read_int_key(out_j, "max_w", tensor.max_w);
        read_int_key(out_j, "max_h", tensor.max_h);
        read_int_key(out_j, "max_stride", tensor.max_stride);
        read_string_key(out_j, "semantic_tag", tensor.semantic_tag);
        stage.outputs.push_back(std::move(tensor));
      }
    }

    if (stage_j.contains("sink_pad_tensor_index_map") &&
        stage_j["sink_pad_tensor_index_map"].is_array()) {
      for (const auto& idx : stage_j["sink_pad_tensor_index_map"]) {
        if (idx.is_number_integer()) {
          stage.sink_pad_tensor_index_map.push_back(idx.get<int>());
        } else if (idx.is_number()) {
          stage.sink_pad_tensor_index_map.push_back(static_cast<int>(idx.get<double>()));
        }
      }
    }

    if (stage_j.contains("sink_routes") && stage_j["sink_routes"].is_array()) {
      for (const auto& route_j : stage_j["sink_routes"]) {
        if (!route_j.is_object())
          continue;
        StageSinkRoute route;
        read_int_key(route_j, "sink_pad_index", route.sink_pad_index);
        read_bool_key(route_j, "required", route.required);
        read_string_key(route_j, "src_stage_id", route.src_stage_id);
        read_int_key(route_j, "src_output_slot", route.src_output_slot);
        read_int_key(route_j, "tensor_index", route.tensor_index);
        read_string_key(route_j, "cm_input_name", route.cm_input_name);
        read_string_key(route_j, "source_segment_name", route.source_segment_name);
        stage.sink_routes.push_back(std::move(route));
      }
    }

    if (stage_j.contains("output_order") && stage_j["output_order"].is_array()) {
      for (const auto& route_j : stage_j["output_order"]) {
        if (!route_j.is_object())
          continue;
        StageOutputRoute route;
        read_int_key(route_j, "output_slot", route.output_slot);
        read_string_key(route_j, "cm_output_name", route.cm_output_name);
        read_string_key(route_j, "segment_name", route.segment_name);
        stage.output_order.push_back(std::move(route));
      }
    }

    if (stage_j.contains("output_quant") && stage_j["output_quant"].is_array()) {
      for (const auto& q_j : stage_j["output_quant"]) {
        if (!q_j.is_object())
          continue;
        QuantStaticSpec quant;
        if (q_j.contains("granularity")) {
          if (const auto g = quant_granularity_from_json(q_j["granularity"]); g.has_value()) {
            quant.granularity = *g;
          }
        }
        read_int_key(q_j, "axis", quant.axis);
        read_number_vector_key<double>(q_j, "scales", quant.scales);
        read_number_vector_key<std::int64_t>(q_j, "zero_points", quant.zero_points);
        stage.output_quant.push_back(std::move(quant));
      }
    }

    if (stage_j.contains("runtime_defaults") && stage_j["runtime_defaults"].is_object()) {
      stage.runtime_defaults = stage_j["runtime_defaults"];
    }

    if (stage_j.contains("resolution_trace") && stage_j["resolution_trace"].is_array()) {
      for (const auto& t_j : stage_j["resolution_trace"]) {
        if (!t_j.is_object())
          continue;
        ResolutionTrace trace;
        read_string_key(t_j, "field", trace.field);
        read_string_key(t_j, "source_used", trace.source_used);
        if (t_j.contains("fallback_chain") && t_j["fallback_chain"].is_array()) {
          for (const auto& f : t_j["fallback_chain"]) {
            if (f.is_string()) {
              trace.fallback_chain.push_back(f.get<std::string>());
            }
          }
        }
        read_bool_key(t_j, "conflict", trace.conflict);
        stage.resolution_trace.push_back(std::move(trace));
      }
    }

    manifest.stages.push_back(std::move(stage));
  }

  return manifest;
}

namespace {

class ManifestAccessorRegistry {
public:
  explicit ManifestAccessorRegistry(SimaPluginStaticManifest manifest)
      : manifest_(std::move(manifest)), manifest_json_(serialize_manifest_json(manifest_)),
        session_id_(manifest_.session_id), model_id_(manifest_.model_id) {
    accessor_.abi_version = SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION;
    accessor_.user_data = this;
    accessor_.manifest_version = &ManifestAccessorRegistry::manifest_version_cb;
    accessor_.manifest_json = &ManifestAccessorRegistry::manifest_json_cb;
    accessor_.session_id = &ManifestAccessorRegistry::session_id_cb;
    accessor_.model_id = &ManifestAccessorRegistry::model_id_cb;
    accessor_.stage_json_by_element_name = &ManifestAccessorRegistry::stage_json_by_element_cb;
    accessor_.stage_json_by_logical_stage_id = &ManifestAccessorRegistry::stage_json_by_id_cb;

    stage_json_by_element_.reserve(manifest_.stages.size());
    stage_json_by_id_.reserve(manifest_.stages.size());
    for (const auto& stage : manifest_.stages) {
      const std::string stage_json = to_json(stage).dump();
      if (!stage.element_name.empty()) {
        stage_json_by_element_.emplace(stage.element_name, stage_json);
      }
      if (!stage.logical_stage_id.empty()) {
        stage_json_by_id_.emplace(stage.logical_stage_id, stage_json);
      }
    }
  }

  const SimaPluginStaticManifestAccessorV1* accessor() const {
    return &accessor_;
  }

  const std::string& payload() const {
    return manifest_json_;
  }

private:
  static guint manifest_version_cb(gpointer user_data) {
    if (!user_data) {
      return 0;
    }
    return static_cast<guint>(
        static_cast<const ManifestAccessorRegistry*>(user_data)->manifest_.manifest_version);
  }

  static const gchar* manifest_json_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    return static_cast<const ManifestAccessorRegistry*>(user_data)->manifest_json_.c_str();
  }

  static const gchar* session_id_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    return self->session_id_.empty() ? nullptr : self->session_id_.c_str();
  }

  static const gchar* model_id_cb(gpointer user_data) {
    if (!user_data) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    return self->model_id_.empty() ? nullptr : self->model_id_.c_str();
  }

  static const gchar* stage_json_by_element_cb(gpointer user_data, const gchar* element_name) {
    if (!user_data || !element_name || !*element_name) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    const auto it = self->stage_json_by_element_.find(element_name);
    if (it == self->stage_json_by_element_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

  static const gchar* stage_json_by_id_cb(gpointer user_data, const gchar* logical_stage_id) {
    if (!user_data || !logical_stage_id || !*logical_stage_id) {
      return nullptr;
    }
    const auto* self = static_cast<const ManifestAccessorRegistry*>(user_data);
    const auto it = self->stage_json_by_id_.find(logical_stage_id);
    if (it == self->stage_json_by_id_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

  SimaPluginStaticManifest manifest_;
  std::string manifest_json_;
  std::string session_id_;
  std::string model_id_;
  SimaPluginStaticManifestAccessorV1 accessor_{};
  std::unordered_map<std::string, std::string> stage_json_by_element_;
  std::unordered_map<std::string, std::string> stage_json_by_id_;
};

std::mutex& manifest_owner_mutex() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<GstElement*, std::shared_ptr<ManifestAccessorRegistry>>& manifest_owner_map() {
  static std::unordered_map<GstElement*, std::shared_ptr<ManifestAccessorRegistry>> owners;
  return owners;
}

void pipeline_owner_weak_notify(gpointer, GObject* where_the_object_was) {
  if (!where_the_object_was) {
    return;
  }
  std::lock_guard<std::mutex> lock(manifest_owner_mutex());
  manifest_owner_map().erase(reinterpret_cast<GstElement*>(where_the_object_was));
}

void hold_manifest_owner_for_pipeline(GstElement* pipeline,
                                      std::shared_ptr<ManifestAccessorRegistry> owner) {
  if (!pipeline || !owner) {
    return;
  }
  std::lock_guard<std::mutex> lock(manifest_owner_mutex());
  const bool first_insert = (manifest_owner_map().find(pipeline) == manifest_owner_map().end());
  manifest_owner_map()[pipeline] = std::move(owner);
  if (first_insert) {
    g_object_weak_ref(G_OBJECT(pipeline), pipeline_owner_weak_notify, nullptr);
  }
}

} // namespace

bool attach_manifest_context(GstElement* pipeline, const SimaPluginStaticManifest& manifest,
                             std::string* error_message) {
  if (error_message)
    error_message->clear();
  if (!pipeline) {
    if (error_message)
      *error_message = "attach manifest context failed: pipeline is null";
    return false;
  }

  const std::string payload = serialize_manifest_json(manifest);
  GstContext* context = gst_context_new(SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE, TRUE);
  if (!context) {
    if (error_message)
      *error_message = "attach manifest context failed: gst_context_new returned null";
    return false;
  }

  const auto owner = std::make_shared<ManifestAccessorRegistry>(manifest);
  if (!owner) {
    if (error_message) {
      *error_message = "attach manifest context failed: unable to create manifest owner";
    }
    gst_context_unref(context);
    return false;
  }

  GstStructure* structure = gst_context_writable_structure(context);
  gst_structure_set(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_VERSION, G_TYPE_UINT,
                    static_cast<guint>(manifest.manifest_version),
                    SIMA_PLUGIN_STATIC_MANIFEST_KEY_JSON, G_TYPE_STRING, payload.c_str(),
                    SIMA_PLUGIN_STATIC_MANIFEST_KEY_ACCESSOR_V1, G_TYPE_POINTER, owner->accessor(),
                    nullptr);
  if (!manifest.session_id.empty()) {
    gst_structure_set(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_SESSION_ID, G_TYPE_STRING,
                      manifest.session_id.c_str(), nullptr);
  }
  if (!manifest.model_id.empty()) {
    gst_structure_set(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_MODEL_ID, G_TYPE_STRING,
                      manifest.model_id.c_str(), nullptr);
  }

  hold_manifest_owner_for_pipeline(pipeline, owner);
  gst_element_set_context(pipeline, context);
  gst_context_unref(context);
  return true;
}

bool extract_manifest_json_from_context(const GstContext* context, std::string& manifest_json_out,
                                        guint* manifest_version_out) {
  manifest_json_out.clear();
  if (manifest_version_out)
    *manifest_version_out = 0;
  if (!context || !sima_plugin_manifest_context_matches(context))
    return false;

  if (const auto* accessor = sima_plugin_manifest_context_accessor(context); accessor) {
    if (accessor->manifest_json) {
      const gchar* payload = accessor->manifest_json(accessor->user_data);
      if (payload && *payload) {
        manifest_json_out.assign(payload);
      }
    }
    if (manifest_version_out && accessor->manifest_version) {
      *manifest_version_out = accessor->manifest_version(accessor->user_data);
    }
    if (!manifest_json_out.empty()) {
      return true;
    }
  }

  const GstStructure* structure = gst_context_get_structure(context);
  if (!structure)
    return false;

  const gchar* payload = gst_structure_get_string(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_JSON);
  if (!payload || !*payload)
    return false;

  manifest_json_out.assign(payload);

  if (manifest_version_out) {
    guint version = 0;
    if (gst_structure_get_uint(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_VERSION, &version)) {
      *manifest_version_out = version;
    }
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima

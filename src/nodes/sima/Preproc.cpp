#include "nodes/sima/Preproc.h"

#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "gst/GstHelpers.h"
#include "builder/ConfigJsonWire.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TempJsonFileUtil.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
using pipeline_internal::upper_copy;

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Preproc::PreprocConfigHolder {
  std::string path;
  bool keep = false;
  json config;
  bool has_config = false;
  ~PreprocConfigHolder() {
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
};

namespace {

int channels_from_format(const std::string& fmt, int fallback) {
  if (!fmt.empty()) {
    const std::string up = upper_copy(fmt);
    if (up == "GRAY" || up == "GRAY8")
      return 1;
    if (up == "RGB" || up == "BGR" || up == "NV12" || up == "I420")
      return 3;
  }
  return (fallback > 0) ? fallback : 3;
}

std::vector<float> ensure_three(const std::vector<float>& v, float def_val) {
  if (v.empty())
    return {def_val, def_val, def_val};
  if (v.size() == 1)
    return {v[0], v[0], v[0]};
  if (v.size() >= 3)
    return {v[0], v[1], v[2]};
  std::vector<float> out = v;
  while (out.size() < 3)
    out.push_back(def_val);
  return out;
}

std::vector<std::string> output_memory_order_default(const PreprocOptions& opt) {
  if (!opt.output_memory_order.empty())
    return opt.output_memory_order;
  const std::string next = upper_copy(opt.next_cpu);
  if (next == "APU") {
    return {"output_rgb_image", "output_tessellated_image"};
  }
  return {"output_tessellated_image", "output_rgb_image"};
}

bool has_next_cpu_manual(const json& j) {
  if (!j.contains("simaai__params") || !j["simaai__params"].is_object())
    return false;
  const json& params = j["simaai__params"];
  if (!params.contains("next_cpu_manual"))
    return false;
  const json& val = params["next_cpu_manual"];
  if (val.is_boolean())
    return val.get<bool>();
  if (val.is_number_integer())
    return val.get<int>() != 0;
  if (val.is_string())
    return val.get<std::string>() != "0";
  return false;
}

std::string read_next_cpu_string(const json& j) {
  auto normalize = [](const json& v) -> std::string {
    if (v.is_number_integer()) {
      const int cpu = v.get<int>();
      if (cpu == 0)
        return "APU";
      if (cpu == 1)
        return "CVU";
      if (cpu == 2)
        return "MLA";
      return {};
    }
    if (v.is_string()) {
      const std::string up = upper_copy(v.get<std::string>());
      if (up == "APU" || up == "A65")
        return "APU";
      if (up == "CVU")
        return "CVU";
      if (up == "MLA")
        return "MLA";
    }
    return {};
  };

  if (j.contains("next_cpu")) {
    const std::string val = normalize(j["next_cpu"]);
    if (!val.empty())
      return val;
  }
  if (j.contains("simaai__params") && j["simaai__params"].is_object()) {
    const auto& params = j["simaai__params"];
    if (params.contains("next_cpu")) {
      return normalize(params["next_cpu"]);
    }
  }
  return {};
}

std::string find_preproc_config_path(const simaai::neat::internal::ModelPack& model) {
  std::string path = model.find_config_path_by_plugin("process_cvu");
  if (path.empty())
    path = model.find_config_path_by_plugin("preproc");
  if (path.empty())
    path = model.find_config_path_by_processor("CVU");
  return path;
}

std::string make_temp_json_path(const std::string& dir) {
  return pipeline_internal::make_temp_json_path(dir, "sima_preproc", "Preproc");
}

json load_json_file(const std::string& path, const char* label) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  json j;
  in >> j;
  return j;
}

void write_json_file(const json& j, const std::string& path, const char* label) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  out << j.dump(4);
}

std::string write_json_temp(const json& j, const std::string& dir) {
  const std::string path = make_temp_json_path(dir);
  write_json_file(j, path, "Preproc");
  return path;
}

//------------------------------------------------------------------------------
// Fast "dims-only" filter: treat changes to width/height as runtime-only.
// We compare configs after normalizing input_width/input_height to the old values.
// This avoids expensive deep-diff and avoids JSON writes for dims-only renegotiation.
//------------------------------------------------------------------------------
bool equal_ignoring_input_dims(const json& a, const json& b) {
  // Fast path: identical JSON object.
  if (a == b)
    return true;

  // Make shallow copies and normalize the two dim fields (only).
  // This is fast enough because configs are small; this runs only on override attempts.
  json aa = a;
  json bb = b;

  // If both have the fields, force them equal (copy from b->a then compare).
  // If one side doesn't have the field, erase it on both (treat missing == missing).
  const auto normalize_field = [&](const char* key) {
    const bool ha = aa.contains(key);
    const bool hb = bb.contains(key);
    if (!ha && !hb)
      return;
    if (ha && hb) {
      aa[key] = bb[key];
      return;
    }
    // One missing: ignore by erasing the present one.
    if (ha)
      aa.erase(key);
    if (hb)
      bb.erase(key);
  };

  normalize_field("input_width");
  normalize_field("input_height");

  return aa == bb;
}

} // namespace

PreprocOptions::PreprocOptions(const simaai::neat::Model& model) {
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  const std::string path = find_preproc_config_path(pack);
  if (path.empty()) {
    throw std::runtime_error("PreprocOptions: failed to locate preproc config");
  }
  config_json = load_json_file(path, "PreprocOptions");
  num_buffers_model = pack.num_buffers_cvu();
  num_buffers = num_buffers_model;
  num_buffers_locked = true;
  next_cpu.clear();
}

namespace {

json build_preproc_json(const PreprocOptions& opt, int in_w, int in_h, int out_w, int out_h,
                        int scaled_w, int scaled_h) {
  const std::string in_fmt = upper_copy(opt.input_img_type);
  const std::string out_fmt = upper_copy(opt.output_img_type);
  const std::vector<float> mean3 = ensure_three(opt.channel_mean, 0.0f);
  const std::vector<float> std3 = ensure_three(opt.channel_stddev, 1.0f);

  json j;
  j["graph_name"] = opt.graph_name;
  j["node_name"] = opt.node_name;
  if (!opt.cpu.empty()) {
    j["cpu"] = upper_copy(opt.cpu);
  }
  if (!opt.next_cpu.empty()) {
    j["next_cpu"] = upper_copy(opt.next_cpu);
  }

  j["input_buffers"] = json::array({
      {
          {"name", opt.upstream_name},
          {"memories", json::array({
                           {
                               {"segment_name", "parent"},
                               {"graph_input_name", opt.graph_input_name},
                           },
                       })},
      },
  });

  j["output_memory_order"] = output_memory_order_default(opt);
  j["debug"] = opt.debug;

  j["input_width"] = in_w;
  j["input_height"] = in_h;
  j["input_offset"] = opt.input_offset;

  j["output_width"] = out_w;
  j["output_height"] = out_h;
  j["scaled_width"] = scaled_w;
  j["scaled_height"] = scaled_h;

  j["batch_size"] = opt.batch_size;
  j["normalize"] = opt.normalize;
  j["aspect_ratio"] = opt.aspect_ratio;
  j["dynamic_input_dims"] = opt.dynamic_input_dims;

  j["tile_width"] = opt.tile_width;
  j["tile_height"] = opt.tile_height;
  j["tile_channels"] = opt.tile_channels;
  j["tessellate"] = opt.tessellate;
  j["input_channels"] = opt.input_channels;
  j["output_channels"] = opt.output_channels;

  j["q_zp"] = opt.q_zp;
  j["q_scale"] = opt.q_scale;
  j["channel_mean"] = json::array({mean3[0], mean3[1], mean3[2]});
  j["channel_stddev"] = json::array({std3[0], std3[1], std3[2]});

  j["input_img_type"] = in_fmt;
  j["output_img_type"] = out_fmt;
  j["scaling_type"] = opt.scaling_type;
  j["padding_type"] = opt.padding_type;
  j["input_stride"] = opt.input_stride;
  j["output_stride"] = opt.output_stride;
  j["output_dtype"] = opt.output_dtype;

  json sink_params = json::array({
      {
          {"name", "format"},
          {"type", "string"},
          {"values", "GRAY, RGB, BGR, I420, NV12"},
          {"json_field", "input_img_type"},
      },
  });
  sink_params.push_back({
      {"name", "width"},
      {"type", "int"},
      {"values", "1 - 4096"},
      {"json_field", "input_width"},
  });
  sink_params.push_back({
      {"name", "height"},
      {"type", "int"},
      {"values", "1 - 4096"},
      {"json_field", "input_height"},
  });

  j["caps"] = {
      {"sink_pads", json::array({
                        {
                            {"media_type", "video/x-raw"},
                            {"params", std::move(sink_params)},
                        },
                    })},
      {"src_pads", json::array({
                       {
                           {"media_type", "video/x-raw"},
                           {"params", json::array({
                                          {
                                              {"name", "format"},
                                              {"type", "string"},
                                              {"values", "RGB, BGR"},
                                              {"json_field", "output_img_type"},
                                          },
                                          {
                                              {"name", "width"},
                                              {"type", "int"},
                                              {"values", "1 - 4096"},
                                              {"json_field", "output_width"},
                                          },
                                          {
                                              {"name", "height"},
                                              {"type", "int"},
                                              {"values", "1 - 4096"},
                                              {"json_field", "output_height"},
                                          },
                                      })},
                       },
                   })},
  };

  return j;
}

} // namespace

Preproc::Preproc(PreprocOptions opt) : opt_(std::move(opt)) {
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error("Preproc: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
          "Preproc: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  const int in_w = (opt_.input_width > 0) ? opt_.input_width : opt_.output_width;
  const int in_h = (opt_.input_height > 0) ? opt_.input_height : opt_.output_height;
  const int out_w = (opt_.output_width > 0) ? opt_.output_width : in_w;
  const int out_h = (opt_.output_height > 0) ? opt_.output_height : in_h;
  const int scaled_w = (opt_.scaled_width > 0) ? opt_.scaled_width : out_w;
  const int scaled_h = (opt_.scaled_height > 0) ? opt_.scaled_height : out_h;
  const bool split_config = opt_.keep_config && opt_.dynamic_input_dims &&
                            opt_.config_path.empty() && !opt_.config_dir.empty();

  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("Preproc: invalid input/output dimensions");
  }

  if (opt_.input_channels <= 0) {
    opt_.input_channels = channels_from_format(opt_.input_img_type, 3);
  }
  if (opt_.output_channels <= 0) {
    opt_.output_channels = channels_from_format(opt_.output_img_type, 3);
  }

  auto holder = std::make_shared<PreprocConfigHolder>();
  if (opt_.config_json.has_value()) {
    holder->config = *opt_.config_json;
    holder->has_config = true;
    if (!opt_.config_path.empty()) {
      config_path_ = opt_.config_path;
      write_json_file(holder->config, config_path_, "Preproc");
      holder->keep = true;
    } else if (split_config) {
      // Keep a stable snapshot for inspection, but use a runtime temp file.
      config_snapshot_path_ = write_json_temp(holder->config, opt_.config_dir);
      config_path_ = write_json_temp(holder->config, /*dir=*/"");
      holder->keep = false;
    } else {
      config_path_ = write_json_temp(holder->config, opt_.config_dir);
      holder->keep = opt_.keep_config;
    }
    holder->path = config_path_;
    config_holder_ = std::move(holder);
    return;
  }

  if (!opt_.config_path.empty()) {
    config_path_ = opt_.config_path;
    holder->config = load_json_file(config_path_, "Preproc");
    holder->has_config = true;
    holder->path = config_path_;
    holder->keep = true;
    config_holder_ = std::move(holder);
    return;
  }

  holder->config = build_preproc_json(opt_, in_w, in_h, out_w, out_h, scaled_w, scaled_h);
  holder->has_config = true;
  if (split_config) {
    config_snapshot_path_ = write_json_temp(holder->config, opt_.config_dir);
    config_path_ = write_json_temp(holder->config, /*dir=*/"");
    holder->keep = false;
  } else {
    config_path_ = write_json_temp(holder->config, opt_.config_dir);
    holder->keep = opt_.keep_config;
  }
  holder->path = config_path_;
  config_holder_ = std::move(holder);
}

std::string Preproc::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatprocesscvu", "Preproc::backend_fragment");
  const char* factory = "neatprocesscvu";
  const std::string name = opt_.element_name.empty()
                               ? ("n" + std::to_string(node_index) + "_preproc")
                               : opt_.element_name;
  ss << factory << " name=" << name << " stage-id=" << name;
  if (!config_path_.empty()) {
    ss << " config=\"" << config_path_ << "\"";
  }
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> Preproc::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_preproc"};
}

OutputSpec Preproc::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = upper_copy(opt_.output_img_type);
  out.width = (opt_.output_width > 0) ? opt_.output_width : opt_.input_width;
  out.height = (opt_.output_height > 0) ? opt_.output_height : opt_.input_height;
  out.depth =
      (opt_.output_channels > 0) ? opt_.output_channels : channels_from_format(out.format, 3);
  out.layout = (out.format == "GRAY" || out.format == "GRAY8") ? "HW" : "HWC";
  out.dtype = "UInt8";
  if (upper_copy(opt_.next_cpu) == "APU") {
    out.memory = "SystemMemory";
  } else if (!input.memory.empty()) {
    out.memory = input.memory;
  } else {
    out.memory = "SimaAI";
  }
  out.certainty = SpecCertainty::Derived;
  out.note = "neatprocesscvu";
  out.byte_size = expected_byte_size(out);
  return out;
}

bool Preproc::set_next_cpu_if_auto(const std::string& next_cpu) {
  if (!config_holder_ || !config_holder_->has_config)
    return false;
  if (next_cpu.empty())
    return false;
  json& j = config_holder_->config;
  if (has_next_cpu_manual(j))
    return false;

  // Semantic change => must rewrite config (mtime changes).
  j["next_cpu"] = next_cpu;
  opt_.next_cpu = next_cpu;

  if (!config_path_.empty()) {
    write_json_file(j, config_path_, "Preproc");
  }
  return true;
}

bool Preproc::wire_input_names(const std::vector<std::string>& upstream_names,
                               const std::string& tag) {
  if (upstream_names.empty() || upstream_names[0].empty())
    return false;
  return override_config_json(
      [&](json& j) { (void)set_input_buffer_name_if_exists(j, upstream_names[0]); }, tag);
}

const nlohmann::json* Preproc::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

bool Preproc::override_config_json(const std::function<void(json&)>& edit, const std::string& tag) {
  if (!config_holder_ || !config_holder_->has_config)
    return false;

  const json& old_cfg = config_holder_->config;
  json new_cfg = old_cfg;
  edit(new_cfg);
  const std::string next_cpu = read_next_cpu_string(new_cfg);
  if (!next_cpu.empty()) {
    opt_.next_cpu = next_cpu;
  }

  // Fast no-op: if the edit only changes input_width/input_height (dims-only reneg),
  // do NOT rewrite the JSON on disk. CVU will handle dims automatically.
  if (equal_ignoring_input_dims(new_cfg, old_cfg)) {
    if (pipeline_internal::env_bool("SIMA_PREPROC_DEBUG_CONFIG", false)) {
      std::fprintf(stderr, "[DBG] Preproc override_config_json no-rewrite tag=%s path=%s\n",
                   tag.c_str(), config_path_.c_str());
    }
    // Still update in-memory config to preserve any runtime-only bookkeeping,
    // but keep disk untouched for dims-only changes.
    config_holder_->config = std::move(new_cfg);
    return true;
  }

  // Semantic change (format, next_cpu, dtype, memory order, etc).
  // Keep the same path if we already have one; this ensures callers observing a single
  // config file see a rewrite (mtime changes) only when non-dims fields change.
  if (!config_path_.empty()) {
    if (pipeline_internal::env_bool("SIMA_PREPROC_DEBUG_CONFIG", false)) {
      std::fprintf(stderr, "[DBG] Preproc override_config_json rewrite tag=%s path=%s\n",
                   tag.c_str(), config_path_.c_str());
    }
    write_json_file(new_cfg, config_path_, "Preproc");
    if (!config_snapshot_path_.empty()) {
      write_json_file(new_cfg, config_snapshot_path_, "Preproc");
    }
    config_holder_->config = std::move(new_cfg);
    config_holder_->path = config_path_;
    // preserve keep flag (do not change lifetime semantics here)
    return true;
  }

  // If we somehow have no config_path yet, fall back to temp.
  config_path_ = write_json_temp(new_cfg, opt_.config_dir);
  auto holder = std::make_shared<PreprocConfigHolder>();
  holder->config = std::move(new_cfg);
  holder->has_config = true;
  holder->path = config_path_;
  holder->keep = opt_.keep_config;
  config_holder_ = std::move(holder);
  return true;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Preproc(PreprocOptions opt) {
  return std::make_shared<simaai::neat::Preproc>(std::move(opt));
}

} // namespace simaai::neat::nodes

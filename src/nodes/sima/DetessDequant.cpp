#include "nodes/sima/DetessDequant.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "mpk/PipelineSequence.h"
#include "builder/ConfigJsonWire.h"
#include "pipeline/internal/TempJsonFileUtil.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace fs = std::filesystem;
using json = nlohmann::json;

struct DetessDequant::ConfigHolder {
  std::string path;
  bool keep = false;
  json config;
  bool has_config = false;
  ~ConfigHolder() {
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
};

namespace {

bool env_bool(const char* name, bool def_val = false) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return def_val;
  std::string s(v);
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (s == "0" || s == "false" || s == "no" || s == "off")
    return false;
  if (s == "1" || s == "true" || s == "yes" || s == "on")
    return true;
  return def_val;
}

std::string make_temp_json_path(const std::string& dir) {
  return pipeline_internal::make_temp_json_path(dir, "sima_detessdequant", "DetessDequant");
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
  out << j.dump(2);
}

std::string write_json_temp(const json& j, const std::string& dir) {
  const std::string path = make_temp_json_path(dir);
  write_json_file(j, path, "DetessDequant");
  return path;
}

std::string find_config_by_kernel(const simaai::neat::internal::ModelPack& model,
                                  const char* kernel, const char* fallback_name) {
  std::vector<simaai::neat::mpk::SequenceEntry> seq =
      simaai::neat::mpk::load_pipeline_sequence(model.etc_dir());
  for (const auto& entry : seq) {
    if (entry.kernel == kernel && !entry.config_path.empty()) {
      return (fs::path(model.etc_dir()) / entry.config_path).string();
    }
  }
  if (fallback_name && *fallback_name) {
    const fs::path fallback = fs::path(model.etc_dir()) / fallback_name;
    if (fs::exists(fallback))
      return fallback.string();
  }
  throw std::runtime_error(std::string("DetessDequant: config not found for kernel ") +
                           (kernel ? kernel : ""));
}

std::string read_node_name(const json& j) {
  if (j.contains("node_name") && j["node_name"].is_string()) {
    return j["node_name"].get<std::string>();
  }
  return "";
}

[[maybe_unused]] bool looks_like_element_name(const std::string& name) {
  if (name.empty())
    return false;
  if (name.find("neatprocess") != std::string::npos)
    return true;
  if (name.find("simaaiprocess") != std::string::npos)
    return true;
  if (name.rfind("n", 0) == 0)
    return true;
  if (name.find("_n") != std::string::npos)
    return true;
  return false;
}

[[maybe_unused]] bool copy_array_if_present(json& dst, const json& src, const char* dst_key,
                                            const char* src_key) {
  if (!src.contains(src_key) || !src[src_key].is_array())
    return false;
  dst[dst_key] = src[src_key];
  return true;
}

void override_cpu_fields(json& j, const char* key, const std::string& cpu_str, int cpu_int) {
  if (!j.contains(key))
    return;
  if (j[key].is_string()) {
    j[key] = cpu_str;
  } else if (j[key].is_number_integer()) {
    j[key] = cpu_int;
  }
}

void override_cpu_fields_in_object(json& j, const char* obj_key, const std::string& cpu_str,
                                   int cpu_int) {
  if (!j.contains(obj_key) || !j[obj_key].is_object())
    return;
  auto& obj = j[obj_key];
  override_cpu_fields(obj, "cpu", cpu_str, cpu_int);
  override_cpu_fields(obj, "next_cpu", cpu_str, cpu_int);
}

void set_input_buffer_name(json& j, const std::string& name) {
  (void)set_input_buffer_name_if_exists(j, name);
}

std::shared_ptr<DetessDequant::ConfigHolder> init_config_holder(const DetessDequantOptions& opt,
                                                                std::string& config_path_out) {
  auto holder = std::make_shared<DetessDequant::ConfigHolder>();
  if (opt.config_json.has_value()) {
    holder->config = *opt.config_json;
    holder->has_config = true;
  } else if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    holder->config = load_json_file(config_path_out, "DetessDequant");
    holder->has_config = true;
    holder->keep = true;
  }

  if (!holder->has_config)
    return holder;

  if (opt.config_json.has_value() && !opt.config_path.empty()) {
    write_json_file(holder->config, config_path_out, "DetessDequant");
    holder->keep = true;
  } else if (opt.config_json.has_value()) {
    config_path_out = write_json_temp(holder->config, opt.config_dir);
    holder->keep = opt.keep_config;
  }

  holder->path = config_path_out;
  return holder;
}

} // namespace

DetessDequantOptions::DetessDequantOptions(const simaai::neat::Model& model) {
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  config_path = find_config_by_kernel(pack, "detessdequant", "0_postproc.json");
  num_buffers_model = pack.num_buffers_cvu();
  num_buffers = num_buffers_model;
  num_buffers_locked = true;
  if (std::getenv("SIMA_KEEP_DETESS_CONFIG")) {
    keep_config = true;
  }
  const auto split = pack.split_sequence();
  if (!split.infer.empty()) {
    upstream_name = split.infer.back().name;
  }
  for (const auto& entry : split.post) {
    if (entry.kernel == "detessdequant" && !entry.name.empty()) {
      element_name = entry.name;
      break;
    }
  }
}

DetessDequant::DetessDequant(DetessDequantOptions opt) : opt_(std::move(opt)) {
  if (std::getenv("SIMA_DETESS_MULTI_BUFFER")) {
    throw std::runtime_error(
        "DetessDequant: SIMA_DETESS_MULTI_BUFFER is not supported; num_buffers is model-managed.");
  }
  if (!opt_.num_buffers_locked) {
    throw std::runtime_error(
        "DetessDequant: num_buffers must be model-managed (use DetessDequantOptions(Model)).");
  }
  if (opt_.num_buffers != opt_.num_buffers_model) {
    throw std::runtime_error(
        "DetessDequant: num_buffers override is not allowed; must match model.");
  }
  if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
    throw std::runtime_error(
        "DetessDequant: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
  }
  config_holder_ = init_config_holder(opt_, config_path_);
  if (!opt_.upstream_name.empty()) {
    override_config_json([&](json& j) { set_input_buffer_name(j, opt_.upstream_name); },
                         "detessdequant_upstream");
  }
  if (env_bool("SIMA_DETESS_FORCE_CPU_OUT", false)) {
    override_config_json(
        [&](json& j) {
          override_cpu_fields(j, "next_cpu", "APU", 0);
          override_cpu_fields_in_object(j, "memory", "APU", 0);
          override_cpu_fields_in_object(j, "simaai__params", "APU", 0);
        },
        "detessdequant_force_cpu_out");
  }
}

std::string DetessDequant::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatprocesscvu", "DetessDequant::backend_fragment");
  const std::string name = opt_.element_name.empty()
                               ? ("n" + std::to_string(node_index) + "_detessdequant")
                               : opt_.element_name;
  ss << "neatprocesscvu name=" << name << " stage-id=" << name;
  if (!config_path_.empty()) {
    ss << " config=\"" << config_path_ << "\"";
  }
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> DetessDequant::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_detessdequant"};
}

OutputSpec DetessDequant::output_spec(const OutputSpec& input) const {
  OutputSpec out = input;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "DETESSDEQUANT";
  if (out.depth <= 0)
    out.depth = input.depth;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatprocesscvu";
  return out;
}

const nlohmann::json* DetessDequant::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

bool DetessDequant::wire_input_names(const std::vector<std::string>& upstream_names,
                                     const std::string& tag) {
  if (upstream_names.empty() || upstream_names[0].empty())
    return false;
  return override_config_json(
      [&](json& j) { (void)set_input_buffer_name_if_exists(j, upstream_names[0]); }, tag);
}

bool DetessDequant::override_config_json(const std::function<void(json&)>& edit,
                                         const std::string& tag) {
  (void)tag;
  if (!config_holder_ || !config_holder_->has_config)
    return false;
  json cfg = config_holder_->config;
  edit(cfg);
  config_path_ = write_json_temp(cfg, opt_.config_dir);
  auto holder = std::make_shared<ConfigHolder>();
  holder->config = std::move(cfg);
  holder->has_config = true;
  holder->path = config_path_;
  holder->keep = opt_.keep_config;
  config_holder_ = std::move(holder);
  return true;
}

void DetessDequant::apply_upstream_config(const nlohmann::json& upstream, const std::string&) {
  if (!config_holder_ || !config_holder_->has_config)
    return;
  const std::string node_name = read_node_name(upstream);
  if (node_name.empty())
    return;
  override_config_json([&](json& j) { (void)set_input_buffer_name_if_exists(j, node_name); },
                       "detessdequant_upstream");
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> DetessDequant(DetessDequantOptions opt) {
  return std::make_shared<simaai::neat::DetessDequant>(std::move(opt));
}
} // namespace simaai::neat::nodes

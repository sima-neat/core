#include "nodes/sima/QuantTess.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "mpk/PipelineSequence.h"
#include "builder/ConfigJsonWire.h"
#include "pipeline/internal/TempJsonFileUtil.h"

#include <nlohmann/json.hpp>

#include <cstdio>
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

struct QuantTess::ConfigHolder {
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

std::string make_temp_json_path(const std::string& dir) {
  return pipeline_internal::make_temp_json_path(dir, "sima_quanttess", "QuantTess");
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
  write_json_file(j, path, "QuantTess");
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
  throw std::runtime_error(std::string("QuantTess: config not found for kernel ") +
                           (kernel ? kernel : ""));
}

std::string read_node_name(const json& j) {
  if (j.contains("node_name") && j["node_name"].is_string()) {
    return j["node_name"].get<std::string>();
  }
  return "";
}

std::string pick_input_name(const std::vector<std::string>& names, std::size_t index) {
  if (index < names.size() && !names[index].empty())
    return names[index];
  for (std::size_t i = names.size(); i > 0; --i) {
    if (!names[i - 1].empty())
      return names[i - 1];
  }
  return "";
}

bool set_input_buffer_names_by_index_if_exists(json& j, const std::vector<std::string>& names) {
  auto rewrite_array = [&](json& arr) -> bool {
    if (!arr.is_array() || arr.empty())
      return false;
    bool changed = false;
    for (std::size_t i = 0; i < arr.size(); ++i) {
      auto& entry = arr[i];
      if (!entry.is_object())
        continue;

      const std::string selected = pick_input_name(names, i);
      if (selected.empty())
        continue;

      if (arr.size() == 1) {
        auto it = entry.find("name");
        if (it == entry.end() || (it->is_string() && it->get<std::string>().empty())) {
          continue;
        }
      }

      auto it = entry.find("name");
      if (it != entry.end() && it->is_string() && it->get<std::string>() == selected) {
        continue;
      }
      entry["name"] = selected;
      changed = true;
    }
    return changed;
  };

  bool changed = false;
  if (j.contains("input_buffers")) {
    changed = rewrite_array(j["input_buffers"]) || changed;
  }
  if (j.contains("buffers") && j["buffers"].is_object() && j["buffers"].contains("input")) {
    changed = rewrite_array(j["buffers"]["input"]) || changed;
  }
  return changed;
}

void set_input_buffer_name(json& j, const std::string& name) {
  (void)set_input_buffer_name_if_exists(j, name);
}

std::shared_ptr<QuantTess::ConfigHolder> init_config_holder(const QuantTessOptions& opt,
                                                            std::string& config_path_out) {
  auto holder = std::make_shared<QuantTess::ConfigHolder>();
  if (opt.config_json.has_value()) {
    holder->config = *opt.config_json;
    holder->has_config = true;
  } else if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    holder->config = load_json_file(config_path_out, "QuantTess");
    holder->has_config = true;
    holder->keep = true;
  }

  if (!holder->has_config)
    return holder;

  if (opt.config_json.has_value() && !opt.config_path.empty()) {
    write_json_file(holder->config, config_path_out, "QuantTess");
    holder->keep = true;
  } else if (opt.config_json.has_value()) {
    config_path_out = write_json_temp(holder->config, opt.config_dir);
    holder->keep = opt.keep_config;
  }

  holder->path = config_path_out;
  return holder;
}

} // namespace

QuantTessOptions::QuantTessOptions(const simaai::neat::Model& model) {
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  config_path = find_config_by_kernel(pack, "quanttess", "0_quanttess.json");
  num_buffers_model = pack.num_buffers_cvu();
  num_buffers = num_buffers_model;
  num_buffers_locked = true;
}

QuantTess::QuantTess(QuantTessOptions opt) : opt_(std::move(opt)) {
  if (opt_.num_buffers_locked) {
    if (opt_.num_buffers != opt_.num_buffers_model) {
      throw std::runtime_error("QuantTess: num_buffers override is not allowed; must match model.");
    }
    if (opt_.num_buffers != 4 && opt_.num_buffers != 1) {
      throw std::runtime_error(
          "QuantTess: num_buffers must be 4 (async) or 1 (sync) for model pipelines.");
    }
  }
  config_holder_ = init_config_holder(opt_, config_path_);
}

std::string QuantTess::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatprocesscvu", "QuantTess::backend_fragment");
  const std::string name = opt_.element_name.empty()
                               ? ("n" + std::to_string(node_index) + "_quanttess")
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

std::vector<std::string> QuantTess::element_names(int node_index) const {
  if (!opt_.element_name.empty())
    return {opt_.element_name};
  return {"n" + std::to_string(node_index) + "_quanttess"};
}

const nlohmann::json* QuantTess::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

bool QuantTess::wire_input_names(const std::vector<std::string>& upstream_names,
                                 const std::string& tag) {
  bool has_name = false;
  for (const auto& name : upstream_names) {
    if (!name.empty()) {
      has_name = true;
      break;
    }
  }
  if (!has_name)
    return false;

  return override_config_json(
      [&](json& j) {
        if (set_input_buffer_names_by_index_if_exists(j, upstream_names))
          return;
        const std::string fallback = pick_input_name(upstream_names, 0);
        if (!fallback.empty()) {
          (void)set_input_buffer_name_if_exists(j, fallback);
        }
      },
      tag);
}

bool QuantTess::override_config_json(const std::function<void(json&)>& edit,
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

void QuantTess::apply_upstream_config(const nlohmann::json& upstream, const std::string&) {
  if (!config_holder_ || !config_holder_->has_config)
    return;
  const std::string node_name = read_node_name(upstream);
  if (node_name.empty())
    return;
  set_input_buffer_name(config_holder_->config, node_name);
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> QuantTess(QuantTessOptions opt) {
  return std::make_shared<simaai::neat::QuantTess>(std::move(opt));
}
} // namespace simaai::neat::nodes

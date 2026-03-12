#include "model/internal/ModelPack.h"
#include "mpk/MpKLoader.h"
#include "mpk/PipelineSequence.h"

#include "builder/ConfigJsonConsumer.h"
#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonOverrideMulti.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/ConfigJsonWire.h"
#include "builder/NextCpuConfigurable.h"
#include "gst/GstHelpers.h"
#include "pipeline/internal/TempJsonFileUtil.h"

#include <nlohmann/json.hpp>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

#include <array>
#include <algorithm>
#include <cctype>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

namespace simaai::neat::internal {
namespace fs = std::filesystem;
using json = nlohmann::json;
using simaai::neat::mpk::load_pipeline_sequence;
using simaai::neat::mpk::SequenceEntry;
using simaai::neat::mpk::SequenceSplit;
using simaai::neat::mpk::split_sequence_for_infer;

namespace {

constexpr const char* kDefaultBaseOutputDir = "/data/simaai/coprocessing/models/";
constexpr const char* kDirConf = "etc";

constexpr const char* kDefaultPreviousNodeName = "decoder";

static std::string to_upper(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::string to_lower(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::string resolve_stage_factory(const std::string& plugin_id) {
  // Runtime policy: always instantiate NEAT factories from model stages.
  static const std::unordered_map<std::string, std::string> kNeatFactoryOverrides = {
      {"processcvu", "neatprocesscvu"}, {"process_cvu", "neatprocesscvu"},
      {"processmla", "neatprocessmla"}, {"process_mla", "neatprocessmla"},
      {"boxdecode", "neatboxdecode"},   {"box_decode", "neatboxdecode"},
      {"pciesrc", "neatpciesrc"},       {"pcie_src", "neatpciesrc"},
      {"pciesink", "neatpciesink"},     {"pcie_sink", "neatpciesink"},
      {"processtvm", "neatprocesstvm"}, {"process_tvm", "neatprocesstvm"},
  };

  std::string factory = plugin_id;
  if (const auto it = kNeatFactoryOverrides.find(plugin_id); it != kNeatFactoryOverrides.end()) {
    factory = it->second;
  }

  if (factory.rfind("simaai", 0) == 0) {
    throw std::runtime_error("ModelFragment: legacy SIMAAI factory is not allowed: " + factory);
  }

  if (!simaai::neat::element_exists(factory.c_str())) {
    throw std::runtime_error("ModelFragment: required NEAT factory not found: " + factory +
                             " (plugin_id=" + plugin_id + ")");
  }
  return factory;
}

static std::string normalize_format(std::string fmt) {
  fmt = to_upper(fmt);
  if (fmt == "GRAY8")
    fmt = "GRAY";
  if (fmt == "I420")
    fmt = "IYUV";
  return fmt;
}

static std::array<float, 3> materialize3(const std::vector<float>& v, float defv) {
  if (v.empty())
    return {defv, defv, defv};
  if (v.size() == 1)
    return {v[0], v[0], v[0]};
  if (v.size() == 3)
    return {v[0], v[1], v[2]};
  throw std::invalid_argument("mean/stddev must have 0, 1, or 3 values.");
}

static std::string append_model_paths_if_exists(json& json_data, const std::string& append_path) {
  if (json_data.contains("simaai__params") && json_data["simaai__params"].contains("model_path")) {
    std::string model_path = json_data["simaai__params"]["model_path"];
    json_data["simaai__params"]["model_path"] = append_path + "/share/" + model_path;
  } else if (json_data.contains("model_info") && json_data["model_info"].contains("path")) {
    std::string model_info_path = json_data["model_info"]["path"];
    json_data["model_info"]["path"] = append_path + "/lib/" + model_info_path;
  }
  return "";
}

static std::string modelpack_output_root() {
  const char* env_root = std::getenv("SIMA_MPK_EXTRACT_ROOT");
  if (env_root && *env_root)
    return std::string(env_root);

  auto is_writable_dir = [](const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
      return false;
    const fs::path probe = dir / ".sima_modelpack_write_probe";
    std::ofstream out(probe, std::ios::out | std::ios::trunc);
    if (!out.is_open())
      return false;
    out << "ok";
    out.close();
    fs::remove(probe, ec);
    return true;
  };

  auto choose_base = [&]() -> fs::path {
    const fs::path preferred(kDefaultBaseOutputDir);
    if (is_writable_dir(preferred))
      return preferred;

    const char* tmpdir = std::getenv("TMPDIR");
    fs::path tmp_base = (tmpdir && *tmpdir) ? fs::path(tmpdir) : fs::path("/tmp");
    tmp_base /= "simaai/coprocessing/models";
    if (is_writable_dir(tmp_base))
      return tmp_base;

    fs::path cwd_base = fs::current_path() / "tmp" / "model_extract";
    if (is_writable_dir(cwd_base))
      return cwd_base;

    return preferred;
  };

  static const std::string root = [&]() {
    fs::path base = choose_base();
    base /= ("proc_" + std::to_string(static_cast<long long>(::getpid())));
    std::error_code ec;
    fs::create_directories(base, ec);
    return base.string();
  }();
  return root;
}

static bool directory_has_json(const fs::path& dir) {
  std::error_code ec;
  if (!fs::exists(dir, ec) || ec || !fs::is_directory(dir, ec)) {
    return false;
  }
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    if (entry.path().extension() == ".json")
      return true;
  }
  return false;
}

static bool extracted_layout_ready(const fs::path& package_root) {
  std::error_code ec;
  const fs::path etc_dir = package_root / "etc";
  const fs::path lib_dir = package_root / "lib";
  const fs::path share_dir = package_root / "share";
  if (!fs::exists(etc_dir, ec) || ec || !fs::is_directory(etc_dir, ec)) {
    return false;
  }
  if (!fs::exists(lib_dir, ec) || ec || !fs::is_directory(lib_dir, ec)) {
    return false;
  }
  if (!fs::exists(share_dir, ec) || ec || !fs::is_directory(share_dir, ec)) {
    return false;
  }
  return directory_has_json(etc_dir);
}

static std::string archive_cache_key(const std::string& tar_path) {
  std::error_code ec;
  fs::path p = fs::absolute(fs::path(tar_path), ec);
  if (ec) {
    p = fs::path(tar_path);
    ec.clear();
  }
  const auto size = fs::file_size(p, ec);
  if (ec) {
    return p.string();
  }
  ec.clear();
  const auto mtime = fs::last_write_time(p, ec);
  if (ec) {
    return p.string() + "|" + std::to_string(static_cast<unsigned long long>(size));
  }
  const auto stamp = mtime.time_since_epoch().count();
  return p.string() + "|" + std::to_string(static_cast<unsigned long long>(size)) + "|" +
         std::to_string(static_cast<long long>(stamp));
}

static std::unordered_map<std::string, std::string>& modelpack_extract_cache() {
  static std::unordered_map<std::string, std::string> cache;
  return cache;
}

static std::mutex& modelpack_extract_cache_mutex() {
  static std::mutex mu;
  return mu;
}

static std::string extract_and_organize(const std::string& tar_path) {
  const std::string cache_key = archive_cache_key(tar_path);
  simaai::neat::mpk::MpKLoaderOptions opt;
  // Runtime model packs may include auxiliary build/report artifacts.
  // Keep strict type validation in security/unit tests (default options),
  // but allow these extras in ModelPack runtime extraction.
  opt.reject_unsupported_file_types = false;
  try {
    std::lock_guard<std::mutex> lock(modelpack_extract_cache_mutex());
    auto& cache = modelpack_extract_cache();
    const auto found = cache.find(cache_key);
    if (found != cache.end() && extracted_layout_ready(fs::path(found->second))) {
      return found->second;
    }

    const auto extracted =
        simaai::neat::mpk::MpKLoader::extract(tar_path, modelpack_output_root(), opt);
    const fs::path target_dir(extracted.package_root);

    // Preserve existing behavior: materialize model-relative paths as absolute paths
    // anchored at extracted package root.
    for (const auto& entry : fs::directory_iterator(extracted.etc_dir)) {
      if (!entry.is_regular_file())
        continue;
      if (entry.path().extension() != ".json")
        continue;
      std::ifstream in(entry.path());
      if (!in.is_open())
        continue;
      json cfg;
      try {
        in >> cfg;
      } catch (const std::exception&) {
        continue;
      }
      append_model_paths_if_exists(cfg, target_dir.string());
      std::ofstream out(entry.path());
      if (!out.is_open())
        continue;
      out << cfg.dump(4);
    }

    cache[cache_key] = target_dir.string();
    return target_dir.string();
  } catch (const simaai::neat::mpk::MpKError& e) {
    throw std::runtime_error(std::string("ModelPack: ") + e.what());
  }
}

static std::string make_temp_json_path(const std::string& dir, const std::string& tag);

static std::string update_input_buffers_name(const std::string& file_path,
                                             const std::string& previous_node_name) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open()) {
    return "Failed to open the JSON file.";
  }

  json json_data;
  json_file >> json_data;

  if (json_data.contains("input_buffers") && json_data["input_buffers"][0].contains("name")) {
    json_data["input_buffers"][0]["name"] = previous_node_name;
    std::ofstream updated_json_file(file_path);
    if (!updated_json_file.is_open()) {
      return "Failed to open the file for writing.";
    }
    updated_json_file << json_data.dump(4);
    return "";
  }

  return "input_buffers->name not found.";
}

static std::string rewrite_node_and_input_names(const std::string& file_path,
                                                const std::string& tag,
                                                const std::string& node_name,
                                                const std::string& previous_node_name,
                                                bool set_next_cpu, int next_cpu) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open()) {
    return file_path;
  }

  json json_data;
  json_file >> json_data;

  if (set_next_cpu && json_data.contains("simaai__params") &&
      json_data["simaai__params"].is_object()) {
    json_data["simaai__params"]["next_cpu"] = next_cpu;
  }

  if (!node_name.empty()) {
    json_data["node_name"] = node_name;
  }

  if (!previous_node_name.empty() && json_data.contains("input_buffers") &&
      json_data["input_buffers"].is_array() && !json_data["input_buffers"].empty() &&
      json_data["input_buffers"][0].is_object()) {
    json_data["input_buffers"][0]["name"] = previous_node_name;
  }

  const std::string out_path = make_temp_json_path("/tmp", tag);
  std::ofstream updated_json_file(out_path);
  if (!updated_json_file.is_open()) {
    return file_path;
  }
  updated_json_file << json_data.dump(4);
  return out_path;
}

static bool parse_mla_next_cpu_override(int& out) {
  const char* v = std::getenv("SIMA_MLA_NEXT_CPU");
  if (!v || !*v)
    return false;
  std::string s(v);
  std::string upper = to_upper(s);
  if (upper == "APU") {
    out = 0;
    return true;
  }
  if (upper == "CVU" || upper == "MLA") {
    out = 1;
    return true;
  }
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (end && *end == '\0') {
    out = static_cast<int>(val);
    return true;
  }
  return false;
}

static void update_mla_next_cpu(const std::string& file_path, int next_cpu) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open())
    return;

  json json_data;
  json_file >> json_data;

  if (!json_data.contains("simaai__params") || !json_data["simaai__params"].is_object()) {
    return;
  }

  json_data["simaai__params"]["next_cpu"] = next_cpu;
  std::ofstream updated_json_file(file_path);
  if (!updated_json_file.is_open())
    return;
  updated_json_file << json_data.dump(4);
}

static std::string make_temp_json_path(const std::string& dir, const std::string& tag) {
  std::string prefix = "sima_mpk";
  if (!tag.empty()) {
    prefix += "_" + tag;
  }
  return simaai::neat::pipeline_internal::make_temp_json_path(dir, prefix, "ModelPack");
}

static bool has_next_cpu_manual(const json& j) {
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

class ModelFragmentNode final : public Node,
                                public ConfigJsonProvider,
                                public ConfigJsonConsumer,
                                public ConfigJsonOverride,
                                public ConfigJsonOverrideMulti,
                                public NextCpuConfigurable {
public:
  ModelFragmentNode(std::string kind, std::string label, std::string fragment,
                    std::vector<std::string> elements, std::vector<std::string> config_paths)
      : kind_(std::move(kind)), label_(std::move(label)), fragment_(std::move(fragment)),
        elements_(std::move(elements)), config_paths_(std::move(config_paths)) {
    load_configs();
    validate_next_cpu(config_entries_);
  }

  std::string kind() const override {
    return kind_;
  }
  std::string user_label() const override {
    return label_;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  bool has_config_json() const override {
    return !config_entries_.empty();
  }
  std::string buffer_name_hint(int) const override {
    if (!elements_.empty())
      return elements_.back();
    return "";
  }
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override {
    if (config_entries_.empty() || upstream_names.empty())
      return false;
    bool any_changed = false;
    for (size_t i = 0; i < config_entries_.size(); ++i) {
      const std::string& name =
          (i < upstream_names.size()) ? upstream_names[i] : upstream_names.back();
      if (name.empty())
        continue;
      nlohmann::json cfg = config_entries_[i].config;
      bool changed = set_input_buffer_name_if_exists(cfg, name);
      if (!changed) {
        if (!cfg.contains("input_buffers") || !cfg["input_buffers"].is_array() ||
            cfg["input_buffers"].empty()) {
          cfg["input_buffers"] = nlohmann::json::array({nlohmann::json::object()});
          cfg["input_buffers"][0]["name"] = name;
          changed = true;
        }
      }
      if (!changed)
        continue;
      const std::string old_path = config_entries_[i].path;
      const std::string new_path = write_temp_config(cfg, tag);
      config_entries_[i].path = new_path;
      config_entries_[i].config = std::move(cfg);
      if (i < config_paths_.size())
        config_paths_[i] = new_path;
      if (!old_path.empty())
        replace_all(fragment_, old_path, new_path);
      temp_paths_.push_back(new_path);
      any_changed = true;
    }
    return any_changed;
  }
  std::string backend_fragment(int) const override {
    return fragment_;
  }
  std::vector<std::string> element_names(int) const override {
    return elements_;
  }
  const nlohmann::json* config_json() const override {
    if (config_entries_.empty())
      return nullptr;
    return &config_entries_.front().config;
  }
  bool set_next_cpu_if_auto(const std::string& next_cpu) override {
    if (next_cpu.empty())
      return false;
    for (size_t i = 0; i < config_entries_.size(); ++i) {
      if (!is_preproc_entry(config_entries_[i]))
        continue;
      if (has_next_cpu_manual(config_entries_[i].config))
        return false;
      config_entries_[i].config["next_cpu"] = next_cpu;
      write_config(i);
      return true;
    }
    return false;
  }
  void apply_upstream_config(const nlohmann::json& upstream, const std::string&) override {
    if (config_entries_.empty())
      return;
    const std::string upstream_name = read_node_name(upstream);
    if (upstream_name.empty())
      return;
    nlohmann::json& j = config_entries_.front().config;
    if (j.contains("input_buffers") && j["input_buffers"].is_array() &&
        !j["input_buffers"].empty() && j["input_buffers"][0].is_object()) {
      j["input_buffers"][0]["name"] = upstream_name;
      write_config(0);
    }
  }

  bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                            const std::string& tag) override {
    if (config_entries_.empty())
      return false;
    bool changed = false;
    for (size_t i = 0; i < config_entries_.size(); ++i) {
      nlohmann::json cfg = config_entries_[i].config;
      edit(cfg);
      if (cfg == config_entries_[i].config)
        continue;
      const std::string old_path = config_entries_[i].path;
      const std::string new_path = write_temp_config(cfg, tag);
      config_entries_[i].path = new_path;
      config_entries_[i].config = std::move(cfg);
      if (i < config_paths_.size())
        config_paths_[i] = new_path;
      if (!old_path.empty())
        replace_all(fragment_, old_path, new_path);
      temp_paths_.push_back(new_path);
      changed = true;
    }
    return changed;
  }

  bool override_config_json_multi(
      const std::function<void(nlohmann::json&, std::size_t, std::size_t)>& edit,
      const std::string& tag) override {
    if (config_entries_.empty())
      return false;
    bool changed = false;
    const std::size_t count = config_entries_.size();
    for (std::size_t i = 0; i < config_entries_.size(); ++i) {
      nlohmann::json cfg = config_entries_[i].config;
      edit(cfg, i, count);
      if (cfg == config_entries_[i].config)
        continue;
      const std::string old_path = config_entries_[i].path;
      const std::string new_path = write_temp_config(cfg, tag);
      config_entries_[i].path = new_path;
      config_entries_[i].config = std::move(cfg);
      if (i < config_paths_.size())
        config_paths_[i] = new_path;
      if (!old_path.empty())
        replace_all(fragment_, old_path, new_path);
      temp_paths_.push_back(new_path);
      changed = true;
    }
    return changed;
  }

private:
  struct ConfigEntry {
    std::string path;
    nlohmann::json config;
  };

  static std::string read_node_name(const nlohmann::json& j) {
    if (j.contains("node_name") && j["node_name"].is_string()) {
      return j["node_name"].get<std::string>();
    }
    return "";
  }

  static bool is_preproc_entry(const ConfigEntry& entry) {
    if (!entry.path.empty()) {
      const std::string lower = to_lower(entry.path);
      if (lower.find("preproc") != std::string::npos ||
          lower.find("process_cvu") != std::string::npos) {
        return true;
      }
    }
    const nlohmann::json& j = entry.config;
    if (j.contains("input_img_type") && j.contains("output_img_type") && j.contains("next_cpu")) {
      return true;
    }
    return false;
  }

  static bool is_valid_next_cpu_value(const nlohmann::json& v) {
    if (v.is_number_integer()) {
      const int cpu = v.get<int>();
      return cpu == 0 || cpu == 1 || cpu == 2;
    }
    if (v.is_string()) {
      const std::string up = to_upper(v.get<std::string>());
      return up == "APU" || up == "CVU" || up == "MLA";
    }
    return true;
  }

  static void validate_next_cpu(const std::vector<ConfigEntry>& entries) {
    for (const auto& entry : entries) {
      const nlohmann::json& j = entry.config;
      if (j.contains("next_cpu") && !is_valid_next_cpu_value(j["next_cpu"])) {
        throw std::runtime_error("ModelFragment: invalid next_cpu value");
      }
      if (j.contains("simaai__params") && j["simaai__params"].is_object()) {
        const auto& params = j["simaai__params"];
        if (params.contains("next_cpu") && !is_valid_next_cpu_value(params["next_cpu"])) {
          throw std::runtime_error("ModelFragment: invalid simaai__params.next_cpu value");
        }
      }
    }
  }

  static void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty())
      return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  }

  void load_configs() {
    for (const auto& path : config_paths_) {
      if (path.empty())
        continue;
      std::ifstream in(path);
      if (!in.is_open())
        continue;
      nlohmann::json j;
      try {
        in >> j;
      } catch (const std::exception&) {
        continue;
      }
      config_entries_.push_back({path, std::move(j)});
    }
  }

  void write_config(size_t index) {
    if (index >= config_entries_.size())
      return;
    const std::string& path = config_entries_[index].path;
    if (path.empty())
      return;
    std::ofstream out(path);
    if (!out.is_open())
      return;
    out << config_entries_[index].config.dump(4);
  }

  std::string write_temp_config(const nlohmann::json& j, const std::string& tag) {
    const std::string path = make_temp_json_path("/tmp", tag);
    std::ofstream out(path);
    if (!out.is_open()) {
      throw std::runtime_error("ModelFragment: failed to write config override");
    }
    out << j.dump(4);
    return path;
  }

  std::string kind_;
  std::string label_;
  std::string fragment_;
  std::vector<std::string> elements_;
  std::vector<std::string> config_paths_;
  std::vector<ConfigEntry> config_entries_;
  std::vector<std::string> temp_paths_;
};

static NodeGroup make_fragment_group(const ModelFragment& frag, const std::string& label) {
  std::vector<std::shared_ptr<Node>> nodes;
  nodes.push_back(std::make_shared<ModelFragmentNode>("ModelFragment", label, frag.gst,
                                                      frag.elements, frag.config_paths));
  return NodeGroup(std::move(nodes));
}

static void set_next_cpu_manual(json& j) {
  if (!j.contains("simaai__params") || !j["simaai__params"].is_object()) {
    j["simaai__params"] = json::object();
  }
  j["simaai__params"]["next_cpu_manual"] = 1;
}

static std::tuple<int, int, std::string>
read_and_update_preproc_json(const std::string& etc_dir, int inWidth, int inHeight,
                             const std::string& inFormat, const std::string& next_cpu_override,
                             bool normalize, const std::vector<float>& channel_mean,
                             const std::vector<float>& channel_stddev) {
  std::string json_path = (fs::path(etc_dir) / "0_preproc.json").string();
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    throw std::runtime_error("ModelPack: failed to open " + json_path);
  }

  json json_data;
  json_file >> json_data;

  int out_w = 0;
  int out_h = 0;
  std::string out_img_type;
  if (inWidth > 0) {
    json_data["input_width"] = inWidth;
  } else if (json_data.contains("output_width")) {
    out_w = json_data["output_width"].get<int>();
    json_data["input_width"] = out_w;
  }
  if (inHeight > 0) {
    json_data["input_height"] = inHeight;
  } else if (json_data.contains("output_height")) {
    out_h = json_data["output_height"].get<int>();
    json_data["input_height"] = out_h;
  }
  if (!inFormat.empty()) {
    json_data["input_img_type"] = inFormat;
  } else if (json_data.contains("output_img_type")) {
    out_img_type = json_data["output_img_type"].get<std::string>();
    json_data["input_img_type"] = out_img_type;
  }

  if (!next_cpu_override.empty()) {
    json_data["next_cpu"] = next_cpu_override;
    set_next_cpu_manual(json_data);
  }

  // Keep preproc in dynamic-input mode by default so runtime input geometry
  // can vary without requiring fixed width/height at the appsrc boundary.
  json_data["dynamic_input_dims"] = true;

  if (normalize) {
    json_data["normalize"] = normalize;
    json_data["channel_mean"] = channel_mean;
    json_data["channel_stddev"] = channel_stddev;
  }

  std::ofstream updated_json_file(json_path);
  if (!updated_json_file.is_open()) {
    throw std::runtime_error("ModelPack: failed to write " + json_path);
  }
  updated_json_file << json_data.dump(4);

  return std::make_tuple(out_w, out_h, out_img_type);
}

static void patch_quanttess_caps_to_tensor_sink(const std::string& etc_dir) {
  const std::string qp = (fs::path(etc_dir) / "0_quanttess.json").string();
  std::ifstream in(qp);
  if (!in.is_open())
    throw std::runtime_error("patch: cannot open " + qp);
  json j;
  in >> j;
  in.close();

  j["caps"]["sink_pads"] = json::array({
      {
          {"media_type", "application/vnd.simaai.tensor"},
          {"params",
           json::array({
               json{{"name", "format"}, {"type", "string"}, {"values", json::array({"FP32"})}},
               json{{"json_field", "input_width"},
                    {"name", "width"},
                    {"type", "int"},
                    {"values", "1 - 4096"}},
               json{{"json_field", "input_height"},
                    {"name", "height"},
                    {"type", "int"},
                    {"values", "1 - 4096"}},
               json{{"json_field", "input_depth"},
                    {"name", "depth"},
                    {"type", "int"},
                    {"values", "1 - 4096"}},
           })},
      },
  });

  j["tile_depth"] = std::max(1, j.value("tile_depth", 3));

  std::ofstream out(qp);
  out << j.dump(4);
}

static int json_first_int(const json& j, const char* key, int fallback = -1) {
  if (!j.contains(key))
    return fallback;
  const auto& v = j.at(key);
  if (v.is_array() && !v.empty()) {
    const auto& first = v.at(0);
    if (first.is_number_integer())
      return first.get<int>();
  }
  if (v.is_number_integer())
    return v.get<int>();
  return fallback;
}

static int64_t json_first_i64(const json& j, const char* key, int64_t fallback = -1) {
  if (!j.contains(key))
    return fallback;
  const auto& v = j.at(key);
  if (v.is_array() && !v.empty()) {
    const auto& first = v.at(0);
    if (first.is_number_integer())
      return first.get<int64_t>();
  }
  if (v.is_number_integer())
    return v.get<int64_t>();
  if (v.is_number())
    return static_cast<int64_t>(v.get<double>());
  return fallback;
}

static std::string json_first_string(const json& j, const char* key,
                                     const std::string& fallback = {}) {
  if (!j.contains(key))
    return fallback;
  const auto& v = j.at(key);
  if (v.is_array() && !v.empty() && v.at(0).is_string()) {
    return v.at(0).get<std::string>();
  }
  if (v.is_string())
    return v.get<std::string>();
  return fallback;
}

static int dtype_bytes_from_string(const std::string& raw) {
  const std::string s = to_upper(raw);
  if (s.find("INT32") != std::string::npos)
    return 4;
  if (s.find("INT16") != std::string::npos)
    return 2;
  if (s.find("INT8") != std::string::npos)
    return 1;
  return 0;
}

static std::string normalize_dtype_string(const std::string& raw) {
  const std::string s = to_upper(raw);
  if (s.find("INT32") != std::string::npos)
    return "INT32";
  if (s.find("INT16") != std::string::npos)
    return "INT16";
  if (s.find("INT8") != std::string::npos)
    return "INT8";
  return {};
}

static int read_dim_from_cfg_file(const fs::path& cfg_path, const std::vector<const char*>& keys) {
  std::ifstream in(cfg_path);
  if (!in.is_open())
    return -1;
  json j;
  try {
    in >> j;
  } catch (const std::exception&) {
    return -1;
  }
  for (const char* key : keys) {
    const int v = json_first_int(j, key, -1);
    if (v > 0)
      return v;
  }
  return -1;
}

static bool normalize_terminal_mla_metadata(const std::string& etc_dir, const SequenceSplit& split,
                                            const std::vector<SequenceEntry>& infer_seq) {
  if (infer_seq.empty())
    return false;
  const auto& terminal = infer_seq.back();
  if (to_upper(terminal.processor) != "MLA")
    return false;

  const fs::path cfg_path = fs::path(etc_dir) / terminal.config_path;
  std::ifstream in(cfg_path);
  if (!in.is_open()) {
    std::ostringstream msg;
    msg << "Inference terminal MLA config not found: " << cfg_path.string();
    throw std::runtime_error(msg.str());
  }

  json cfg;
  try {
    in >> cfg;
  } catch (const std::exception& e) {
    std::ostringstream msg;
    msg << "Failed parsing terminal MLA config '" << cfg_path.string() << "': " << e.what();
    throw std::runtime_error(msg.str());
  }

  int out_w = json_first_int(cfg, "output_width", -1);
  if (out_w <= 0)
    out_w = json_first_int(cfg, "slice_width", -1);
  int out_h = json_first_int(cfg, "output_height", -1);
  if (out_h <= 0)
    out_h = json_first_int(cfg, "slice_height", -1);
  int out_d = json_first_int(cfg, "output_depth", -1);
  if (out_d <= 0)
    out_d = json_first_int(cfg, "slice_depth", -1);
  std::string dtype = normalize_dtype_string(json_first_string(cfg, "data_type"));

  // If MLA metadata is incomplete, infer spatial dims from upstream stage configs.
  if (out_w <= 0 || out_h <= 0) {
    auto infer_upstream_cfg_dims = [&](const SequenceEntry& e) {
      const fs::path p = fs::path(etc_dir) / e.config_path;
      if (out_w <= 0) {
        out_w = read_dim_from_cfg_file(
            p, {"output_width", "slice_width", "input_width", "scaled_width", "tile_width"});
      }
      if (out_h <= 0) {
        out_h = read_dim_from_cfg_file(
            p, {"output_height", "slice_height", "input_height", "scaled_height", "tile_height"});
      }
    };

    if (infer_seq.size() >= 2) {
      infer_upstream_cfg_dims(infer_seq[infer_seq.size() - 2]);
    }
    if ((out_w <= 0 || out_h <= 0) && !split.pre.empty()) {
      infer_upstream_cfg_dims(split.pre.back());
    }
    if (out_w <= 0) {
      out_w = read_dim_from_cfg_file(fs::path(etc_dir) / "0_preproc.json",
                                     {"output_width", "scaled_width", "input_width"});
    }
    if (out_h <= 0) {
      out_h = read_dim_from_cfg_file(fs::path(etc_dir) / "0_preproc.json",
                                     {"output_height", "scaled_height", "input_height"});
    }
  }

  int64_t out_size = -1;
  if (cfg.contains("simaai__params") && cfg["simaai__params"].is_object()) {
    const auto& params = cfg["simaai__params"];
    if (params.contains("outputs") && params["outputs"].is_array() && !params["outputs"].empty() &&
        params["outputs"][0].is_object()) {
      out_size = json_first_i64(params["outputs"][0], "size", -1);
    }
  }

  const int64_t pixels =
      (out_w > 0 && out_h > 0) ? static_cast<int64_t>(out_w) * static_cast<int64_t>(out_h) : -1;
  const int64_t bytes_per_pixel =
      (out_size > 0 && pixels > 0 && (out_size % pixels) == 0) ? (out_size / pixels) : -1;

  if (!dtype.empty() && out_d <= 0 && bytes_per_pixel > 0) {
    const int elem_bytes = dtype_bytes_from_string(dtype);
    if (elem_bytes > 0 && (bytes_per_pixel % elem_bytes) == 0) {
      const int64_t d = bytes_per_pixel / elem_bytes;
      if (d > 0 && d <= 4096)
        out_d = static_cast<int>(d);
    }
  }

  if (dtype.empty() && bytes_per_pixel > 0) {
    struct Candidate {
      const char* dtype;
      int bytes;
    };
    constexpr Candidate kCandidates[] = {{"INT32", 4}, {"INT16", 2}, {"INT8", 1}};
    for (const auto& c : kCandidates) {
      if ((bytes_per_pixel % c.bytes) != 0)
        continue;
      const int64_t d = bytes_per_pixel / c.bytes;
      if (d <= 0 || d > 4096)
        continue;
      dtype = c.dtype;
      if (out_d <= 0)
        out_d = static_cast<int>(d);
      break;
    }
  }

  // Fail fast instead of letting runtime stall on incomplete terminal metadata.
  if (out_w <= 0 || out_h <= 0 || out_d <= 0 || dtype.empty()) {
    std::ostringstream msg;
    msg << "Terminal MLA metadata incomplete after infer trimming"
        << " stage='" << terminal.name << "'"
        << " config='" << cfg_path.string() << "'"
        << " resolved={w=" << out_w << ",h=" << out_h << ",d=" << out_d
        << ",dtype=" << (dtype.empty() ? "<empty>" : dtype) << ",size_bytes=" << out_size << "}";
    throw std::runtime_error(msg.str());
  }

  bool changed = false;
  auto set_array_i = [&](const char* key, int v) {
    const json want = json::array({v});
    if (!cfg.contains(key) || cfg[key] != want) {
      cfg[key] = want;
      changed = true;
    }
  };
  auto set_array_s = [&](const char* key, const std::string& v) {
    const json want = json::array({v});
    if (!cfg.contains(key) || cfg[key] != want) {
      cfg[key] = want;
      changed = true;
    }
  };

  set_array_i("output_width", out_w);
  set_array_i("output_height", out_h);
  set_array_i("output_depth", out_d);
  set_array_i("slice_width", out_w);
  set_array_i("slice_height", out_h);
  set_array_i("slice_depth", out_d);
  set_array_s("data_type", dtype);

  if (!changed)
    return false;

  std::ofstream out(cfg_path);
  if (!out.is_open()) {
    std::ostringstream msg;
    msg << "Failed to write terminal MLA metadata config: " << cfg_path.string();
    throw std::runtime_error(msg.str());
  }
  out << cfg.dump(4);

  std::cerr << "[ModelPack] normalized terminal MLA metadata stage=" << terminal.name
            << " w=" << out_w << " h=" << out_h << " d=" << out_d << " dtype=" << dtype
            << " size_bytes=" << out_size << "\n";
  return true;
}

static bool caps_param_exists(const json& params, const std::string& name) {
  if (!params.is_array())
    return false;
  for (const auto& p : params) {
    if (!p.is_object())
      continue;
    if (p.value("name", "") == name)
      return true;
  }
  return false;
}

static void ensure_caps_param(json& params, const char* name, const char* type, const json& values,
                              const char* json_field = nullptr) {
  if (!params.is_array())
    params = json::array();
  if (caps_param_exists(params, name))
    return;
  json p;
  p["name"] = name;
  p["type"] = type;
  if (json_field) {
    p["json_field"] = json_field;
  } else {
    p["json_field"] = nullptr;
  }
  p["values"] = values;
  params.push_back(std::move(p));
}

[[maybe_unused]] static void patch_detessdequant_caps(const std::string& etc_dir) {
  const fs::path seq_path = fs::path(etc_dir) / "pipeline_sequence.json";
  std::ifstream seq_in(seq_path);
  if (!seq_in.is_open())
    return;
  json seq;
  seq_in >> seq;

  if (!seq.contains("pipelines") || !seq["pipelines"].is_array())
    return;
  for (auto& pipe : seq["pipelines"]) {
    if (!pipe.contains("sequence") || !pipe["sequence"].is_array())
      continue;
    for (auto& entry : pipe["sequence"]) {
      if (entry.value("kernel", "") != "detessdequant")
        continue;
      const std::string cfg_rel = entry.value("configPath", "");
      if (cfg_rel.empty())
        continue;
      const fs::path cfg_path = fs::path(etc_dir) / cfg_rel;
      std::ifstream cfg_in(cfg_path);
      if (!cfg_in.is_open())
        continue;
      json cfg;
      cfg_in >> cfg;

      const char* w_field = cfg.contains("slice_width")
                                ? "slice_width"
                                : (cfg.contains("output_width")
                                       ? "output_width"
                                       : (cfg.contains("input_width") ? "input_width" : nullptr));
      const char* h_field = cfg.contains("slice_height")
                                ? "slice_height"
                                : (cfg.contains("output_height")
                                       ? "output_height"
                                       : (cfg.contains("input_height") ? "input_height" : nullptr));
      const char* d_field = cfg.contains("slice_depth")
                                ? "slice_depth"
                                : (cfg.contains("output_depth")
                                       ? "output_depth"
                                       : (cfg.contains("input_depth") ? "input_depth" : nullptr));

      if (!cfg.contains("caps") || !cfg["caps"].is_object()) {
        cfg["caps"] = json::object();
      }
      json& caps = cfg["caps"];

      auto patch_pad = [&](const char* key, const char* fmt_value) {
        if (!caps.contains(key) || !caps[key].is_array() || caps[key].empty()) {
          caps[key] = json::array(
              {{{"media_type", "application/vnd.simaai.tensor"}, {"params", json::array()}}});
        }
        json& pad0 = caps[key][0];
        if (!pad0.contains("media_type")) {
          pad0["media_type"] = "application/vnd.simaai.tensor";
        }
        if (!pad0.contains("params") || !pad0["params"].is_array()) {
          pad0["params"] = json::array();
        }
        json& params = pad0["params"];
        if (fmt_value && *fmt_value) {
          ensure_caps_param(params, "format", "string", fmt_value);
        }
        if (w_field) {
          ensure_caps_param(params, "width", "int", "1 - 4096", w_field);
        }
        if (h_field) {
          ensure_caps_param(params, "height", "int", "1 - 4096", h_field);
        }
        if (d_field) {
          ensure_caps_param(params, "depth", "int", "1 - 4096", d_field);
        }
      };

      patch_pad("sink_pads", "MLA");
      patch_pad("src_pads", "DETESSDEQUANT");

      std::ofstream cfg_out(cfg_path);
      if (cfg_out.is_open()) {
        cfg_out << cfg.dump(4);
      }
    }
  }
}

static std::tuple<bool, int, int, int> rewrite_preproc_to_quanttess(const std::string& etc_dir) {
  const fs::path seq_path = fs::path(etc_dir) / "pipeline_sequence.json";
  std::ifstream in(seq_path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open " + seq_path.string());
  }

  json j;
  in >> j;
  in.close();

  if (!j.contains("pipelines") || !j["pipelines"].is_array() || j["pipelines"].empty()) {
    throw std::runtime_error("pipeline_sequence.json: missing/empty 'pipelines'");
  }

  json& pipe = j["pipelines"][0];
  if (!pipe.contains("sequence") || !pipe["sequence"].is_array()) {
    throw std::runtime_error("pipeline_sequence.json: missing 'sequence'");
  }

  auto& seq = pipe["sequence"];

  int preproc_idx = -1;
  int quanttess_idx = -1;
  std::string old_name;
  int old_seq_id = 0;
  for (int i = 0; i < (int)seq.size(); ++i) {
    auto& s = seq[i];
    if (s.value("kernel", "") == "preproc") {
      preproc_idx = i;
      old_name = s.value("name", "neatprocesspreproc_1");
      old_seq_id = s.value("sequence_id", i + 1);
      break;
    }
    if (quanttess_idx < 0 && s.value("kernel", "") == "quanttess") {
      quanttess_idx = i;
    }
  }
  if (preproc_idx < 0 && quanttess_idx < 0) {
    throw std::runtime_error("Could not find preproc on first segment");
  }

  const fs::path quanttess_path = fs::path(etc_dir) / "0_quanttess.json";
  if (!fs::exists(quanttess_path)) {
    throw std::runtime_error("INT8 path requested but 0_quanttess.json not found at " +
                             quanttess_path.string());
  }

  json qj;
  {
    std::ifstream qin(quanttess_path);
    if (!qin.is_open())
      throw std::runtime_error("Cannot open " + quanttess_path.string());
    qj = json::parse(qin, nullptr, /*allow_exceptions=*/true);
  }

  if (qj.contains("input_buffers") && qj["input_buffers"].is_array() &&
      !qj["input_buffers"].empty()) {
    qj["input_buffers"][0]["name"] = kDefaultPreviousNodeName;
  }

  {
    std::ofstream qout(quanttess_path);
    qout << qj.dump(4);
  }

  const int out_w = qj.value("input_width", 0);
  const int out_h = qj.value("input_height", 0);
  const int out_d = qj.value("input_depth", 0);
  if (out_w <= 0 || out_h <= 0 || out_d <= 0) {
    throw std::runtime_error("Invalid input dims in 0_quanttess.json");
  }

  if (preproc_idx < 0) {
    return {false, out_w, out_h, out_d};
  }

  json new_stage = {
      {"configPath", "0_quanttess.json"},  {"executable", nullptr},
      {"input", kDefaultPreviousNodeName}, {"kernel", "quanttess"},
      {"name", "neatprocessquanttess_1"},  {"params", json::object()},
      {"pluginId", "processcvu"},          {"processor", "CVU"},
      {"sequence_id", old_seq_id},
  };

  seq[preproc_idx] = new_stage;

  const std::string new_name = "neatprocessquanttess_1";
  for (auto& s : seq) {
    if (!s.contains("input"))
      continue;
    auto& inp = s["input"];
    if (inp.is_string()) {
      if (inp.get<std::string>() == old_name)
        inp = new_name;
    } else if (inp.is_array()) {
      for (auto& e : inp) {
        if (e.is_string() && e.get<std::string>() == old_name)
          e = new_name;
      }
    }
  }

  {
    std::ofstream out(seq_path);
    out << j.dump(4);
  }
  return {true, out_w, out_h, out_d};
}

static PipelineType get_pipeline_type(const std::string& folder_path,
                                      const std::string& input_format) {
  if (input_format.empty()) {
    fs::path mla_cfg_path;
    try {
      const auto sequence = load_pipeline_sequence(folder_path);
      for (const auto& stage : sequence) {
        if (to_lower(stage.processor) == "mla" || to_lower(stage.plugin_id) == "processmla") {
          mla_cfg_path = fs::path(folder_path) / stage.config_path;
          break;
        }
      }
    } catch (const std::exception&) {
      // Keep legacy fallback below for malformed or absent pipeline_sequence.json.
    }
    if (mla_cfg_path.empty()) {
      mla_cfg_path = fs::path(folder_path) / "0_process_mla.json";
    }

    std::ifstream in(mla_cfg_path);
    if (!in.is_open())
      throw std::runtime_error("Cannot open " + mla_cfg_path.string());

    json j;
    in >> j;
    in.close();

    if (!j.contains("data_type") || !j["data_type"].is_array() || j["data_type"].empty() ||
        !j["data_type"][0].is_string()) {
      throw std::runtime_error(
          "0_process_mla.json: 'data_type' must be an array with a string element");
    }

    const std::string data_type_raw = j["data_type"][0].get<std::string>();
    const std::string data_type_norm = normalize_dtype_string(data_type_raw);
    const std::string data_type_up = to_upper(data_type_raw);

    // BF16 inference paths require explicit cast handling.
    if (data_type_up == "EVXX_BFLOAT16" || data_type_up == "BFLOAT16" || data_type_up == "BF16") {
      return PipelineType::CastTess;
    }

    // QuantTess is an optional front-end stage for INT8-like paths. The MLA dtype
    // itself can still be non-INT8 on valid packs, in which case we keep preproc mode.
    if (data_type_norm == "INT8") {
      const fs::path quant_cfg = fs::path(folder_path) / "0_quanttess.json";
      if (fs::exists(quant_cfg)) {
        return PipelineType::QuantTess;
      }
    }

    // For INT16/INT32 (and other non-BF16 types we can carry in tensor specs),
    // use regular preproc-driven model flow.
    return PipelineType::Preproc;
  }

  return PipelineType::Preproc;
}

// Sequence parsing moved to mpk/PipelineSequence.{h,cpp}

static std::string find_config_by_substr(const std::string& etc_dir, const std::string& needle) {
  if (needle.empty())
    return "";
  const std::string want = to_lower(needle);
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(etc_dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path();
    if (p.extension() != ".json")
      continue;
    const std::string name = to_lower(p.filename().string());
    if (name.find(want) != std::string::npos) {
      return p.string();
    }
  }
  return "";
}

static std::pair<int, int> mla_range(const std::vector<SequenceEntry>& seq) {
  int first = -1;
  int last = -1;
  for (size_t i = 0; i < seq.size(); ++i) {
    if (seq[i].processor == "MLA") {
      if (first < 0)
        first = static_cast<int>(i);
      last = static_cast<int>(i);
    }
  }
  return {first, last};
}

static bool has_terminal_policy(const InferenceTerminalPolicy& policy) {
  return policy.mla_only || policy.last_stage_index.has_value() ||
         policy.last_stage_name.has_value() || policy.last_plugin_id.has_value() ||
         policy.last_processor.has_value();
}

static std::string infer_stage_summary(const std::vector<SequenceEntry>& seq) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    if (i)
      oss << ", ";
    oss << "#" << i << "{name=" << seq[i].name << ",plugin=" << seq[i].plugin_id
        << ",kernel=" << seq[i].kernel << ",processor=" << seq[i].processor << "}";
  }
  return oss.str();
}

static std::size_t resolve_terminal_index_or_throw(const std::vector<SequenceEntry>& infer_seq,
                                                   const InferenceTerminalPolicy& policy) {
  if (infer_seq.empty()) {
    throw std::runtime_error(
        "Inference terminal policy cannot resolve terminal stage: infer block is empty");
  }

  if (policy.last_stage_index.has_value()) {
    const std::size_t idx = *policy.last_stage_index;
    if (idx >= infer_seq.size()) {
      std::ostringstream msg;
      msg << "Inference terminal policy index out of range: requested=" << idx
          << " infer_size=" << infer_seq.size();
      throw std::runtime_error(msg.str());
    }
    return idx;
  }

  if (policy.last_stage_name.has_value()) {
    const std::string want = *policy.last_stage_name;
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (infer_seq[i].name == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by name='" << want
        << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.last_plugin_id.has_value()) {
    const std::string want = to_lower(*policy.last_plugin_id);
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (to_lower(infer_seq[i].plugin_id) == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by plugin='"
        << *policy.last_plugin_id << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.last_processor.has_value()) {
    const std::string want = to_lower(*policy.last_processor);
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (to_lower(infer_seq[i].processor) == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by processor='"
        << *policy.last_processor << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.mla_only) {
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (to_upper(infer_seq[i].processor) == "MLA")
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve last MLA stage"
        << " infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  throw std::runtime_error(
      "Inference terminal policy requested but no terminal selector was provided");
}

static void validate_infer_sequence_or_throw(const std::string& etc_dir,
                                             const std::vector<SequenceEntry>& infer_seq) {
  if (infer_seq.empty()) {
    throw std::runtime_error("Inference block is empty after terminal policy application");
  }
  for (std::size_t i = 0; i < infer_seq.size(); ++i) {
    const auto& e = infer_seq[i];
    if (e.name.empty()) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << ": empty stage name";
      throw std::runtime_error(msg.str());
    }
    if (e.plugin_id.empty()) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << " name='" << e.name << "': empty plugin_id";
      throw std::runtime_error(msg.str());
    }
    if (e.config_path.empty()) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << " name='" << e.name << "': empty config_path";
      throw std::runtime_error(msg.str());
    }
    const fs::path cfg = fs::path(etc_dir) / e.config_path;
    if (!fs::exists(cfg)) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << " name='" << e.name
          << "': config not found: " << cfg.string();
      throw std::runtime_error(msg.str());
    }
  }
}

static std::vector<SequenceEntry> select_stage(const std::vector<SequenceEntry>& seq,
                                               ModelStage stage) {
  if (stage == ModelStage::Full)
    return seq;
  auto [first_mla, last_mla] = mla_range(seq);

  std::vector<SequenceEntry> out;
  if (stage == ModelStage::MlaOnly) {
    for (const auto& e : seq) {
      if (e.processor == "MLA")
        out.push_back(e);
    }
    return out;
  }

  if (first_mla < 0 || last_mla < 0) {
    return out;
  }

  if (stage == ModelStage::Preprocess) {
    for (int i = 0; i < first_mla; ++i)
      out.push_back(seq[static_cast<size_t>(i)]);
    return out;
  }

  if (stage == ModelStage::Postprocess) {
    for (int i = last_mla + 1; i < static_cast<int>(seq.size()); ++i) {
      out.push_back(seq[static_cast<size_t>(i)]);
    }
  }
  return out;
}

static const char* stage_label(ModelStage stage) {
  switch (stage) {
  case ModelStage::Preprocess:
    return "preprocess";
  case ModelStage::MlaOnly:
    return "mla_only";
  case ModelStage::Postprocess:
    return "postprocess";
  case ModelStage::Full:
    return "full";
  }
  return "full";
}

static std::string upstream_name_for_stage(const std::vector<SequenceEntry>& seq,
                                           ModelStage stage) {
  if (seq.empty())
    return kDefaultPreviousNodeName;
  auto [first_mla, last_mla] = mla_range(seq);
  if (stage == ModelStage::MlaOnly) {
    if (first_mla > 0 && first_mla <= static_cast<int>(seq.size())) {
      const std::string& name = seq[static_cast<size_t>(first_mla - 1)].name;
      if (!name.empty())
        return name;
    }
    return kDefaultPreviousNodeName;
  }
  if (stage == ModelStage::Postprocess) {
    if (last_mla >= 0 && last_mla < static_cast<int>(seq.size())) {
      const std::string& name = seq[static_cast<size_t>(last_mla)].name;
      if (!name.empty())
        return name;
    }
    return kDefaultPreviousNodeName;
  }
  return kDefaultPreviousNodeName;
}

static ModelFragment build_fragment_linear(const std::string& etc_dir,
                                           const std::vector<SequenceEntry>& seq,
                                           const std::string& initial_input_name,
                                           const std::string& queue_prefix, int num_buffers_cvu,
                                           int num_buffers_mla, const std::string& name_suffix) {
  ModelFragment frag;
  if (seq.empty())
    return frag;

  std::ostringstream pipelineStr;
  std::string previous_node_name =
      initial_input_name.empty() ? kDefaultPreviousNodeName : initial_input_name;

  (void)queue_prefix;
  int mla_next_cpu_override = -1;
  const bool has_mla_next_cpu = parse_mla_next_cpu_override(mla_next_cpu_override);

  for (size_t i = 0; i < seq.size(); ++i) {
    const auto& elem = seq[i];
    const std::string plugin = resolve_stage_factory(elem.plugin_id);
    const std::string base_name = elem.name;
    const std::string name = name_suffix.empty() ? base_name : (base_name + name_suffix);
    std::string config = (fs::path(etc_dir) / elem.config_path).string();
    const bool update_next_cpu = has_mla_next_cpu && elem.plugin_id == "processmla";

    if (!name_suffix.empty()) {
      const std::string tag = base_name + name_suffix;
      config = rewrite_node_and_input_names(config, tag, name, previous_node_name, update_next_cpu,
                                            mla_next_cpu_override);
    } else {
      if (update_next_cpu) {
        update_mla_next_cpu(config, mla_next_cpu_override);
      }
      update_input_buffers_name(config, previous_node_name);
    }

    if (i)
      pipelineStr << "! ";
    pipelineStr << plugin << " name=" << name << " stage-id=" << name << " config=" << config
                << " ";
    if (elem.plugin_id == "processcvu") {
      if (num_buffers_cvu > 0) {
        pipelineStr << " num-buffers=" << num_buffers_cvu << " ";
      }
    } else if (elem.plugin_id == "processmla") {
      pipelineStr << "multi-pipeline=true ";
      if (num_buffers_mla > 0) {
        pipelineStr << " num-buffers=" << num_buffers_mla << " ";
      }
    }

    frag.elements.push_back(name);
    frag.config_paths.push_back(config);
    previous_node_name = name;
  }

  frag.gst = pipelineStr.str();
  return frag;
}

} // namespace

ModelPack::ModelPack(const std::string& tar_gz) {
  init(tar_gz);
}

ModelPack::ModelPack(const std::string& tar_gz, const std::string& media_type,
                     const std::string& format, int width, int height, int depth, int max_width,
                     int max_height, int max_depth, bool normalize, std::vector<float> mean,
                     std::vector<float> stddev, const std::string& preproc_next_cpu,
                     const std::string& upstream_name, int num_buffers_cvu, int num_buffers_mla,
                     int queue_max_buffers, int64_t queue_max_time_ns,
                     const std::string& queue_leaky, const std::string& name_suffix,
                     const InferenceTerminalPolicy& terminal_policy) {
  Config cfg;
  cfg.normalize = normalize;
  cfg.mean = std::move(mean);
  cfg.stddev = std::move(stddev);
  cfg.input_width = width;
  cfg.input_height = height;
  cfg.input_depth = depth;
  cfg.max_input_width = max_width;
  cfg.max_input_height = max_height;
  cfg.max_input_depth = max_depth;
  cfg.preproc_next_cpu = preproc_next_cpu;
  if (!upstream_name.empty())
    cfg.upstream_name = upstream_name;
  cfg.num_buffers_cvu = num_buffers_cvu;
  cfg.num_buffers_mla = num_buffers_mla;
  cfg.queue_max_buffers = queue_max_buffers;
  cfg.queue_max_time_ns = queue_max_time_ns;
  cfg.queue_leaky = queue_leaky;
  cfg.name_suffix = name_suffix;
  cfg.terminal_policy = terminal_policy;

  if (!media_type.empty() && media_type != "video/x-raw" &&
      media_type != "application/vnd.simaai.tensor") {
    throw std::invalid_argument("ModelPack: unsupported media_type: " + media_type);
  }
  if (media_type == "application/vnd.simaai.tensor") {
    cfg.input_format.clear();
  } else {
    cfg.input_format = format;
  }
  init_from_config(tar_gz, std::move(cfg));
}

#if defined(SIMA_WITH_OPENCV)
ModelPack::ModelPack(const std::string& tar_gz, const cv::Mat& mat, int max_width, int max_height,
                     int max_depth, bool normalize, std::vector<float> mean,
                     std::vector<float> stddev, const std::string& preproc_next_cpu,
                     const std::string& upstream_name, int num_buffers_cvu, int num_buffers_mla,
                     int queue_max_buffers, int64_t queue_max_time_ns,
                     const std::string& queue_leaky, const std::string& name_suffix,
                     const InferenceTerminalPolicy& terminal_policy)
    : ModelPack(tar_gz, "video/x-raw", (mat.channels() == 1) ? "GRAY" : "BGR", mat.cols, mat.rows,
                mat.channels(), max_width, max_height, max_depth, normalize, std::move(mean),
                std::move(stddev), preproc_next_cpu, upstream_name, num_buffers_cvu,
                num_buffers_mla, queue_max_buffers, queue_max_time_ns, queue_leaky, name_suffix,
                terminal_policy) {}
#endif

ModelPack ModelPack::clone_with_buffers(int num_buffers_cvu, int num_buffers_mla) const {
  ModelPack out = *this;
  out.options_.num_buffers_cvu = num_buffers_cvu;
  out.options_.num_buffers_mla = num_buffers_mla;
  return out;
}

ModelPack ModelPack::clone_with_overrides(const std::string& upstream_name,
                                          const std::string& name_suffix) const {
  ModelPack out = *this;
  if (!upstream_name.empty()) {
    out.options_.upstream_name = upstream_name;
  }
  if (!name_suffix.empty()) {
    out.options_.name_suffix = name_suffix;
  }
  return out;
}

void ModelPack::init(const std::string& tar_gz) {
  Config cfg;
  init_from_config(tar_gz, std::move(cfg));
}

void ModelPack::init_from_config(const std::string& tar_gz, Config cfg) {
  options_ = std::move(cfg);

  if (options_.num_buffers_cvu != 4 || options_.num_buffers_mla != 4) {
    throw std::runtime_error(
        "ModelPack: num_buffers_cvu/num_buffers_mla must be 4 for model pipelines.");
  }

  std::string fmt = normalize_format(options_.input_format);
  options_.input_format = fmt;

  if (options_.max_input_width <= 0) {
    options_.max_input_width = (options_.input_width > 0) ? options_.input_width : 1920;
  }
  if (options_.max_input_height <= 0) {
    options_.max_input_height = (options_.input_height > 0) ? options_.input_height : 1080;
  }
  if (options_.max_input_depth <= 0) {
    options_.max_input_depth = (options_.input_depth > 0) ? options_.input_depth : 3;
  }

  if (options_.input_depth == 0 && !options_.input_format.empty()) {
    if (options_.input_format == "GRAY") {
      options_.input_depth = 1;
    } else if (options_.input_format == "RGB" || options_.input_format == "BGR") {
      options_.input_depth = 3;
    }
  }

  std::string extracted = extract_and_organize(tar_gz);
  etc_dir_ = (fs::path(extracted) / kDirConf).string();

  pipeline_type_ = get_pipeline_type(etc_dir_, options_.input_format);
  if (pipeline_type_ == PipelineType::CastTess) {
    throw std::runtime_error("ModelPack: BF16/CastTess is not supported yet");
  }

  int out_w = 0;
  int out_h = 0;
  int out_c = 0;
  std::string out_fmt;

  if (pipeline_type_ == PipelineType::QuantTess) {
    std::tie(std::ignore, out_w, out_h, out_c) = rewrite_preproc_to_quanttess(etc_dir_);
    patch_quanttess_caps_to_tensor_sink(etc_dir_);
  } else if (pipeline_type_ == PipelineType::Preproc) {
    std::array<float, 3> mean3 = materialize3(options_.mean, 0.0f);
    std::array<float, 3> std3 = materialize3(options_.stddev, 1.0f);
    auto [pw, ph, pformat] = read_and_update_preproc_json(
        etc_dir_, options_.input_width, options_.input_height, options_.input_format,
        options_.preproc_next_cpu, options_.normalize,
        options_.normalize ? std::vector<float>{mean3[0], mean3[1], mean3[2]}
                           : std::vector<float>{},
        options_.normalize ? std::vector<float>{std3[0], std3[1], std3[2]} : std::vector<float>{});
    // Do not mutate detessdequant caps here; detess config should remain as in the tar.
    out_w = pw;
    out_h = ph;
    out_fmt = normalize_format(pformat);
    if (!out_fmt.empty()) {
      out_c = (out_fmt == "GRAY") ? 1 : 3;
    }
  }

  if (out_w > 0)
    options_.input_width = out_w;
  if (out_h > 0)
    options_.input_height = out_h;
  if (!out_fmt.empty())
    options_.input_format = out_fmt;
  if (out_c > 0)
    options_.input_depth = out_c;
  if (out_c > 0 && options_.input_format.empty()) {
    options_.input_format = (out_c == 1) ? "GRAY" : "RGB";
  }
  if (options_.input_depth == 0 && !options_.input_format.empty()) {
    options_.input_depth = (options_.input_format == "GRAY") ? 1 : 3;
  }
}

std::string ModelPack::find_config_path_by_plugin(const std::string& plugin_id) const {
  if (plugin_id.empty())
    return "";
  try {
    std::vector<SequenceEntry> seq = load_pipeline_sequence(etc_dir_);
    const std::string want = to_lower(plugin_id);
    for (const auto& entry : seq) {
      if (to_lower(entry.plugin_id) == want) {
        return (fs::path(etc_dir_) / entry.config_path).string();
      }
    }
  } catch (const std::exception&) {
    // Fall back to filename scan.
  }
  return find_config_by_substr(etc_dir_, plugin_id);
}

std::string ModelPack::find_config_path_by_processor(const std::string& processor) const {
  if (processor.empty())
    return "";
  try {
    std::vector<SequenceEntry> seq = load_pipeline_sequence(etc_dir_);
    const std::string want = to_lower(processor);
    for (const auto& entry : seq) {
      if (to_lower(entry.processor) == want) {
        return (fs::path(etc_dir_) / entry.config_path).string();
      }
    }
  } catch (const std::exception&) {
    // Fall back to filename scan.
  }
  return find_config_by_substr(etc_dir_, processor);
}

ModelFragment ModelPack::fragment(ModelStage stage) const {
  std::vector<SequenceEntry> seq = load_pipeline_sequence(etc_dir_);
  std::vector<SequenceEntry> sel = select_stage(seq, stage);
  if (sel.empty())
    return {};

  std::string upstream;
  if (stage == ModelStage::Preprocess || stage == ModelStage::Full) {
    upstream = options_.upstream_name.empty() ? upstream_name_for_stage(seq, stage)
                                              : options_.upstream_name;
  } else {
    upstream = upstream_name_for_stage(seq, stage);
  }
  const std::string queue_prefix = std::string("q_") + stage_label(stage) + "_";
  return build_fragment_linear(etc_dir_, sel, upstream, queue_prefix, options_.num_buffers_cvu,
                               options_.num_buffers_mla, options_.name_suffix);
}

std::string ModelPack::backend_fragment(ModelStage stage) const {
  return fragment(stage).gst;
}

NodeGroup ModelPack::to_node_group(ModelStage stage) const {
  ModelFragment frag = fragment(stage);
  if (frag.gst.empty())
    return NodeGroup{};
  const std::string label = stage_label(stage);
  return make_fragment_group(frag, label);
}

NodeGroup ModelPack::infer_block(const std::string& upstream_name) const {
  const SequenceSplit split = split_sequence();
  if (split.infer.empty()) {
    throw std::runtime_error("ModelPack::infer_block: pipeline has no infer stages");
  }
  std::vector<SequenceEntry> infer_seq = split.infer;

  if (has_terminal_policy(options_.terminal_policy)) {
    const std::size_t terminal_idx =
        resolve_terminal_index_or_throw(infer_seq, options_.terminal_policy);
    if (terminal_idx + 1 < infer_seq.size()) {
      std::ostringstream log;
      log << "[ModelPack] inference terminal stage index=" << terminal_idx
          << " name=" << infer_seq[terminal_idx].name
          << " plugin=" << infer_seq[terminal_idx].plugin_id
          << " processor=" << infer_seq[terminal_idx].processor
          << " dropped_tail=" << (infer_seq.size() - (terminal_idx + 1));
      std::cerr << log.str() << "\n";
    }
    infer_seq.resize(terminal_idx + 1);
    if (infer_seq.empty()) {
      throw std::runtime_error("Inference terminal policy removed all infer stages");
    }
  }
  validate_infer_sequence_or_throw(etc_dir_, infer_seq);

  // Keep terminal MLA stage self-consistent after infer truncation.
  // Without this, some models expose incomplete MLA output metadata and can
  // stall at runtime instead of failing early.
  if (has_terminal_policy(options_.terminal_policy)) {
    (void)normalize_terminal_mla_metadata(etc_dir_, split, infer_seq);
  }

  std::string upstream = upstream_name;
  if (upstream.empty()) {
    upstream = options_.upstream_name.empty() ? kDefaultPreviousNodeName : options_.upstream_name;
  }

  const std::string queue_prefix = "q_infer_";
  ModelFragment frag =
      build_fragment_linear(etc_dir_, infer_seq, upstream, queue_prefix, options_.num_buffers_cvu,
                            options_.num_buffers_mla, options_.name_suffix);
  if (frag.gst.empty())
    return NodeGroup{};
  return make_fragment_group(frag, "infer");
}

SequenceSplit ModelPack::split_sequence() const {
  return split_sequence_for_infer(load_pipeline_sequence(etc_dir_));
}

std::string ModelPack::apply_name_suffix(const std::string& base) const {
  if (options_.name_suffix.empty())
    return base;
  return base + options_.name_suffix;
}

InputOptions ModelPack::input_appsrc_options(bool tensor_mode) const {
  InputOptions opt;

  if (tensor_mode) {
    opt.media_type = "application/vnd.simaai.tensor";
    opt.format = "FP32";
    opt.max_width = options_.max_input_width;
    opt.max_height = options_.max_input_height;
    opt.max_depth = options_.max_input_depth;
    return opt;
  }

  std::string fmt = options_.input_format;
  if (fmt.empty())
    fmt = "RGB";
  if (fmt == "GRAY")
    fmt = "GRAY8";

  opt.media_type = "video/x-raw";
  opt.format = fmt;
  opt.max_width = options_.max_input_width;
  opt.max_height = options_.max_input_height;
  opt.max_depth = options_.max_input_depth;
  return opt;
}

} // namespace simaai::neat::internal

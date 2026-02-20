#include "nodes/sima/SimaRender.h"

#include "builder/ConfigJsonWire.h"
#include "gst/GstHelpers.h"
#include "pipeline/internal/TempJsonFileUtil.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace fs = std::filesystem;
using json = nlohmann::json;

struct SimaRender::ConfigHolder {
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
  return pipeline_internal::make_temp_json_path(dir, "sima_render", "SimaRender");
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
  write_json_file(j, path, "SimaRender");
  return path;
}

} // namespace

SimaRender::SimaRender(SimaRenderOptions opt) : opt_(std::move(opt)) {
  if (!opt_.config_path.empty()) {
    auto holder = std::make_shared<ConfigHolder>();
    holder->keep = true;
    holder->path = opt_.config_path;
    config_holder_ = std::move(holder);
    config_path_ = opt_.config_path;
  }
}

bool SimaRender::has_config_json() const {
  return !config_path_.empty();
}

std::string SimaRender::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatrender", "SimaRender::backend_fragment");
  ss << "neatrender name=n" << node_index << "_render";
  if (!config_path_.empty()) {
    ss << " config=\"" << config_path_ << "\"";
  }
  ss << " silent=" << (opt_.silent ? "true" : "false");
  ss << " emit-signals=" << (opt_.emit_signals ? "true" : "false");
  if (opt_.sima_allocator_type > 0) {
    ss << " sima-allocator-type=" << opt_.sima_allocator_type;
  }
  ss << " transmit=" << (opt_.transmit ? "true" : "false");
  return ss.str();
}

std::vector<std::string> SimaRender::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_render"};
}

OutputSpec SimaRender::output_spec(const OutputSpec& input) const {
  OutputSpec out = input;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatrender";
  return out;
}

bool SimaRender::wire_input_names(const std::vector<std::string>& upstream_names,
                                  const std::string& tag) {
  (void)tag;
  if (upstream_names.empty() || upstream_names[0].empty())
    return false;
  if (!config_holder_)
    return false;
  if (!config_holder_->has_config) {
    if (config_holder_->path.empty())
      return false;
    config_holder_->config = load_json_file(config_holder_->path, "SimaRender");
    config_holder_->has_config = true;
  }

  json cfg = config_holder_->config;
  if (!set_input_buffer_name_if_exists(cfg, upstream_names[0]))
    return false;

  if (config_holder_->keep) {
    config_path_ = write_json_temp(cfg, /*dir=*/"");
    config_holder_->keep = false;
  } else if (!config_path_.empty()) {
    write_json_file(cfg, config_path_, "SimaRender");
  } else {
    config_path_ = write_json_temp(cfg, /*dir=*/"");
  }

  config_holder_->config = std::move(cfg);
  config_holder_->path = config_path_;
  return true;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> SimaRender(SimaRenderOptions opt) {
  return std::make_shared<simaai::neat::SimaRender>(std::move(opt));
}

} // namespace simaai::neat::nodes

#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/NodeConfigHelpers.h"

#include "builder/ConfigJsonWire.h"
#include "gst/GstHelpers.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
using json = nlohmann::json;

struct SimaArgMax::ConfigHolder {
  std::string path;
  bool keep = false;
  int memfd = -1;
  json config;
  bool has_config = false;
  ~ConfigHolder() {
    if (memfd >= 0) {
      ::close(memfd);
      memfd = -1;
      return;
    }
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
};

SimaArgMax::SimaArgMax(SimaArgMaxOptions opt) : opt_(std::move(opt)) {
  if (!opt_.config_path.empty()) {
    auto holder = std::make_shared<ConfigHolder>();
    holder->keep = true;
    holder->path = opt_.config_path;
    config_holder_ = std::move(holder);
    config_path_ = opt_.config_path;
  }
}

bool SimaArgMax::has_config_json() const {
  return !config_path_.empty();
}

std::string SimaArgMax::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatargmax", "SimaArgMax::backend_fragment");
  ss << "neatargmax name=n" << node_index << "_argmax";
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

std::vector<std::string> SimaArgMax::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_argmax"};
}

OutputSpec SimaArgMax::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "ARGMAX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatargmax";
  return out;
}

bool SimaArgMax::wire_input_names(const std::vector<std::string>& upstream_names,
                                  const std::string& tag) {
  (void)tag;
  if (upstream_names.empty() || upstream_names[0].empty())
    return false;
  if (!config_holder_)
    return false;
  if (!config_holder_->has_config) {
    if (config_holder_->path.empty())
      return false;
    config_holder_->config = node_helpers::load_json_file(config_holder_->path, "SimaArgMax");
    config_holder_->has_config = true;
  }

  json cfg = config_holder_->config;
  if (!set_input_buffer_name_if_exists(cfg, upstream_names[0]))
    return false;

  if (config_holder_->keep) {
    config_path_ =
        node_helpers::write_json_memfd(cfg, &config_holder_->memfd, "SimaArgMax", "sima_argmax_cfg");
    config_holder_->keep = false;
  } else if (!config_path_.empty()) {
    if (config_holder_->memfd >= 0) {
      node_helpers::rewrite_json_memfd(config_holder_->memfd, cfg, "SimaArgMax");
    } else {
      node_helpers::write_json_file(cfg, config_path_, "SimaArgMax");
    }
  } else {
    config_path_ =
        node_helpers::write_json_memfd(cfg, &config_holder_->memfd, "SimaArgMax", "sima_argmax_cfg");
  }

  config_holder_->config = std::move(cfg);
  config_holder_->path = config_path_;
  return true;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> SimaArgMax(SimaArgMaxOptions opt) {
  return std::make_shared<simaai::neat::SimaArgMax>(std::move(opt));
}

} // namespace simaai::neat::nodes

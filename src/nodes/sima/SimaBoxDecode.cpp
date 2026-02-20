#include "nodes/sima/SimaBoxDecode.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat {
using pipeline_internal::lower_copy;

struct BoxDecodeOptionsInternal {
  std::string config_path;
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
  std::string decode_type;
  int top_k = 0;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int original_width = 0;
  int original_height = 0;
};

struct SimaBoxDecode::BoxDecodeConfigHolder {
  nlohmann::json config;
  bool has_config = false;
};

namespace {

std::string find_boxdecode_config_path(const simaai::neat::internal::ModelPack& model) {
  std::string path = model.find_config_path_by_plugin("boxdecode");
  if (path.empty())
    path = model.find_config_path_by_plugin("box_decode");
  return path;
}

std::string infer_decode_type_from_model_path(const simaai::neat::internal::ModelPack& model) {
  const std::string guess = lower_copy(model.etc_dir());
  if (guess.find("yolov8") != std::string::npos || guess.find("yolo_v8") != std::string::npos) {
    return "yolov8";
  }
  if (guess.find("yolov9") != std::string::npos || guess.find("yolo_v9") != std::string::npos) {
    return "yolov9";
  }
  return {};
}

nlohmann::json load_json_file(const std::string& path, const char* label) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  nlohmann::json j;
  in >> j;
  return j;
}

} // namespace

static BoxDecodeOptionsInternal options_from_model(const simaai::neat::internal::ModelPack& model) {
  BoxDecodeOptionsInternal opt;
  const std::string path = find_boxdecode_config_path(model);
  if (path.empty()) {
    throw std::runtime_error("SimaBoxDecode: failed to locate boxdecode config");
  }
  opt.config_path = path;
  opt.decode_type = infer_decode_type_from_model_path(model);
  return opt;
}

SimaBoxDecode::SimaBoxDecode(const simaai::neat::Model& model, const std::string& decode_type,
                             int original_width, int original_height, double detection_threshold,
                             double nms_iou_threshold, int top_k) {
  const auto& pack = simaai::neat::internal::ModelAccess::pack(model);
  auto opt = std::make_unique<BoxDecodeOptionsInternal>(options_from_model(pack));
  if (!decode_type.empty())
    opt->decode_type = decode_type;
  if (original_width > 0)
    opt->original_width = original_width;
  if (original_height > 0)
    opt->original_height = original_height;
  if (detection_threshold > 0.0)
    opt->detection_threshold = detection_threshold;
  if (nms_iou_threshold > 0.0)
    opt->nms_iou_threshold = nms_iou_threshold;
  if (top_k > 0)
    opt->top_k = top_k;

  config_path_ = opt->config_path;
  opt_ = std::move(opt);

  config_holder_ = std::make_shared<BoxDecodeConfigHolder>();
  try {
    config_holder_->config = load_json_file(config_path_, "SimaBoxDecode");
    config_holder_->has_config = config_holder_->config.is_object();
  } catch (...) {
    config_holder_->config = nlohmann::json::object();
    config_holder_->has_config = false;
  }
}

std::string SimaBoxDecode::backend_fragment(int node_index) const {
  std::ostringstream ss;
  require_element("neatboxdecode", "SimaBoxDecode::backend_fragment");
  const char* factory = "neatboxdecode";
  const std::string name = "n" + std::to_string(node_index) + "_boxdecode";
  ss << factory << " name=" << name << " stage-id=" << name;
  const std::string& cfg = config_path_.empty() ? opt_->config_path : config_path_;
  if (!cfg.empty()) {
    ss << " config=\"" << cfg << "\"";
  }
  ss << " silent=" << (opt_->silent ? "true" : "false");
  ss << " emit-signals=" << (opt_->emit_signals ? "true" : "false");
  if (opt_->sima_allocator_type > 0) {
    ss << " sima-allocator-type=" << opt_->sima_allocator_type;
  }
  if (!opt_->decode_type.empty()) {
    ss << " decode-type=\"" << opt_->decode_type << "\"";
  }
  if (opt_->detection_threshold > 0.0) {
    ss << " detection-threshold=" << opt_->detection_threshold;
  }
  if (opt_->nms_iou_threshold > 0.0) {
    ss << " nms-iou-threshold=" << opt_->nms_iou_threshold;
  }
  if (opt_->top_k > 0) {
    ss << " topk=" << opt_->top_k;
  }
  ss << " transmit=" << (opt_->transmit ? "true" : "false");
  return ss.str();
}

std::vector<std::string> SimaBoxDecode::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_boxdecode"};
}

bool SimaBoxDecode::has_config_json() const {
  return config_holder_ && config_holder_->has_config;
}

OutputSpec SimaBoxDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BBOX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "neatboxdecode";
  return out;
}

const nlohmann::json* SimaBoxDecode::config_json() const {
  if (!config_holder_ || !config_holder_->has_config)
    return nullptr;
  return &config_holder_->config;
}

bool SimaBoxDecode::wire_input_names(const std::vector<std::string>& upstream_names,
                                     const std::string& tag) {
  if (upstream_names.empty() || upstream_names[0].empty())
    return false;
  (void)tag;
  return false;
}

bool SimaBoxDecode::override_config_json(const std::function<void(nlohmann::json&)>& edit,
                                         const std::string& tag) {
  (void)tag;
  if (!config_holder_ || !config_holder_->has_config)
    return false;
  edit(config_holder_->config);
  return true;
}

void SimaBoxDecode::apply_upstream_config(const nlohmann::json& upstream, const std::string&) {
  (void)upstream;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> SimaBoxDecode(const simaai::neat::Model& model,
                                                  const std::string& decode_type,
                                                  int original_width, int original_height,
                                                  double detection_threshold,
                                                  double nms_iou_threshold, int top_k) {
  return std::make_shared<simaai::neat::SimaBoxDecode>(model, decode_type, original_width,
                                                       original_height, detection_threshold,
                                                       nms_iou_threshold, top_k);
}

} // namespace simaai::neat::nodes

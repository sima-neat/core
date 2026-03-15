#include "nodes/sima/SimaBoxDecode.h"

#include "gst/GstHelpers.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/TempJsonFileUtil.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat {
using pipeline_internal::lower_copy;
using json = nlohmann::json;

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
  int num_buffers = 0;
  int num_buffers_model = 0;
  bool num_buffers_locked = false;
};

struct SimaBoxDecode::BoxDecodeConfigHolder {
  std::string path;
  bool keep = false;
  json config;
  bool has_config = false;
  ~BoxDecodeConfigHolder() {
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
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

std::string make_temp_json_path(const std::string& dir) {
  return pipeline_internal::make_temp_json_path(dir, "sima_boxdecode", "SimaBoxDecode");
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
  write_json_file(j, path, "SimaBoxDecode");
  return path;
}

std::string effective_decode_type(const BoxDecodeOptionsInternal& opt, const json* cfg) {
  if (!opt.decode_type.empty())
    return lower_copy(opt.decode_type);
  if (cfg && cfg->is_object()) {
    const auto it = cfg->find("decode_type");
    if (it != cfg->end() && it->is_string()) {
      return lower_copy(it->get<std::string>());
    }
  }
  return {};
}

std::optional<double> effective_detection_threshold(const BoxDecodeOptionsInternal& opt,
                                                    const json* cfg) {
  if (opt.detection_threshold > 0.0)
    return opt.detection_threshold;
  if (cfg && cfg->is_object()) {
    const auto it = cfg->find("detection_threshold");
    if (it != cfg->end() && it->is_number()) {
      return it->get<double>();
    }
  }
  return std::nullopt;
}

void maybe_warn_yolov8_threshold_cliff(const BoxDecodeOptionsInternal& opt, const json* cfg) {
  const std::string decode_type = effective_decode_type(opt, cfg);
  const std::optional<double> threshold = effective_detection_threshold(opt, cfg);
  if (decode_type != "yolov8" || !threshold.has_value() || *threshold > 0.5) {
    return;
  }
  std::ostringstream msg;
  msg << std::fixed << std::setprecision(3);
  msg << "[WARN] SimaBoxDecode: resolved detection-threshold=" << *threshold
      << " for decode-type=yolov8. Thresholds <= 0.5 are risky for YOLOv8 cut-model outputs "
         "because they admit borderline 0-logit candidates and can cause severe latency cliffs "
         "before NMS/topk. Set an explicit threshold >= 0.51 (commonly 0.52+) in the app or "
         "model pack.\n";
  std::cerr << msg.str();
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
  opt.num_buffers_model = model.num_buffers_cvu();
  opt.num_buffers = opt.num_buffers_model;
  opt.num_buffers_locked = true;
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
    config_holder_->path = config_path_;
    config_holder_->keep = true;
  } catch (...) {
    config_holder_->config = json::object();
    config_holder_->has_config = false;
  }

  if (config_holder_->has_config &&
      (opt_->num_buffers > 0 || original_width > 0 || original_height > 0)) {
    override_config_json(
        [&](json& j) {
          if (opt_->num_buffers > 0) {
            if (!j.contains("system") || !j["system"].is_object()) {
              j["system"] = json::object();
            }
            j["system"]["out_buf_queue"] = opt_->num_buffers;
          }
          if (original_width > 0)
            j["original_width"] = original_width;
          if (original_height > 0)
            j["original_height"] = original_height;
        },
        "ctor.runtime_dims");
  }

  const json* effective_cfg =
      (config_holder_ && config_holder_->has_config) ? &config_holder_->config : nullptr;
  maybe_warn_yolov8_threshold_cliff(*opt_, effective_cfg);
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
  if (opt_->num_buffers > 0) {
    ss << " num-buffers=" << opt_->num_buffers;
  }
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
  json cfg = config_holder_->config;
  edit(cfg);
  config_path_ = write_json_temp(cfg, /*dir=*/"");
  auto holder = std::make_shared<BoxDecodeConfigHolder>();
  holder->config = std::move(cfg);
  holder->has_config = true;
  holder->path = config_path_;
  holder->keep = false;
  config_holder_ = std::move(holder);
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

#include "pipeline/Session.h"
#include "pipeline/StageRun.h"
#include "pipeline/TessellatedTensor.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::step_log;

enum class StageRoute {
  StageOnly,
  PreprocPipeline,
};

struct StageTestConfig {
  int iters = 1;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct PreprocWireInfo {
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string find_preproc_config_path(const simaai::neat::Model& model) {
  std::string path = model.find_config_path_by_plugin("process_cvu");
  if (path.empty())
    path = model.find_config_path_by_plugin("preproc");
  if (path.empty())
    path = model.find_config_path_by_processor("CVU");
  return path;
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

int read_first_int(const nlohmann::json& v) {
  if (v.is_number_integer())
    return v.get<int>();
  if (v.is_number())
    return static_cast<int>(v.get<double>());
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_number_integer())
        return entry.get<int>();
      if (entry.is_number())
        return static_cast<int>(entry.get<double>());
    }
  }
  return 0;
}

int read_int_field(const nlohmann::json& j, const char* key) {
  if (!j.contains(key))
    return 0;
  return read_first_int(j.at(key));
}

PreprocWireInfo read_preproc_wire_info(const simaai::neat::Model& model,
                                       const simaai::neat::Tensor& tensor, const cv::Mat& img) {
  PreprocWireInfo info;
  const std::string path = find_preproc_config_path(model);
  if (!path.empty()) {
    const nlohmann::json j = load_json_file(path, "yolov8_stage_route_test");
    if (j.contains("output_img_type") && j["output_img_type"].is_string()) {
      info.format = j["output_img_type"].get<std::string>();
    }
    info.width = read_int_field(j, "output_width");
    info.height = read_int_field(j, "output_height");
    info.depth = read_int_field(j, "output_channels");
    if (info.depth <= 0)
      info.depth = read_int_field(j, "tile_channels");
  }

  if (info.format.empty())
    info.format = "RGB";
  info.format = upper_copy(info.format);

  if (info.width <= 0 && tensor.shape.size() > 1) {
    info.width = static_cast<int>(tensor.shape[1]);
  }
  if (info.height <= 0 && tensor.shape.size() > 0) {
    info.height = static_cast<int>(tensor.shape[0]);
  }
  if (info.width <= 0)
    info.width = img.cols;
  if (info.height <= 0)
    info.height = img.rows;
  if (info.depth <= 0 && tensor.shape.size() >= 3) {
    info.depth = static_cast<int>(tensor.shape[2]);
  }
  if (info.depth <= 0)
    info.depth = img.channels();

  require(info.width > 0 && info.height > 0, "Preproc wire info missing width/height");
  return info;
}

StageRoute parse_route(int argc, char** argv, fs::path& root) {
  StageRoute route = StageRoute::StageOnly;
  bool root_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--route=", 0) == 0) {
      const std::string val = arg.substr(std::string("--route=").size());
      if (val == "stage") {
        route = StageRoute::StageOnly;
      } else if (val == "preproc-pipeline") {
        route = StageRoute::PreprocPipeline;
      } else {
        throw std::invalid_argument("Unknown route: " + val);
      }
      continue;
    }
    if (!root_set && !arg.empty() && arg[0] != '-') {
      root = fs::path(arg);
      root_set = true;
    }
  }
  if (!root_set)
    root = fs::current_path();
  return route;
}

std::string route_name(StageRoute route) {
  switch (route) {
  case StageRoute::StageOnly:
    return "stage";
  case StageRoute::PreprocPipeline:
    return "preproc-pipeline";
  }
  return "unknown";
}

std::vector<objdet::Box> to_objdet_boxes(const std::vector<simaai::neat::Box>& boxes) {
  std::vector<objdet::Box> out;
  out.reserve(boxes.size());
  for (const auto& b : boxes) {
    out.push_back({b.x1, b.y1, b.x2, b.y2, b.score, b.class_id});
  }
  return out;
}

std::string tensor_format(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    switch (tensor.semantic.image->format) {
    case simaai::neat::ImageSpec::PixelFormat::RGB:
      return "RGB";
    case simaai::neat::ImageSpec::PixelFormat::BGR:
      return "BGR";
    case simaai::neat::ImageSpec::PixelFormat::GRAY8:
      return "GRAY8";
    case simaai::neat::ImageSpec::PixelFormat::NV12:
      return "NV12";
    case simaai::neat::ImageSpec::PixelFormat::I420:
      return "I420";
    case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
      break;
    }
  }
  return "";
}

int tensor_width(const simaai::neat::Tensor& tensor) {
  return (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1;
}

int tensor_height(const simaai::neat::Tensor& tensor) {
  return (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1;
}

size_t tensor_dtype_bytes(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return 1;
  case simaai::neat::TensorDType::Int8:
    return 1;
  case simaai::neat::TensorDType::UInt16:
    return 2;
  case simaai::neat::TensorDType::Int16:
    return 2;
  case simaai::neat::TensorDType::Int32:
    return 4;
  case simaai::neat::TensorDType::BFloat16:
    return 2;
  case simaai::neat::TensorDType::Float32:
    return 4;
  case simaai::neat::TensorDType::Float64:
    return 8;
  }
  return 1;
}

bool is_int8_tensor(const simaai::neat::Tensor& tensor) {
  const std::string fmt = tensor_format(tensor);
  return simaai::neat::is_tessellated_int8_format(fmt) ||
         tensor.dtype == simaai::neat::TensorDType::Int8;
}

void run_stage_route_once(const cv::Mat& img_bgr, const simaai::neat::Model& model,
                          const StageTestConfig& cfg,
                          const simaai::neat::stages::BoxDecodeOptions& box_opt,
                          const std::vector<objdet::ExpectedBox>& expected, int iter) {
  step_log("stage_route: begin");
  auto pre = simaai::neat::stages::Preproc(img_bgr, model);
  require(is_int8_tensor(pre), "Preproc output is not INT8 tessellated");
  require(tensor_width(pre) > 0 && tensor_height(pre) > 0, "Preproc output missing width/height");
  require(pre.shape.size() >= 3, "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  auto infer = simaai::neat::stages::Infer(pre, model);
  require(is_int8_tensor(infer), "Infer output is not INT8 tessellated");
  require(tensor_width(infer) > 0 && tensor_height(infer) > 0, "Infer output missing width/height");
  require(infer.shape.size() >= 3, "Infer output missing shape dims");
  std::cout << "Infer passed\n" << std::endl;

  step_log("stage_route: before BoxDecode");
  const auto out = simaai::neat::stages::BoxDecode(infer, model, box_opt);
  std::cout << "BoxDecode passed\n" << std::endl;
  step_log("stage_route: after BoxDecode");
  const auto boxes = to_objdet_boxes(out.boxes);
  const objdet::MatchResult match =
      objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
  require(match.ok, "verify_mismatch iter=" + std::to_string(iter) + " " + match.note);
}

void run_preproc_pipeline_once(const cv::Mat& img_bgr, const simaai::neat::Model& model,
                               const StageTestConfig& cfg,
                               const simaai::neat::stages::BoxDecodeOptions& box_opt,
                               const std::vector<objdet::ExpectedBox>& expected, int iter) {
  step_log("preproc_pipeline: begin");
  auto pre = simaai::neat::stages::Preproc(img_bgr, model);
  require(is_int8_tensor(pre), "Preproc output is not INT8 tessellated");
  require(tensor_width(pre) > 0 && tensor_height(pre) > 0, "Preproc output missing width/height");
  require(pre.shape.size() >= 3, "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  const PreprocWireInfo wire_info = read_preproc_wire_info(model, pre, img_bgr);

  simaai::neat::Tensor wire = pre;
  wire.dtype = simaai::neat::TensorDType::UInt8;
  wire.semantic.tess.reset();
  {
    simaai::neat::ImageSpec image;
    image.format = simaai::neat::ImageSpec::PixelFormat::RGB;
    if (upper_copy(wire_info.format) == "BGR") {
      image.format = simaai::neat::ImageSpec::PixelFormat::BGR;
    }
    wire.semantic.image = image;
  }
  wire.shape = {wire_info.height, wire_info.width, wire_info.depth};
  wire.strides_bytes.clear();

  simaai::neat::InputOptions src_opt = model.input_appsrc_options(false);
  src_opt.media_type = "video/x-raw";
  src_opt.format = wire_info.format;
  src_opt.width = wire_info.width;
  src_opt.height = wire_info.height;
  src_opt.depth = -1;

  const int topk = (box_opt.top_k > 0) ? box_opt.top_k : 100;
  simaai::neat::Session p;
  p.add(simaai::neat::nodes::Input(src_opt));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", img_bgr.cols, img_bgr.rows,
                                           cfg.min_score, 0.5f, topk));
  p.add(simaai::neat::nodes::Output());

  step_log("preproc_pipeline: before p.run");
  const simaai::neat::Sample out = p.run(wire);
  step_log("preproc_pipeline: after p.run");
  std::vector<uint8_t> payload;
  std::string err;
  require(objdet::extract_bbox_payload(out, payload, err), err);

  const auto boxes = objdet::parse_boxes_strict(payload, img_bgr.cols, img_bgr.rows, topk, false);
  const objdet::MatchResult match =
      objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
  require(match.ok, "verify_mismatch iter=" + std::to_string(iter) + " " + match.note);
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path root;
    const StageRoute route = parse_route(argc, argv, root);
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    simaai::neat::Model::Options model_opt;
    model_opt.format = "BGR";
    model_opt.input_max_width = img_bgr.cols;
    model_opt.input_max_height = img_bgr.rows;
    model_opt.input_max_depth = 3;
    model_opt.preproc.normalize = false;
    model_opt.upstream_name = "decoder";
    auto model = simaai::neat::Model(tar_gz, model_opt);

    StageTestConfig cfg;

    simaai::neat::stages::BoxDecodeOptions box_opt;
    box_opt.decode_type = "yolov8";
    box_opt.original_width = img_bgr.cols;
    box_opt.original_height = img_bgr.rows;
    box_opt.detection_threshold = cfg.min_score;
    box_opt.nms_iou_threshold = 0.5f;
    box_opt.top_k = 100;

    const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

    for (int i = 0; i < cfg.iters; ++i) {
      if (route == StageRoute::StageOnly) {
        run_stage_route_once(img_bgr, model, cfg, box_opt, expected, i);
      } else {
        run_preproc_pipeline_once(img_bgr, model, cfg, box_opt, expected, i);
      }
    }

    std::cout << "[OK] yolov8_stage_route_test passed route=" << route_name(route) << "\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

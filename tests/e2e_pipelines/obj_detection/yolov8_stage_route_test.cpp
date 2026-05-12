#include "pipeline/Session.h"
#include "pipeline/StageRun.h"
#include "pipeline/TessellatedTensor.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/Preproc.h"
#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <filesystem>
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
  float boxdecode_score_threshold = 0.50f;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct PreprocWireInfo {
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

template <typename Fn>
auto run_with_report(simaai::neat::Run& runner, const std::string& label, Fn&& fn)
    -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::exception& e) {
    throw std::runtime_error(label + " failed: " + e.what() + "\n" + runner.report());
  }
}

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

PreprocWireInfo read_preproc_wire_info(const simaai::neat::Model& model,
                                       const simaai::neat::Tensor& tensor, const cv::Mat& img) {
  PreprocWireInfo info;
  const simaai::neat::PreprocOptions opt(model);
  info.format = opt.output_img_type;
  info.width = opt.output_width();
  info.height = opt.output_height();
  info.depth = (opt.output_channels() > 0) ? opt.output_channels() : opt.slice_channels();

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
  const simaai::neat::TensorList pre_tensors =
      simaai::neat::stages::Preproc(std::vector<cv::Mat>{img_bgr}, model);
  require(!pre_tensors.empty(), "Preproc output missing tensor outputs");
  require(is_int8_tensor(pre_tensors.front()), "Preproc output is not INT8 tessellated");
  require(tensor_width(pre_tensors.front()) > 0 && tensor_height(pre_tensors.front()) > 0,
          "Preproc output missing width/height");
  require(pre_tensors.front().shape.size() >= 3, "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  const simaai::neat::Sample pre = simaai::neat::sample_from_tensors(pre_tensors);
  const simaai::neat::SampleList infer_samples =
      simaai::neat::stages::Infer(simaai::neat::SampleList{pre}, model);
  require(infer_samples.size() == 1U, "Infer should return exactly one sample");
  const simaai::neat::Sample infer = infer_samples.front();
  const auto infer_tensors = simaai::neat::stages::Tensors(infer);
  require(!infer_tensors.empty(), "Infer output missing tensor outputs");
  require(is_int8_tensor(infer_tensors.front()), "Infer output is not INT8 tessellated");
  require(tensor_width(infer_tensors.front()) > 0 && tensor_height(infer_tensors.front()) > 0,
          "Infer output missing width/height");
  require(infer_tensors.front().shape.size() >= 3, "Infer output missing shape dims");
  std::cout << "Infer passed\n" << std::endl;

  step_log("stage_route: before BoxDecode");
  const simaai::neat::SampleList out_samples =
      simaai::neat::stages::BoxDecode(simaai::neat::SampleList{infer}, model, box_opt);
  require(out_samples.size() == 1U, "BoxDecode should return exactly one sample");
  const simaai::neat::Sample out = out_samples.front();
  std::cout << "BoxDecode passed\n" << std::endl;
  step_log("stage_route: after BoxDecode");
  std::vector<uint8_t> payload;
  std::string err;
  require(objdet::extract_bbox_payload(out, iter, payload, err), err);
  const auto boxes =
      objdet::parse_boxes_strict(payload, img_bgr.cols, img_bgr.rows, box_opt.top_k, false);
  const objdet::MatchResult match =
      objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
  require(match.ok, "verify_mismatch iter=" + std::to_string(iter) + " " + match.note);
}

void run_preproc_pipeline_once(const cv::Mat& img_bgr, const simaai::neat::Model& model,
                               const StageTestConfig& cfg,
                               const simaai::neat::stages::BoxDecodeOptions& box_opt,
                               const std::vector<objdet::ExpectedBox>& expected, int iter) {
  constexpr int kStageTimeoutMs = 30000;
  step_log("preproc_pipeline: begin");
  const simaai::neat::TensorList pre_tensors =
      simaai::neat::stages::Preproc(std::vector<cv::Mat>{img_bgr}, model);
  const simaai::neat::Sample wire = simaai::neat::sample_from_tensors(pre_tensors);
  std::cout << "[DBG] preproc wire kind=" << static_cast<int>(wire.kind)
            << " media_type=" << wire.media_type << " format=" << wire.format
            << " payload_tag=" << wire.payload_tag << "\n";
  require(!pre_tensors.empty(), "Preproc output missing tensor outputs");
  require(is_int8_tensor(pre_tensors.front()), "Preproc output is not INT8 tessellated");
  require(tensor_width(pre_tensors.front()) > 0 && tensor_height(pre_tensors.front()) > 0,
          "Preproc output missing width/height");
  require(pre_tensors.front().shape.size() >= 3, "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  const int topk = (box_opt.top_k > 0) ? box_opt.top_k : 100;
  simaai::neat::Session p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                           cfg.boxdecode_score_threshold, 0.5f, topk));
  p.add(simaai::neat::nodes::Output());

  step_log("preproc_pipeline: before p.run");
  auto runner = p.build(simaai::neat::SampleList{wire}, simaai::neat::RunMode::Sync);
  const simaai::neat::SampleList out_samples =
      run_with_report(runner, "preproc_pipeline p.run", [&]() {
        return runner.run(simaai::neat::SampleList{wire}, kStageTimeoutMs);
      });
  require(out_samples.size() == 1U, "preproc pipeline should return exactly one sample");
  const simaai::neat::Sample out = out_samples.front();
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

    StageTestConfig cfg;

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.upstream_name = "decoder";
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    model_opt.score_threshold = cfg.min_score;
    model_opt.nms_iou_threshold = 0.5f;
    model_opt.top_k = 100;
    auto model = simaai::neat::Model(tar_gz, model_opt);

    simaai::neat::stages::BoxDecodeOptions box_opt(simaai::neat::BoxDecodeType::YoloV8);
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

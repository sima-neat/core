#include "gst/GstHelpers.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/Session.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::append_note;
using sima_yolov8_test::step_log;

struct RunSummary {
  bool ok = false;
  int boxes = 0;
  std::string note;
  std::string pipeline;
  std::string pre_cfg;
  std::string mla_cfg;
  std::string box_cfg;
};

std::string resolve_pre_cfg(const simaai::neat::Model& model) {
  std::string path = model.find_config_path_by_plugin("process_cvu");
  if (path.empty())
    path = model.find_config_path_by_plugin("preproc");
  if (path.empty())
    path = model.find_config_path_by_processor("CVU");
  return path;
}

std::string runtime_pre_cfg_from_preprocess_group(const simaai::neat::Model& model,
                                                  std::shared_ptr<simaai::neat::Preproc>& keep_alive) {
  auto pre_group = model.preprocess();
  for (const auto& node : pre_group.nodes()) {
    auto pre = std::dynamic_pointer_cast<simaai::neat::Preproc>(node);
    if (pre) {
      keep_alive = std::move(pre);
      break;
    }
  }
  if (!keep_alive) {
    return {};
  }
  return keep_alive->config_path();
}

std::string resolve_mla_cfg(const simaai::neat::Model& model) {
  std::string path = model.find_config_path_by_plugin("processmla");
  if (path.empty())
    path = model.find_config_path_by_plugin("process_mla");
  if (path.empty())
    path = model.find_config_path_by_processor("MLA");
  return path;
}

std::string resolve_box_cfg(const simaai::neat::Model& model) {
  std::string path = model.find_config_path_by_plugin("boxdecode");
  if (path.empty())
    path = model.find_config_path_by_plugin("box_decode");
  return path;
}

RunSummary run_custom_neat_yolov8(const std::string& tar_gz, const cv::Mat& img) {
  RunSummary res;
  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  const int topk = 100;
  const float score = 0.52f;
  const float iou = 0.5f;

  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "BGR";
  model_opt.input_max_width = img.cols;
  model_opt.input_max_height = img.rows;
  model_opt.input_max_depth = 3;
  // Custom fragments bypass legacy JSON wiring; align upstream name with Input() element name.
  model_opt.upstream_name = "mysrc";
  auto model = simaai::neat::Model(tar_gz, model_opt);

  std::shared_ptr<simaai::neat::Preproc> pre_cfg_keep_alive;
  res.pre_cfg = runtime_pre_cfg_from_preprocess_group(model, pre_cfg_keep_alive);
  if (res.pre_cfg.empty()) {
    res.pre_cfg = resolve_pre_cfg(model);
  }
  res.mla_cfg = resolve_mla_cfg(model);
  res.box_cfg = resolve_box_cfg(model);
  require(!res.pre_cfg.empty(), "Failed to resolve preproc config from model");
  require(!res.mla_cfg.empty(), "Failed to resolve mla config from model");
  require(!res.box_cfg.empty(), "Failed to resolve boxdecode config from model");
  require(file_exists(res.pre_cfg), "Preproc config path missing: " + res.pre_cfg);
  require(file_exists(res.mla_cfg), "MLA config path missing: " + res.mla_cfg);
  require(file_exists(res.box_cfg), "Boxdecode config path missing: " + res.box_cfg);

  simaai::neat::Session p;
  p.add(simaai::neat::nodes::Input());
  p.custom("neatprocesscvu name=neatprocesspreproc_1 config=\"" +
           res.pre_cfg + "\" num-buffers=2");
  p.custom("neatprocessmla name=neatprocessmla_1 config=\"" +
           res.mla_cfg + "\" multi-pipeline=true num-buffers=2");
  p.custom("neatboxdecode name=n3_boxdecode_1 config=\"" + res.box_cfg +
           "\" silent=true emit-signals=false sima-allocator-type=2 decode-type=yolov8 "
           "detection-threshold=0.52 nms-iou-threshold=0.5 topk=100 transmit=false num-buffers=2");
  p.add(simaai::neat::nodes::Output());

  step_log("custom-neat: before build");
  (void)p.build(img, simaai::neat::RunMode::Sync);
  step_log("custom-neat: after build");
  const simaai::neat::Sample out = p.run(img);
  res.pipeline = p.last_pipeline();

  std::vector<uint8_t> payload;
  std::string err;
  if (!objdet::extract_bbox_payload(out, payload, err)) {
    append_note(res.note, err);
    return res;
  }
  const std::vector<objdet::Box> boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, topk, false);
  res.boxes = static_cast<int>(boxes.size());

  const auto expected = objdet::expected_people_boxes();
  const objdet::MatchResult match = objdet::match_expected_boxes(boxes, expected, score, 0.30f);
  if (!match.ok) {
    append_note(res.note, "verify_mismatch " + match.note);
    return res;
  }

  res.ok = true;
  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing NEAT preproc plugin (neatprocesscvu)");
    require(simaai::neat::element_exists("neatprocessmla"),
            "Missing NEAT MLA plugin (neatprocessmla)");
    require(simaai::neat::element_exists("neatboxdecode"),
            "Missing NEAT boxdecode plugin (neatboxdecode)");

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    RunSummary res = run_custom_neat_yolov8(tar_gz, img_bgr);
    std::cout << "NEAT_CUSTOM_YOLOV8 ok=" << (res.ok ? "1" : "0") << " boxes=" << res.boxes
              << " note=" << res.note << "\n";
    std::cout << "NEAT_CUSTOM_YOLOV8 cfg pre=" << res.pre_cfg << " mla=" << res.mla_cfg
              << " box=" << res.box_cfg << "\n";
    std::cout << "NEAT_CUSTOM_YOLOV8 pipeline\n" << res.pipeline << "\n";
    return res.ok ? 0 : 2;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

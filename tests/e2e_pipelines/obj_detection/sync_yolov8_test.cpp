/**
 * @example sync_yolov8_test.cpp
 * Canonical production pipeline: input -> preprocess -> Infer -> boxdecode.
 */
#include "pipeline/Graph.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sima_yolov8_test::append_note;
using sima_yolov8_test::env_int;
using sima_yolov8_test::sanitize_note;
using sima_yolov8_test::step_log;

struct SyncTestConfig {
  int iters = 20;
  float boxdecode_score_threshold = 0.49f;
  float min_score = 0.49f;
  float min_iou = 0.30f;
};

struct RunSummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

RunSummary run_yolov8_sync(const std::string& tar_gz, const cv::Mat& img,
                           const SyncTestConfig& cfg) {
  RunSummary res;

  require(!tar_gz.empty(), "Failed to locate yolo_v8s model archive");

  const int num_both = env_int("SIMA_SYNC_NUM_BUFFERS", -1);
  int num_cvu = env_int("SIMA_SYNC_NUM_BUFFERS_CVU", num_both);
  int num_mla = env_int("SIMA_SYNC_NUM_BUFFERS_MLA", num_both);
  const bool override_num = (num_cvu >= 0 || num_mla >= 0);
  if (override_num) {
    if (num_cvu < 0 || num_mla < 0) {
      append_note(res.note, "num_buffers_requires_both");
      return res;
    }
    if (!((num_cvu == 0 || num_cvu == 1) && (num_mla == 0 || num_mla == 1))) {
      append_note(res.note, "num_buffers_invalid");
      return res;
    }
  }

  (void)num_cvu;
  (void)num_mla;
  const int topk = 100;

  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  model_opt.score_threshold = cfg.boxdecode_score_threshold;
  model_opt.nms_iou_threshold = 0.5f;
  model_opt.top_k = topk;
  model_opt.upstream_name = "decoder";
  auto model = simaai::neat::Model(tar_gz, model_opt);

  // [canonical_pipeline]
  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                           cfg.boxdecode_score_threshold, 0.5f, topk));
  p.add(simaai::neat::nodes::Output());
  // [canonical_pipeline]

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

  step_log("sync: before build");
  auto run = p.build(
      simaai::neat::Sample{simaai::neat::Sample::from_image(
          img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
      simaai::neat::RunMode::Sync);
  step_log("sync: after build");

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    std::cout << "SYNC_YOLOV8 iter " << (i + 1) << "/" << cfg.iters << "\n";
    std::cout.flush();
    simaai::neat::Sample out;
    try {
      step_log("sync: before run");
      simaai::neat::Sample outs = run.run(
          simaai::neat::Sample{simaai::neat::Sample::from_image(
              img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
          30000);
      require(!outs.empty(), "sync run expected at least one sample");
      out = outs.front();
      step_log("sync: after run");
    } catch (const std::exception& e) {
      append_note(res.note, "run_error=" + sanitize_note(e.what()));
      break;
    }

    std::vector<uint8_t> payload;
    std::string err;
    if (!objdet::extract_bbox_payload(out, i, payload, err)) {
      append_note(res.note, err);
      break;
    }

    const auto boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, topk, false);
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected, cfg.min_score, cfg.min_iou);
    if (!match.ok) {
      append_note(res.note, "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
      break;
    }

    res.outputs += 1;
  }
  const auto end = std::chrono::steady_clock::now();

  res.diagnostics = p.last_pipeline();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.ok = (res.outputs == cfg.iters);
  if (elapsed_s <= 0.0) {
    append_note(res.note, "sync_timing_incomplete");
    res.ok = false;
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    SyncTestConfig cfg;
    RunSummary res = run_yolov8_sync(tar_gz, img_bgr, cfg);

    std::cout << "SYNC_YOLOV8 outputs=" << res.outputs << " avg_fps=" << res.avg_fps
              << " ok=" << (res.ok ? "1" : "0") << " note=" << res.note << "\n";
    if (!res.diagnostics.empty()) {
      std::cout << "SYNC_YOLOV8 diagnostics\n" << res.diagnostics << "\n";
    }

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

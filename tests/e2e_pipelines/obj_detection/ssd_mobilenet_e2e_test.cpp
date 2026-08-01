#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example ssd_mobilenet_e2e_test.cpp
 * On-device SSD-MobileNetV2-COCO end-to-end validation:
 *   input -> preprocess -> Infer -> SimaBoxDecode(Ssd) matches the golden person detections on
 *   the zidane COCO sample. Performance lives in perf_ssd_mobilenet_boxdecode_test so correctness
 *   failures and performance regressions have separate ownership and diagnostics.
 *
 * The decode runs the on-device SSD box-decode (neatobjectdecode) SSD-MobileNet variant: feature
 * maps {19,10,5,3,2,1} -> 1917 priors, anchor-major heads, sigmoid scoring. The preprocess resize
 * is STRETCH (this TF model was trained with fixed_shape_resizer), matching the SSD box
 * back-projection, so boxes map straight back to the original frame.
 */
#include "pipeline/Graph.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "model/Model.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/ssd_mobilenet_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void append_note(std::string& note, const std::string& part) {
  if (part.empty())
    return;
  if (!note.empty())
    note += ";";
  note += part;
}

std::string sanitize_note(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back((c == '\n' || c == '\r') ? '|' : c);
  return out;
}

struct AccuracyConfig : sima_ssd_mobilenet_test::ModelConfig {
  int iters = 20;
  float match_min_score = 0.50f; // people score ~0.88 on this image
  float match_min_iou = 0.45f;
};

struct AccuracySummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

AccuracySummary run_ssd_accuracy(const std::string& tar_gz, const cv::Mat& img,
                                 const AccuracyConfig& cfg) {
  AccuracySummary res;
  require(!tar_gz.empty(), "Failed to locate ssd_mobilenet model archive");

  auto model = simaai::neat::Model(tar_gz, sima_ssd_mobilenet_test::make_model_options(cfg));

  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::Ssd,
                                           cfg.score_threshold, cfg.nms_iou, cfg.top_k));
  p.add(simaai::neat::nodes::Output());

  const std::vector<objdet::ExpectedBox> expected =
      sima_ssd_mobilenet_test::expected_ssd_people_boxes();

  auto run = p.build_seeded_internal(
      simaai::neat::Sample{simaai::neat::Sample::from_image(
          img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
      simaai::neat::RunMode::Sync);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    simaai::neat::Sample out;
    try {
      simaai::neat::Sample outs = run.run(
          simaai::neat::Sample{simaai::neat::Sample::from_image(
              img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)},
          30000);
      require(!outs.empty(), "ssd sync run expected at least one sample");
      out = outs.front();
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

    const auto boxes = objdet::parse_boxes_strict(payload, img.cols, img.rows, cfg.top_k, false);
    const objdet::MatchResult match =
        objdet::match_expected_boxes(boxes, expected, cfg.match_min_score, cfg.match_min_iou);
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
  res.ok = (res.outputs == cfg.iters) && (elapsed_s > 0.0);
  if (elapsed_s <= 0.0)
    append_note(res.note, "sync_timing_incomplete");
  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_ssd_mobilenet_test::resolve_ssd_mobilenet_tar_or_skip(root);
    cv::Mat img_bgr = sima_ssd_mobilenet_test::load_coco_people_image_or_skip(root);

    const AccuracyConfig cfg;
    const AccuracySummary res = run_ssd_accuracy(tar_gz, img_bgr, cfg);
    std::cout << "SSD_MOBILENET_E2E outputs=" << res.outputs << " avg_fps=" << res.avg_fps
              << " ok=" << (res.ok ? "1" : "0") << " note=" << res.note << "\n";
    if (!res.diagnostics.empty())
      std::cout << "SSD_MOBILENET_E2E diagnostics\n" << res.diagnostics << "\n";
    if (!res.ok)
      return 2;

    std::cout << "[OK] ssd_mobilenet_e2e_test passed\n";
    return 0;
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

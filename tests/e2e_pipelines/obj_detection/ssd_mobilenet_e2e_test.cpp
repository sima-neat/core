#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example ssd_mobilenet_e2e_test.cpp
 * On-device SSD-MobileNetV2-COCO end-to-end validation:
 *   accuracy  -- input -> preprocess -> Infer -> SimaBoxDecode(Ssd) matches the golden person
 *                detections on the zidane COCO sample, and
 *   performance -- Model::benchmark() reports finite/positive latency/fps/power/energy.
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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kBenchmarkSamples = 100;

int env_int(const char* name, int def) {
  const char* val = std::getenv(name);
  return (val && *val) ? std::atoi(val) : def;
}

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

struct AccuracyConfig {
  int iters = 20;
  int top_k = 24;                          // must match the pack's baked topk
  float boxdecode_score_threshold = 0.30f; // on-device decode gate (matches the model config)
  float nms_iou = 0.60f;
  float match_min_score = 0.50f; // people score ~0.88 on this image
  float match_min_iou = 0.45f;
  int num_classes = 91; // background + 90 COCO classes
  int model_size = 300;
};

struct AccuracySummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

simaai::neat::Model::Options make_ssd_options(const AccuracyConfig& cfg) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  // STRETCH, not the default Letterbox: the TF model trains on a direct 300x300 resize and the SSD
  // box back-projection maps per-axis, so both sides must squash to the model input.
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.resize.mode = simaai::neat::ResizeMode::Stretch;
  opt.preprocess.resize.width = cfg.model_size;
  opt.preprocess.resize.height = cfg.model_size;
  // Model input range is [-1,1] = (pixel/127.5 - 1); the CVU computes (pixel/255 - mean)/stddev.
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.5f, 0.5f, 0.5f};
  opt.preprocess.normalize.stddev = {0.5f, 0.5f, 0.5f};
  opt.preprocess.normalize.has_explicit_stats = true;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.decode_type = simaai::neat::BoxDecodeType::Ssd;
  opt.num_classes = cfg.num_classes;
  opt.score_threshold = cfg.boxdecode_score_threshold;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.top_k;
  opt.upstream_name = "decoder";
  return opt;
}

AccuracySummary run_ssd_accuracy(const std::string& tar_gz, const cv::Mat& img,
                                 const AccuracyConfig& cfg) {
  AccuracySummary res;
  require(!tar_gz.empty(), "Failed to locate ssd_mobilenet model archive");

  auto model = simaai::neat::Model(tar_gz, make_ssd_options(cfg));

  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::Ssd,
                                           cfg.boxdecode_score_threshold, cfg.nms_iou, cfg.top_k));
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

void run_ssd_benchmark(const std::string& tar_gz) {
  simaai::neat::Model model(tar_gz);

  const auto specs = model.input_specs();
  require(!specs.empty(), "ssd benchmark: model.input_specs() returned no inputs");
  const int compiled_batch_size = model.compiled_batch_size();
  require(compiled_batch_size > 0, "ssd benchmark: compiled batch size must be positive");
  std::cout << "[e2e_benchmark] compiled_batch_size=" << compiled_batch_size << "\n";

  const int samples = env_int("SIMA_SSD_BENCHMARK_SAMPLES", kBenchmarkSamples);
  const simaai::neat::BenchmarkReport report = model.benchmark(samples);
  std::cout << "[e2e_benchmark] latency_ms=" << report.latency_ms << "\n";
  std::cout << "[e2e_benchmark] fps=" << report.fps << "\n";
  std::cout << "[e2e_benchmark] avg_power_watts=" << report.avg_power_watts << "\n";
  std::cout << "[e2e_benchmark] energy_joules=" << report.energy_joules << "\n";

  require(std::isfinite(report.latency_ms), "ssd benchmark: latency must be finite");
  require(std::isfinite(report.fps), "ssd benchmark: FPS must be finite");
  require(std::isfinite(report.avg_power_watts), "ssd benchmark: power must be finite");
  require(std::isfinite(report.energy_joules), "ssd benchmark: energy must be finite");
  require(report.latency_ms > 0.0, "ssd benchmark: latency must be positive");
  require(report.fps > 0.0, "ssd benchmark: FPS must be positive");
  require(report.avg_power_watts >= 0.0, "ssd benchmark: power must be non-negative");
  require(report.energy_joules >= 0.0, "ssd benchmark: energy must be non-negative");
}

// Times the decode-terminated graph, so latency includes the on-device box decode
// (run_ssd_benchmark times only the model forward pass).
void run_ssd_boxdecode_perf(const std::string& tar_gz, const cv::Mat& img,
                            const AccuracyConfig& cfg) {
  auto model = simaai::neat::Model(tar_gz, make_ssd_options(cfg));

  simaai::neat::Graph p;
  p.add(simaai::neat::nodes::Input());
  p.add(simaai::neat::nodes::groups::Preprocess(model));
  p.add(simaai::neat::nodes::groups::Infer(model));
  p.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::Ssd,
                                           cfg.boxdecode_score_threshold, cfg.nms_iou, cfg.top_k));
  p.add(simaai::neat::nodes::Output());

  auto make_sample = [&]() {
    return simaai::neat::Sample{simaai::neat::Sample::from_image(
        img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74)};
  };

  auto run = p.build_seeded_internal(make_sample(), simaai::neat::RunMode::Sync);

  const int warmup = 5;
  const int measured = env_int("SIMA_SSD_BOXDECODE_PERF_ITERS", 50);
  for (int i = 0; i < warmup; ++i) {
    simaai::neat::Sample outs = run.run(make_sample(), 30000);
    require(!outs.empty(), "ssd boxdecode perf warmup expected at least one sample");
  }

  const auto start = std::chrono::steady_clock::now();
  int completed = 0;
  for (int i = 0; i < measured; ++i) {
    simaai::neat::Sample outs = run.run(make_sample(), 30000);
    require(!outs.empty(), "ssd boxdecode perf run expected at least one sample");
    completed += 1;
  }
  const auto end = std::chrono::steady_clock::now();

  const double elapsed_s = std::chrono::duration<double>(end - start).count();
  const double mean_latency_ms = (completed > 0) ? (elapsed_s / completed) * 1000.0 : 0.0;
  const double decode_fps = (mean_latency_ms > 0.0) ? (1000.0 / mean_latency_ms) : 0.0;
  std::cout << "[boxdecode_perf] iters=" << completed << " mean_latency_ms=" << mean_latency_ms
            << " fps=" << decode_fps << "\n";

  require(completed == measured, "ssd boxdecode perf: not all iterations completed");
  require(std::isfinite(mean_latency_ms), "ssd boxdecode perf: latency must be finite");
  require(std::isfinite(decode_fps), "ssd boxdecode perf: fps must be finite");
  require(mean_latency_ms > 0.0, "ssd boxdecode perf: latency must be positive");
  require(decode_fps > 0.0, "ssd boxdecode perf: fps must be positive");
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

    run_ssd_boxdecode_perf(tar_gz, img_bgr, cfg);
    run_ssd_benchmark(tar_gz);

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

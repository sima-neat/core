#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example yolov8_dynamic_boxdecode_geometry_test.cpp
 * Build once, then run YOLOv8 boxdecode with changing input geometry.
 */
#include "model/Model.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Graph.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

static_assert(sizeof(RawBox) == 24, "BBOX payload layout changed");

struct DynamicGeometryConfig {
  float boxdecode_score_threshold = 0.49f;
  float min_score = 0.49f;
  float nms_iou_threshold = 0.5f;
  int topk = 100;
  int timeout_ms = 30000;
};

enum class DynamicRunMode {
  Both,
  Sync,
  Async,
};

struct CliArgs {
  fs::path root = fs::current_path();
  DynamicRunMode mode = DynamicRunMode::Both;
};

DynamicRunMode parse_mode_token(const std::string& token) {
  if (token == "both")
    return DynamicRunMode::Both;
  if (token == "sync")
    return DynamicRunMode::Sync;
  if (token == "async")
    return DynamicRunMode::Async;
  throw std::runtime_error("unsupported dynamic boxdecode mode: " + token);
}

std::string mode_name(DynamicRunMode mode) {
  switch (mode) {
  case DynamicRunMode::Both:
    return "both";
  case DynamicRunMode::Sync:
    return "sync";
  case DynamicRunMode::Async:
    return "async";
  }
  return "unknown";
}

CliArgs parse_cli(int argc, char** argv) {
  CliArgs out;
  bool root_seen = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i] ? argv[i] : "");
    if (arg == "--mode") {
      require((i + 1) < argc, "--mode requires sync, async, or both");
      out.mode = parse_mode_token(argv[++i]);
    } else if (arg.rfind("--mode=", 0) == 0) {
      out.mode = parse_mode_token(arg.substr(std::strlen("--mode=")));
    } else if (!root_seen) {
      out.root = fs::path(arg);
      root_seen = true;
    } else {
      throw std::runtime_error("unexpected argument: " + arg);
    }
  }
  return out;
}

cv::Mat make_half_resolution_frame(const cv::Mat& image) {
  require(image.cols > 64 && image.rows > 64,
          "dynamic boxdecode test image is too small for a geometry-change run");
  const int width = std::max(32, image.cols / 2);
  const int height = std::max(32, image.rows / 2);
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
  require(!resized.empty(), "failed to resize dynamic boxdecode test image");
  require(resized.cols != image.cols || resized.rows != image.rows,
          "dynamic boxdecode variant must change geometry");
  return resized;
}

void validate_raw_boxes(const std::vector<uint8_t>& bytes, const cv::Mat& image,
                        const DynamicGeometryConfig& cfg, const std::string& label) {
  require(bytes.size() >= sizeof(uint32_t), label + ": bbox buffer too small");

  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));
  const size_t payload_size = bytes.size() - sizeof(header);
  require(payload_size >= sizeof(RawBox), label + ": bbox payload too small");

  const size_t max_boxes = payload_size / sizeof(RawBox);
  require(header <= max_boxes, label + ": bbox header exceeds payload count");
  require(header <= static_cast<uint32_t>(cfg.topk), label + ": bbox header exceeds topk");

  const int tolerance_px = 4;
  const uint8_t* base = bytes.data() + sizeof(header);
  int valid_boxes = 0;
  float previous_score = std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < header; ++i) {
    RawBox box{};
    std::memcpy(&box, base + i * sizeof(RawBox), sizeof(box));

    require(std::isfinite(box.score), label + ": bbox score is not finite");
    require(box.score <= previous_score,
            label + ": bbox output is not globally confidence-descending");
    previous_score = box.score;
    if (box.score < cfg.min_score)
      continue;

    const int64_t x2 = static_cast<int64_t>(box.x) + static_cast<int64_t>(box.w);
    const int64_t y2 = static_cast<int64_t>(box.y) + static_cast<int64_t>(box.h);
    require(box.w > 0 && box.h > 0, label + ": bbox must have positive extent");
    require(box.x >= -tolerance_px && box.y >= -tolerance_px,
            label + ": bbox origin is outside the current frame");
    require(x2 <= static_cast<int64_t>(image.cols + tolerance_px) &&
                y2 <= static_cast<int64_t>(image.rows + tolerance_px),
            label + ": bbox extent is outside the current frame");
    ++valid_boxes;
  }

  require(valid_boxes > 0, label + ": expected at least one decoded box above threshold");
}

simaai::neat::Sample make_bgr_ev74_sample(const cv::Mat& frame) {
  return simaai::neat::Sample::from_image(frame, simaai::neat::ImageSpec::PixelFormat::BGR,
                                          simaai::neat::TensorMemory::EV74);
}

simaai::neat::Graph make_dynamic_boxdecode_graph(const simaai::neat::Model& model,
                                                 const DynamicGeometryConfig& cfg) {
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::Input());
  graph.add(simaai::neat::nodes::groups::Preprocess(model));
  graph.add(simaai::neat::nodes::groups::Infer(model));
  graph.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                               cfg.boxdecode_score_threshold, cfg.nms_iou_threshold,
                                               cfg.topk));
  graph.add(simaai::neat::nodes::Output());
  return graph;
}

void validate_boxdecode_output(const simaai::neat::Sample& outs, int output_index,
                               const cv::Mat& frame, const DynamicGeometryConfig& cfg,
                               const std::string& label) {
  require(!outs.empty(), label + ": expected at least one output sample");

  std::vector<uint8_t> payload;
  std::string err;
  require(objdet::extract_bbox_payload(outs.front(), output_index, payload, err), err);
  validate_raw_boxes(payload, frame, cfg, label);
}

struct RunResult {
  int outputs = 0;
  std::string pipeline;
};

RunResult run_sync_dynamic_geometry(const simaai::neat::Model& model,
                                    const DynamicGeometryConfig& cfg, const cv::Mat& original,
                                    const cv::Mat& half) {
  RunResult result;
  simaai::neat::Graph graph = make_dynamic_boxdecode_graph(model, cfg);
  sima_yolov8_test::step_log("dynamic-sync: before build");
  auto run =
      graph.build_seeded_internal(make_bgr_ev74_sample(original), simaai::neat::RunMode::Sync);
  sima_yolov8_test::step_log("dynamic-sync: after build");

  const auto run_frame = [&](const cv::Mat& frame, const std::string& label) {
    const std::string before_label = "dynamic-sync: before run " + label;
    sima_yolov8_test::step_log(before_label.c_str());
    simaai::neat::Sample outs = run.run(make_bgr_ev74_sample(frame), cfg.timeout_ms);
    const std::string after_label = "dynamic-sync: after run " + label;
    sima_yolov8_test::step_log(after_label.c_str());

    validate_boxdecode_output(outs, result.outputs, frame, cfg, "sync-" + label);
    ++result.outputs;
  };

  run_frame(original, "original");
  run_frame(half, "half");
  run_frame(original, "original-again");
  result.pipeline = graph.last_pipeline();
  run.close_input();
  run.close();
  return result;
}

RunResult run_async_dynamic_geometry(const simaai::neat::Model& model,
                                     const DynamicGeometryConfig& cfg, const cv::Mat& original,
                                     const cv::Mat& half) {
  RunResult result;
  simaai::neat::Graph graph = make_dynamic_boxdecode_graph(model, cfg);
  simaai::neat::RunOptions run_opt;
  run_opt.preset = simaai::neat::RunPreset::Realtime;

  sima_yolov8_test::step_log("dynamic-async: before build");
  auto run = graph.build(make_bgr_ev74_sample(original), run_opt);
  sima_yolov8_test::step_log("dynamic-async: after build");

  const auto push_pull_frame = [&](const cv::Mat& frame, const std::string& label) {
    const std::string before_label = "dynamic-async: before push " + label;
    sima_yolov8_test::step_log(before_label.c_str());
    require(run.push(make_bgr_ev74_sample(frame)), "async-" + label + ": push failed");
    simaai::neat::Sample outs = run.pull_samples(cfg.timeout_ms);
    const std::string after_label = "dynamic-async: after pull " + label;
    sima_yolov8_test::step_log(after_label.c_str());

    validate_boxdecode_output(outs, result.outputs, frame, cfg, "async-" + label);
    ++result.outputs;
  };

  push_pull_frame(original, "original");
  push_pull_frame(half, "half");
  push_pull_frame(original, "original-again");
  result.pipeline = graph.last_pipeline();
  run.close_input();
  run.close();
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs cli = parse_cli(argc, argv);
    const fs::path root = cli.root;
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    cv::Mat original = sima_yolov8_test::load_people_image_or_skip(root);
    cv::Mat half = make_half_resolution_frame(original);

    DynamicGeometryConfig cfg;

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    model_opt.score_threshold = cfg.boxdecode_score_threshold;
    model_opt.nms_iou_threshold = cfg.nms_iou_threshold;
    model_opt.top_k = cfg.topk;
    model_opt.upstream_name = "decoder";
    auto model = simaai::neat::Model(tar_gz, model_opt);

    RunResult sync_result;
    RunResult async_result;
    if (cli.mode == DynamicRunMode::Both || cli.mode == DynamicRunMode::Sync) {
      sync_result = run_sync_dynamic_geometry(model, cfg, original, half);
    }
    if (cli.mode == DynamicRunMode::Both || cli.mode == DynamicRunMode::Async) {
      async_result = run_async_dynamic_geometry(model, cfg, original, half);
    }

    std::cout << "DYNAMIC_BOXDECODE mode=" << mode_name(cli.mode)
              << " sync_outputs=" << sync_result.outputs
              << " async_outputs=" << async_result.outputs << " original=" << original.cols << "x"
              << original.rows << " half=" << half.cols << "x" << half.rows << " ok=1\n";
    if (!sync_result.pipeline.empty()) {
      std::cout << "DYNAMIC_BOXDECODE sync diagnostics\n" << sync_result.pipeline << "\n";
    }
    if (!async_result.pipeline.empty()) {
      std::cout << "DYNAMIC_BOXDECODE async diagnostics\n" << async_result.pipeline << "\n";
    }
    return 0;
  } catch (const SkipTest& e) {
    std::cerr << "[ERR] prerequisite unavailable: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

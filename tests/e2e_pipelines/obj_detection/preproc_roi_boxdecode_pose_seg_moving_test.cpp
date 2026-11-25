#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example preproc_roi_boxdecode_pose_seg_moving_test.cpp
 * One built Preproc ROI -> MLA -> SimaBoxDecode graph over an ROI that moves between frames, for a
 * pose model (SIMA_ROI_TEST_POSE) or a segmentation model.
 *
 * The same image patch sits at two positions of a canvas and two same-sized ROIs select it, so the
 * model sees identical pixels both times. Every box (and every keypoint) must move by exactly the
 * offset between the ROIs. Segmentation masks stay in model space: the model saw the same pixels,
 * so each mask must be byte-identical to its counterpart's.
 */
#include "model/Model.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/Graph.h"
#include "pipeline/StageRun.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

#ifdef SIMA_ROI_TEST_POSE
constexpr bool kPose = true;
constexpr const char* kTarEnv = "SIMA_POSE_TAR";
#else
constexpr bool kPose = false;
constexpr const char* kTarEnv = "SIMA_SEG_TAR";
#endif
constexpr int kTopK = 100;
constexpr int kTimeoutMs = 30000;
constexpr float kMinScore = 0.5f;
// Coordinate rounding of two ROI placements and the decoder.
constexpr float kTolerancePx = 3.0f;

std::vector<float> floats(const simaai::neat::Tensor& tensor) {
  const auto bytes = tensor.copy_payload_bytes();
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

struct Decoded {
  std::vector<float> boxes;        // [N, 6]: x1, y1, x2, y2, score, class
  std::vector<float> keypoints;    // [N, 17, 3] (pose)
  std::vector<std::uint8_t> masks; // [N, 160, 160] (segmentation)
  std::size_t count = 0;
};

Decoded decode(simaai::neat::Run& run, const simaai::neat::Sample& wire, const cv::Size& canvas,
               const std::string& label) {
  const simaai::neat::Sample out = run.run(simaai::neat::Sample{wire}, kTimeoutMs);
  require(out.size() == 1U, label + ": expected one output sample");
  const simaai::neat::Sample& sample = out.front();
  const simaai::neat::Tensor* tensor = nullptr;
  if (simaai::neat::sample_has_tensor_list(sample) && sample.tensors.size() == 1U)
    tensor = &sample.tensors.front();
  else if (sample.tensor.has_value())
    tensor = &sample.tensor.value();
  require(tensor != nullptr, label + ": expected one tensor");
  Decoded result;
  if (kPose) {
    const auto pose =
        simaai::neat::decode_pose_tensor(*tensor, canvas.width, canvas.height, kTopK, true);
    result.boxes = floats(pose.boxes);
    result.keypoints = floats(pose.keypoints);
  } else {
    const auto seg =
        simaai::neat::decode_segmentation_tensor(*tensor, canvas.width, canvas.height, kTopK, true);
    result.boxes = floats(seg.boxes);
    result.masks = seg.masks.copy_payload_bytes();
  }
  // Keep the confident detections; they come first, ordered by score.
  std::size_t n = 0;
  while ((n + 1) * 6 <= result.boxes.size() && result.boxes[n * 6 + 4] >= kMinScore)
    ++n;
  result.count = n;
  return result;
}

bool near(float a, float b, float shift) {
  return std::abs(a + shift - b) <= kTolerancePx;
}

// Every detection of `from` must have the same detection in `to`, shifted by (dx, dy).
void require_shifted(const Decoded& from, const Decoded& to, float dx, float dy,
                     const std::string& label) {
  require(from.count == to.count, label + ": detection count changed with the ROI");
  for (std::size_t i = 0; i < from.count; ++i) {
    const float* a = &from.boxes[i * 6];
    const float* b = &to.boxes[i * 6];
    require(near(a[0], b[0], dx) && near(a[1], b[1], dy) && near(a[2], b[2], dx) &&
                near(a[3], b[3], dy) && a[5] == b[5],
            label + ": box " + std::to_string(i) + " did not move by the ROI offset");
    if (kPose) {
      for (std::size_t k = 0; k < 17; ++k) {
        const float* p = &from.keypoints[(i * 17 + k) * 3];
        const float* q = &to.keypoints[(i * 17 + k) * 3];
        if (p[2] < kMinScore)
          continue;
        require(near(p[0], q[0], dx) && near(p[1], q[1], dy),
                label + ": keypoint " + std::to_string(k) + " did not move by the ROI offset");
      }
    } else {
      // The mask is in model space and the model saw the same pixels.
      constexpr std::size_t kMask = 160 * 160;
      require(from.masks.size() >= (i + 1) * kMask && to.masks.size() >= (i + 1) * kMask &&
                  std::memcmp(&from.masks[i * kMask], &to.masks[i * kMask], kMask) == 0,
              label + ": mask " + std::to_string(i) + " changed with the ROI");
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path root = argc > 1 && argv[1][0] != '-' ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const char* tar = std::getenv(kTarEnv);
    if (!tar || !*tar || !fs::exists(tar))
      skip_long_test_exception(std::string("Set ") + kTarEnv + " to the model archive");
    const cv::Mat patch = sima_yolov8_test::load_people_image_or_skip(root);

    simaai::neat::Model::Options opt;
    opt.preprocess.kind = simaai::neat::InputKind::Image;
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    opt.decode_type =
        kPose ? simaai::neat::BoxDecodeType::YoloV26Pose : simaai::neat::BoxDecodeType::YoloV26Seg;
    opt.score_threshold = kMinScore;
    opt.nms_iou_threshold = 0.5f;
    opt.top_k = kTopK;
    opt.upstream_name = "decoder";
    const simaai::neat::Model model(tar, opt);

    // The same patch at two offsets of one canvas.
    constexpr int kShiftX = 600, kShiftY = 300;
    const cv::Size canvas(kShiftX + patch.cols, kShiftY + patch.rows);
    require(canvas.width <= 1920 && canvas.height <= 1080,
            "the canvas exceeds the default dynamic image capacity");
    cv::Mat frame(canvas, patch.type(), cv::Scalar(114, 114, 114));
    patch.copyTo(frame(cv::Rect(0, 0, patch.cols, patch.rows)));
    cv::Mat moved_frame(canvas, patch.type(), cv::Scalar(114, 114, 114));
    patch.copyTo(moved_frame(cv::Rect(kShiftX, kShiftY, patch.cols, patch.rows)));
    const simaai::neat::PreprocessRoi first{0, 0, 0, patch.cols, patch.rows};
    const simaai::neat::PreprocessRoi second{0, kShiftX, kShiftY, patch.cols, patch.rows};

    const auto wire = [&](const cv::Mat& image, const simaai::neat::PreprocessRoi& roi) {
      const simaai::neat::TensorList pre = simaai::neat::stages::Preproc(
          std::vector<cv::Mat>{image}, model, std::vector<simaai::neat::PreprocessRoi>{roi});
      require(pre.size() == 1U, "Preproc must return one tensor");
      return simaai::neat::sample_from_tensors(pre);
    };
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input());
    graph.add(simaai::neat::nodes::groups::Infer(model));
    graph.add(simaai::neat::nodes::SimaBoxDecode(model, opt.decode_type, kMinScore, 0.5f, kTopK));
    graph.add(simaai::neat::nodes::Output());
    auto run = graph.build_seeded_internal(simaai::neat::Sample{wire(frame, first)},
                                           simaai::neat::RunMode::Sync);

    const auto at_first = decode(run, wire(frame, first), canvas, "first ROI");
    require(at_first.count > 0, "the first ROI produced no detections");
    const auto at_second = decode(run, wire(moved_frame, second), canvas, "second ROI");
    require_shifted(at_first, at_second, float(kShiftX), float(kShiftY), "second ROI");
    // Returning to the first ROI must not keep the second ROI's transform.
    const auto again = decode(run, wire(frame, first), canvas, "first ROI again");
    require_shifted(at_first, again, 0.0f, 0.0f, "first ROI again");
    run.close_input();
    run.close();
    std::cout << "[OK] preproc_roi_boxdecode_" << (kPose ? "pose" : "seg")
              << "_moving_test detections=" << at_first.count << "\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

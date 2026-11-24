#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/StageRun.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kCanvas = 640;

Model::Options letterbox_options() {
  Model::Options options;
  options.preprocess.resize.enable = AutoFlag::On;
  options.preprocess.resize.width = kCanvas;
  options.preprocess.resize.height = kCanvas;
  options.preprocess.resize.mode = ResizeMode::Letterbox;
  options.preprocess.resize.pad_value = 114;
  return options;
}

// Runs only the Model's preprocess stage (no MLA) and returns the signed INT8 output bytes.
std::vector<std::int8_t> run_preprocess(const std::string& tar_path, const Model::Options& options,
                                        const cv::Mat& frame) {
  const Model model(tar_path, options);
  const auto outputs = stages::Preproc(std::vector<cv::Mat>{frame}, model);
  require(outputs.size() == 1U, "preprocess must return one tensor");
  const auto bytes = outputs.front().copy_payload_bytes();
  require(bytes.size() == static_cast<std::size_t>(kCanvas) * kCanvas * 3U,
          "unexpected preprocess output size " + std::to_string(bytes.size()));
  return {bytes.begin(), bytes.end()};
}

} // namespace

// With normalization Off (explicit, or Auto resolving Off) the stage still quantizes with the
// pack's quantizer: pixel 200 with multiplier 0.5 and zero point 0 becomes INT8 100. With
// normalization On on the 255.07/-128 pack the letterbox pad 114 becomes INT8 -14.
RUN_TEST("model_preproc_normalize_off_quantize_regression_test", ([] {
           const auto half = sima_test::make_strict_model_archive_fixture_with_quantizer(
               "model_preproc_off_quantize_e2e_half", 0.5, 0);
           const cv::Mat uniform(kCanvas, kCanvas, CV_8UC3, cv::Scalar::all(200));

           auto off = letterbox_options();
           off.preprocess.normalize.enable = AutoFlag::Off;
           const auto require_all_100 = [&](const Model::Options& options,
                                            const std::string& label) {
             const auto bytes = run_preprocess(half.tar_path, options, uniform);
             require(std::all_of(bytes.begin(), bytes.end(),
                                 [](std::int8_t value) { return value == 100; }),
                     label + ": expected every INT8 byte to be 100");
           };
           require_all_100(off, "normalize Off");
           require_all_100(letterbox_options(), "normalize Auto");

           const auto strict = sima_test::make_strict_model_archive_fixture(
               "model_preproc_off_quantize_e2e_strict", {}, true, "yolo_v9c_seg",
               /*include_topology_artifacts=*/true);
           auto on = letterbox_options();
           on.preprocess.normalize.enable = AutoFlag::On;
           // A 640x320 frame leaves 160 padded rows above and below the content.
           const cv::Mat wide(kCanvas / 2, kCanvas, CV_8UC3, cv::Scalar::all(200));
           const auto bytes = run_preprocess(strict.tar_path, on, wide);
           const auto first_row_end = bytes.begin() + static_cast<std::ptrdiff_t>(kCanvas) * 3;
           require(std::all_of(bytes.begin(), first_row_end,
                               [](std::int8_t value) { return value == -14; }),
                   "normalize On: expected the first padded row to be INT8 -14");
         }));

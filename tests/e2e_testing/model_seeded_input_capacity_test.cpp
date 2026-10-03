#include "asset_utils.h"
#include "model_archive_test_utils.h"
#include "neat.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

Tensor image(int width, int height) {
  return Tensor::from_cv_mat(cv::Mat(height, width, CV_8UC3, cv::Scalar::all(17)),
                             ImageSpec::PixelFormat::BGR, TensorMemory::EV74);
}

template <typename Inputs>
void check_runner(Model::Runner& runner, const Inputs& seed, const Inputs& larger,
                  const Inputs& limit, const Inputs& oversized, const char* axis) {
  for (const auto* inputs : {&seed, &larger, &limit}) {
    require(!runner.run(*inputs, 30000).empty(), "in-capacity input must produce model output");
  }
  bool rejected = false;
  try {
    (void)runner.run(oversized, 30000);
  } catch (const NeatError& error) {
    rejected = true;
    require_contains(error.what(), axis, "oversize error must identify the exceeded axis");
    require_contains(error.what(), "exceeds configured capacity",
                     "oversize input must fail the capacity guard");
  }
  require(rejected, "input above the declared capacity must be rejected");
  runner.close();
}

} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: model_seeded_input_capacity_test <repo-root> <mode>");
    const std::string mode = argv[2];
    // Seedless build uses a CPU dummy; exercise its real device-visible inputs below.
    const sima_test::ScopedEnvVar preflight("SIMA_INPUTSTREAM_PREFLIGHT_RUN",
                                            mode == "unseeded" ? "0" : "1");
    const auto path = sima_test::resolve_yolov8s_tar(std::filesystem::path(argv[1]));
    require(!path.empty(), "missing model archive; set SIMA_MODEL_TAR or install yolo_v8s");

    Model::Options options;
    options.preprocess.kind = InputKind::Image;
    options.preprocess.enable = AutoFlag::On;
    options.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
    if (mode != "defaults") {
      options.preprocess.input_max_width = 1920;
      options.preprocess.input_max_height = 1920;
      options.preprocess.input_max_depth = 3;
    }
    Model model(path, options);
    const auto capacity = model.input_appsrc_options(false);
    const int max_width = capacity.max_width;
    const int max_height = capacity.max_height;
    require(max_width == 1920 && max_height == (mode == "defaults" ? 1080 : 1920),
            "unseeded model capacity must match the requested or default bounds");

    RunOptions run_options;
    run_options.preset = RunPreset::Realtime;

    if (mode == "oversize_width" || mode == "oversize_height") {
      const bool width = mode == "oversize_width";
      bool rejected = false;
      try {
        auto runner = model.build(
            TensorList{image(width ? max_width + 2 : 640, width ? 640 : max_height + 2)},
            Model::RouteOptions{}, run_options);
        runner.close();
      } catch (const NeatError& error) {
        rejected = true;
        require_contains(error.what(), width ? "width" : "height",
                         "oversize seed error must identify the exceeded axis");
        require_contains(error.what(), "capacity", "oversize seed must fail the capacity guard");
      }
      require(rejected, "an oversized seed must not expand the model capacity");
    } else if (mode == "mat") {
      const auto mats = [](int width, int height) {
        return std::vector<cv::Mat>{cv::Mat(height, width, CV_8UC3, cv::Scalar::all(17))};
      };
      const auto seed = mats(640, 640);
      auto runner = model.build(seed, Model::RouteOptions{}, run_options);
      check_runner(runner, seed, mats(800, 800), mats(max_width, max_height),
                   mats(max_width + 2, 640), "width");
    } else if (mode == "sample") {
      const auto samples = [](int width, int height) {
        return sample_from_tensors(TensorList{image(width, height)});
      };
      const auto seed = samples(640, 640);
      auto runner = model.build(seed, Model::RouteOptions{}, run_options);
      check_runner(runner, seed, samples(1280, 720), samples(max_width, max_height),
                   samples(640, max_height + 2), "height");
    } else {
      require(mode == "tensor" || mode == "defaults" || mode == "unseeded",
              "unknown test mode: " + mode);
      const TensorList seed{image(640, 640)};
      auto runner = mode == "unseeded" ? model.build(Model::RouteOptions{}, run_options)
                                       : model.build(seed, Model::RouteOptions{}, run_options);
      check_runner(runner, seed, TensorList{image(1280, 720)},
                   TensorList{image(max_width, max_height)}, TensorList{image(640, max_height + 2)},
                   "height");
    }
    std::cout << "[OK] seeded input capacity: " << mode << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] seeded input capacity: " << error.what() << '\n';
    return 1;
  }
}

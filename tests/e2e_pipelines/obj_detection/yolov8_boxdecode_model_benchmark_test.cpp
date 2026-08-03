#include "model/Model.h"

#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

simaai::neat::Model::Options boxdecode_model_options() {
  simaai::neat::Model::Options options;
  options.preprocess.kind = simaai::neat::InputKind::Tensor;
  options.preprocess.enable = simaai::neat::AutoFlag::Auto;
  options.preprocess.resize.enable = simaai::neat::AutoFlag::Off;
  options.preprocess.color_convert.enable = simaai::neat::AutoFlag::Off;
  options.preprocess.layout_convert.enable = simaai::neat::AutoFlag::Off;
  options.preprocess.normalize.enable = simaai::neat::AutoFlag::Off;
  options.preprocess.quantize.enable = simaai::neat::AutoFlag::Auto;
  options.preprocess.tessellate.enable = simaai::neat::AutoFlag::Auto;
  options.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  options.score_threshold = 0.52F;
  options.nms_iou_threshold = 0.5F;
  options.top_k = 100;
  options.upstream_name = "decoder";
  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    simaai::neat::Model model(tar_gz, boxdecode_model_options());
    require(model.info().selection.selected_post_kind == "boxdecode",
            "yolov8 BoxDecode benchmark: model route did not select BoxDecode");

    simaai::neat::BenchmarkOptions options;
    options.num_samples = sima_yolov8_test::env_int("SIMA_YOLOV8_BOXDECODE_BENCHMARK_SAMPLES", 1);
    options.original_width = 1280;
    options.original_height = 720;
    options.resize_mode = simaai::neat::ResizeMode::Letterbox;

    const simaai::neat::BenchmarkReport report = model.benchmark(options);
    std::cout << "[boxdecode_benchmark] latency_ms=" << report.latency_ms << "\n";
    std::cout << "[boxdecode_benchmark] fps=" << report.fps << "\n";

    require(std::isfinite(report.latency_ms), "yolov8 BoxDecode benchmark: latency must be finite");
    require(std::isfinite(report.fps), "yolov8 BoxDecode benchmark: FPS must be finite");
    require(report.latency_ms > 0.0, "yolov8 BoxDecode benchmark: latency must be positive");
    require(report.fps > 0.0, "yolov8 BoxDecode benchmark: FPS must be positive");

    std::cout << "[OK] yolov8_boxdecode_model_benchmark_test passed\n";
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

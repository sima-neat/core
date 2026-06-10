#include "yolov8_variant_route_matrix_common.inc"

#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <algorithm>
#include <string>
#include <vector>

#if !defined(SIMA_WITH_OPENCV)
RUN_TEST("graph_migration_phase3_connected_yolo_route_test",
         [] { skip_long_test_exception("OpenCV required for connected YOLO route coverage"); });
#else

namespace {

simaai::neat::RunOptions phase3_connected_yolo_run_options() {
  simaai::neat::RunOptions opt;
  opt.output_memory = simaai::neat::OutputMemory::Owned;
  opt.queue_depth = 1;
  return opt;
}

std::vector<fs::path> selected_models(const fs::path& root) {
  if (sima_yolov8_test::env_bool("SIMA_GRAPH_PHASE3_CONNECTED_YOLO_FULL_MATRIX", false)) {
    return graph_phase3_test::resolve_yolov8n_variants_or_throw(root);
  }
  return {
      graph_phase3_test::resolve_yolov8n_variant_or_throw(root, "yolov8n_A_W_INT8_MLATess.tar.gz"),
      graph_phase3_test::resolve_yolov8n_variant_or_throw(root,
                                                          "yolov8n_A_BF16_W_INT8_MLATess.tar.gz")};
}

simaai::neat::Sample run_connected_image_route(const cv::Mat& img_bgr, simaai::neat::Model& model) {
  simaai::neat::Graph input("image");
  input.add(simaai::neat::nodes::Input("image"));

  simaai::neat::Graph route("yolo");
  route.add(model);

  simaai::neat::Graph output("detections");
  output.add(simaai::neat::nodes::Output("detections"));

  simaai::neat::Graph app;
  app.connect(input, route);
  app.connect(route, output);

  simaai::neat::Run run =
      app.build(std::vector<cv::Mat>{img_bgr}, phase3_connected_yolo_run_options());
  require(static_cast<bool>(run), "connected YOLO route build failed");
  const std::vector<std::string> input_names = run.input_names();
  const std::vector<std::string> output_names = run.output_names();
  require(std::find(input_names.begin(), input_names.end(), "image") != input_names.end(),
          "connected YOLO route missing named input endpoint");
  require(std::find(output_names.begin(), output_names.end(), "detections") != output_names.end(),
          "connected YOLO route missing named output endpoint");
  require(run.push("image", std::vector<cv::Mat>{img_bgr}),
          "connected YOLO route named push failed");
  simaai::neat::Sample out = run.pull_samples("detections", default_model_run_timeout_ms());
  require(out.size() == 1U, "connected YOLO route expected one output sample");
  run.close();
  return out.front();
}

} // namespace

RUN_TEST("graph_migration_phase3_connected_yolo_route_test", [] {
  const fs::path root = graph_phase3_test::repo_root();
  const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);
  const std::vector<fs::path> models = selected_models(root);

  int passed = 0;
  for (const auto& tar : models) {
    simaai::neat::Model model(tar.string(),
                              graph_phase3_test::yolo_image_bgr_to_rgb_bf16_options());
    const simaai::neat::Sample sample = run_connected_image_route(img_bgr, model);
    require_preprocess_meta_on_output_local(sample, img_bgr.cols, img_bgr.rows,
                                            "phase3_connected_yolo_output");
    const AccuracyResult acc =
        run_framework_boxdecode_accuracy(sample, model, img_bgr, BoxDecodeRunMode::NoModel);
    require(acc.ok, "connected YOLO public Graph route accuracy failed for " +
                        tar.filename().string() + ": " + acc.note);
    std::cout << "GRAPH_PHASE3_CONNECTED_YOLO model=" << tar.filename().string()
              << " status=OK accuracy=\"" << acc.note << "\"\n";
    ++passed;
  }
  require(passed == static_cast<int>(models.size()),
          "connected YOLO route did not execute all models");
});
#endif

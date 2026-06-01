#include "yolov8_variant_route_matrix_common.inc"

#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#if !defined(SIMA_WITH_OPENCV)
RUN_TEST("graph_migration_phase3_yolov8_bgr_bf16_preproc_test", [] {
  skip_long_test_exception("OpenCV required for true BGR BF16 YOLO preproc coverage");
});
#else

namespace {

simaai::neat::RunOptions phase3_yolo_run_options() {
  simaai::neat::RunOptions opt;
  opt.output_memory = simaai::neat::OutputMemory::Owned;
  opt.queue_depth = 1;
  return opt;
}

} // namespace

RUN_TEST("graph_migration_phase3_yolov8_bgr_bf16_preproc_test", [] {
  const fs::path root = graph_phase3_test::repo_root();
  const fs::path tar = graph_phase3_test::resolve_yolov8n_variant_or_throw(
      root, "yolov8n_A_BF16_W_INT8_MLATess.tar.gz");
  const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

  simaai::neat::Model model(tar.string(), graph_phase3_test::yolo_image_bgr_to_rgb_bf16_options());

  simaai::neat::Graph g("yolo");
  g.add(simaai::neat::nodes::Input("image"));
  g.add(model);
  g.add(simaai::neat::nodes::Output("detections"));

  simaai::neat::Run run = g.build(std::vector<cv::Mat>{img_bgr}, simaai::neat::RunMode::Async,
                                  phase3_yolo_run_options());
  require(static_cast<bool>(run), "BGR input -> RGB BF16 Graph build failed");
  require(run.push("image", std::vector<cv::Mat>{img_bgr}),
          "BGR input -> RGB BF16 named push failed");

  const simaai::neat::Sample outputs =
      run.pull_samples("detections", default_model_run_timeout_ms());
  require(outputs.size() == 1U, "BGR input -> RGB BF16 expected one output sample");
  run.close();

  require_preprocess_meta_on_output_local(outputs.front(), img_bgr.cols, img_bgr.rows,
                                          "phase3_bgr_bf16_output");
  const AccuracyResult acc =
      run_framework_boxdecode_accuracy(outputs.front(), model, img_bgr, BoxDecodeRunMode::NoModel);
  require(acc.ok, "BGR input -> RGB BF16 accuracy failed: " + acc.note);
  std::cout << "GRAPH_PHASE3_BGR_BF16 model=" << tar.filename().string() << " status=OK accuracy=\""
            << acc.note << "\"\n";
});
#endif

#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/TensorOpenCV.h"
#include "test_main.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#if !defined(SIMA_WITH_OPENCV)
RUN_TEST("graph_migration_phase3_lazy_dynamic_spec_test",
         [] { skip_long_test_exception("OpenCV required for lazy dynamic spec model coverage"); });
#else
namespace {

simaai::neat::TensorList run_lazy_connected_model(simaai::neat::Model& model,
                                                  const cv::Mat& seedless_first_input,
                                                  const cv::Mat& second_input,
                                                  simaai::neat::ImageSpec::PixelFormat input_format,
                                                  const std::string& label) {
  simaai::neat::Graph input("image");
  input.add(simaai::neat::nodes::Input("image"));
  simaai::neat::Graph route("model");
  route.add(model);
  simaai::neat::Graph output("classes");
  output.add(simaai::neat::nodes::Output("classes"));

  simaai::neat::Graph app;
  app.connect(input, route);
  app.connect(route, output);

  const auto graph_inputs = app.inputs();
  const auto graph_outputs = app.outputs();
  require(graph_inputs.size() == 1U && graph_inputs.front() == "image",
          label + ": app.inputs() should expose only the external image input");
  require(graph_outputs.size() == 1U && graph_outputs.front() == "classes",
          label + ": app.outputs() should expose only the external classes output");

  simaai::neat::RunOptions opt;
  opt.output_memory = simaai::neat::OutputMemory::Owned;

  // Build intentionally has no seed; the first pushed sample must complete the input spec lazily.
  simaai::neat::Run run = app.build(opt);
  require(static_cast<bool>(run), label + ": seedless connected Graph build failed");
  const auto run_inputs = run.input_names();
  const auto run_outputs = run.output_names();
  require(run_inputs.size() == 1U && run_inputs.front() == "image",
          label + ": Run input_names() should expose only image");
  require(run_outputs.size() == 1U && run_outputs.front() == "classes",
          label + ": Run output_names() should expose only classes");
  require(
      run.push("image", simaai::neat::TensorList{simaai::neat::from_cv_mat(
                            seedless_first_input, input_format, simaai::neat::TensorMemory::EV74)}),
      label + ": first lazy push failed");
  simaai::neat::TensorList first = run.pull_tensors("classes", 20000);
  graph_phase3_test::require_nonempty_tensor_output(first, label + ": first lazy output");

  require(run.push("image", simaai::neat::TensorList{simaai::neat::from_cv_mat(
                                second_input, input_format, simaai::neat::TensorMemory::EV74)}),
          label + ": second lazy push failed");
  simaai::neat::TensorList second = run.pull_tensors("classes", 20000);
  graph_phase3_test::require_nonempty_tensor_output(second, label + ": second lazy output");
  const std::string report = run.report();
  require(report.find("incomplete input spec") == std::string::npos,
          label + ": report should not retain incomplete-input-spec failure after lazy build");
  run.close();
  return second;
}

} // namespace

RUN_TEST("graph_migration_phase3_lazy_dynamic_spec_test", [] {
  const std::filesystem::path root = graph_phase3_test::repo_root();

  {
    const std::filesystem::path resnet_tar = graph_phase3_test::resolve_resnet50_or_throw(root);
    simaai::neat::Model resnet(resnet_tar.string(),
                               graph_phase3_test::resnet_imagenet_bgr_options());
    const cv::Mat img0(224, 224, CV_8UC3, cv::Scalar(11, 47, 83));
    const cv::Mat img1(224, 224, CV_8UC3, cv::Scalar(17, 31, 127));
    (void)run_lazy_connected_model(resnet, img0, img1, simaai::neat::ImageSpec::PixelFormat::BGR,
                                   "ResNet lazy connected Graph");
  }

  {
    const std::filesystem::path mnist_tar = graph_phase3_test::resolve_mnist_or_throw(root);
    simaai::neat::Model mnist(mnist_tar.string(), graph_phase3_test::mnist_gray8_options());
    const cv::Mat digit0(28, 28, CV_8UC1, cv::Scalar(0));
    const cv::Mat digit1(28, 28, CV_8UC1, cv::Scalar(255));
    (void)run_lazy_connected_model(mnist, digit0, digit1,
                                   simaai::neat::ImageSpec::PixelFormat::GRAY8,
                                   "MNIST lazy connected Graph");
  }
});
#endif

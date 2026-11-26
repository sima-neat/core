#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
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

simaai::neat::Model::Options resnet_imagenet_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  opt.upstream_name = "decoder";
  return opt;
}

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
  const bool first_pushed =
      run.push("image", simaai::neat::TensorList{simaai::neat::from_cv_mat(
                            seedless_first_input, input_format, simaai::neat::TensorMemory::EV74)});
  require(first_pushed, label + ": first lazy push failed: " + run.last_error());
  simaai::neat::TensorList first = run.pull_tensors("classes", 20000);
  graph_phase3_test::require_nonempty_tensor_output(first, label + ": first lazy output");

  const bool second_pushed =
      run.push("image", simaai::neat::TensorList{simaai::neat::from_cv_mat(
                            second_input, input_format, simaai::neat::TensorMemory::EV74)});
  require(second_pushed, label + ": second lazy push failed: " + run.last_error());
  simaai::neat::TensorList second = run.pull_tensors("classes", 20000);
  graph_phase3_test::require_nonempty_tensor_output(second, label + ": second lazy output");
  const std::string last_error = run.last_error();
  require(last_error.find("incomplete input spec") == std::string::npos,
          label + ": last_error should not retain incomplete-input-spec failure after lazy build");
  run.close();
  return second;
}

} // namespace

RUN_TEST("graph_migration_phase3_lazy_dynamic_spec_test", [] {
  const std::filesystem::path root = graph_phase3_test::repo_root();
  const std::filesystem::path resnet_tar = graph_phase3_test::resolve_resnet50_or_throw(root);

  {
    simaai::neat::Model::Options options = resnet_imagenet_options();
    options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    simaai::neat::Model resnet(resnet_tar.string(), options);
    const cv::Mat img0(224, 224, CV_8UC3, cv::Scalar(11, 47, 83));
    const cv::Mat img1(224, 224, CV_8UC3, cv::Scalar(17, 31, 127));
    (void)run_lazy_connected_model(resnet, img0, img1, simaai::neat::ImageSpec::PixelFormat::BGR,
                                   "ResNet lazy connected Graph");
  }

  {
    // Single-channel ingress: the first GRAY8 push completes the input spec, and preprocessing
    // expands it to the RGB input the model expects.
    simaai::neat::Model::Options options = resnet_imagenet_options();
    options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::GRAY8;
    options.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
    simaai::neat::Model resnet(resnet_tar.string(), options);
    const cv::Mat gray0(224, 224, CV_8UC1, cv::Scalar(0));
    const cv::Mat gray1(224, 224, CV_8UC1, cv::Scalar(255));
    (void)run_lazy_connected_model(resnet, gray0, gray1,
                                   simaai::neat::ImageSpec::PixelFormat::GRAY8,
                                   "ResNet GRAY8 lazy connected Graph");
  }
});
#endif

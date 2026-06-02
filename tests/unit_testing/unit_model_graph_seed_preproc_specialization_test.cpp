#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorCore.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace {

sima_test::ModelArchiveFixture make_seed_preproc_fixture(const std::string& tag) {
  return sima_test::make_strict_model_archive_fixture(tag,
                                                      {
                                                          {"etc/pipeline_sequence.json",
                                                           R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      }
    ]
  }]
})json"},
                                                          {"etc/0_preproc.json",
                                                           R"json({
  "node_name": "preproc_0",
  "input_width": 640,
  "input_height": 640,
  "input_img_type": "RGB",
  "output_width": 224,
  "output_height": 224,
  "output_img_type": "RGB",
  "out_data_type": "INT8"
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [7],
  "output_height": [7],
  "output_depth": [16]
})json"},
                                                      },
                                                      true);
}

std::string image_format_token(simaai::neat::ImageSpec::PixelFormat format) {
  switch (format) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
  default:
    return "";
  }
}

simaai::neat::Tensor make_image_tensor(int width, int height,
                                       simaai::neat::ImageSpec::PixelFormat format) {
  const int channels = format == simaai::neat::ImageSpec::PixelFormat::GRAY8 ? 1 : 3;
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                            static_cast<std::size_t>(channels);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0U) {
    std::memset(map.data, 0x3a, map.size_bytes);
  }

  simaai::neat::Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.layout = channels == 1 ? simaai::neat::TensorLayout::HW : simaai::neat::TensorLayout::HWC;
  tensor.shape = channels == 1 ? std::vector<int64_t>{height, width}
                               : std::vector<int64_t>{height, width, channels};
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.read_only = true;
  tensor.semantic.image = simaai::neat::ImageSpec{format, ""};
  return tensor;
}

simaai::neat::Sample make_image_tensor_sample(int width, int height,
                                              simaai::neat::ImageSpec::PixelFormat format) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::TensorSet;
  sample.tensors = simaai::neat::TensorList{make_image_tensor(width, height, format)};
  sample.payload_type = simaai::neat::PayloadType::Image;
  sample.media_type = "video/x-raw";
  sample.payload_tag = image_format_token(format);
  sample.format = sample.payload_tag;
  return sample;
}

const simaai::neat::PreprocOptions&
require_model_preproc_options(const std::vector<std::shared_ptr<simaai::neat::Node>>& nodes,
                              const std::string& where) {
  for (const auto& node : nodes) {
    const auto* preproc = dynamic_cast<const simaai::neat::Preproc*>(node.get());
    if (!preproc) {
      continue;
    }
    const auto& opt = preproc->options();
    if (opt.model_managed_contract) {
      return opt;
    }
  }
  throw std::runtime_error(where + ": model-managed Preproc node not found");
}

const simaai::neat::PreprocOptions&
require_single_segment_preproc_options(const simaai::neat::runtime::ExecutionGraphPlan& plan,
                                       const std::string& where) {
  require(plan.pipeline_segments.size() == 1U, where +
                                                   ": expected one linear pipeline segment, got " +
                                                   std::to_string(plan.pipeline_segments.size()));
  return require_model_preproc_options(plan.pipeline_segments.front().nodes, where);
}

std::vector<simaai::neat::PreprocOptions>
collect_model_preproc_options(const simaai::neat::runtime::ExecutionGraphPlan& plan) {
  std::vector<simaai::neat::PreprocOptions> out;
  for (const auto& segment : plan.pipeline_segments) {
    for (const auto& node : segment.nodes) {
      const auto* preproc = dynamic_cast<const simaai::neat::Preproc*>(node.get());
      if (!preproc) {
        continue;
      }
      const auto& opt = preproc->options();
      if (opt.model_managed_contract) {
        out.push_back(opt);
      }
    }
  }
  return out;
}

bool contains_preproc_contract(const std::vector<simaai::neat::PreprocOptions>& options,
                               const std::string& input_type, std::vector<int> shape) {
  return std::any_of(options.begin(), options.end(), [&](const simaai::neat::PreprocOptions& opt) {
    return opt.input_img_type == input_type && opt.input_shape == shape;
  });
}

void require_same_preproc_seed_contract(const simaai::neat::PreprocOptions& expected,
                                        const simaai::neat::PreprocOptions& actual,
                                        const std::string& where) {
  require(actual.input_img_type == expected.input_img_type,
          where + ": input_img_type mismatch expected=" + expected.input_img_type +
              " actual=" + actual.input_img_type);
  require(actual.input_shape == expected.input_shape, where + ": input_shape mismatch");
  require(actual.output_img_type == expected.output_img_type,
          where + ": output_img_type mismatch expected=" + expected.output_img_type +
              " actual=" + actual.output_img_type);
  require(actual.output_shape == expected.output_shape, where + ": output_shape mismatch");
}

simaai::neat::Model::RouteOptions runnable_route_options() {
  simaai::neat::Model::RouteOptions opt;
  opt.include_input = true;
  opt.include_output = true;
  return opt;
}

simaai::neat::Model::Options image_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = 640;
  opt.preprocess.input_max_height = 640;
  opt.preprocess.input_max_depth = 3;
  return opt;
}

} // namespace

RUN_TEST(
    "unit_model_graph_seed_preproc_specialization_test", ([] {
      using namespace simaai::neat;

      const auto fixture = make_seed_preproc_fixture("model_graph_seed_preproc_specialize");
      Model model(fixture.tar_path, image_model_options());
      const Model::RouteOptions route_opt = runnable_route_options();

      const Sample bgr_seed = make_image_tensor_sample(321, 123, ImageSpec::PixelFormat::BGR);
      const auto expected_bgr_nodes =
          internal::ModelAccess::build_public_route_nodes_for_seed(model, route_opt, bgr_seed);
      const auto default_nodes = internal::ModelAccess::build_public_route_nodes(model, route_opt);

      const PreprocOptions& expected_bgr =
          require_model_preproc_options(expected_bgr_nodes, "Model::build(Sample) seed route");
      const PreprocOptions& default_preproc =
          require_model_preproc_options(default_nodes, "Model::graph() unseeded route");

      require(default_preproc.input_img_type == "RGB",
              "fixture should expose an unseeded RGB model-pack default");
      require(default_preproc.input_shape != expected_bgr.input_shape ||
                  default_preproc.input_img_type != expected_bgr.input_img_type,
              "BGR seed must differ from the unseeded model-pack default");
      require(expected_bgr.input_img_type == "BGR",
              "Model::build(Sample) seed route should honor explicit BGR");
      require(expected_bgr.input_shape == std::vector<int>({123, 321, 3}),
              "Model::build(Sample) seed route should honor BGR seed shape");

      Model missing_image_options_model(fixture.tar_path);
      Graph missing_image_options_graph = missing_image_options_model.graph(route_opt);
      try {
        (void)runtime::compile_public_graph(missing_image_options_graph, RunOptions{}, bgr_seed);
        throw std::runtime_error("expected raw image seed without model-managed Preproc to fail");
      } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        require_contains(msg, "no model-managed Preproc node",
                         "diagnostic should name missing model-managed Preproc");
        require_contains(msg, "preprocess.kind = InputKind::Image",
                         "diagnostic should explain how to enable image preprocess");
        require_contains(msg, "preprocessed tensors",
                         "diagnostic should mention the tensor-input alternative");
      }

      Graph seeded_graph = model.graph(route_opt);
      const runtime::ExecutionGraphPlan bgr_plan =
          runtime::compile_public_graph(seeded_graph, RunOptions{}, bgr_seed);
      const PreprocOptions& graph_bgr =
          require_single_segment_preproc_options(bgr_plan, "model.graph().build(Sample)");
      require_same_preproc_seed_contract(expected_bgr, graph_bgr,
                                         "Model::build(Sample) vs model.graph().build(Sample)");

      const Sample rgb_seed = make_image_tensor_sample(257, 199, ImageSpec::PixelFormat::RGB);
      const auto expected_rgb_nodes =
          internal::ModelAccess::build_public_route_nodes_for_seed(model, route_opt, rgb_seed);
      const PreprocOptions& expected_rgb =
          require_model_preproc_options(expected_rgb_nodes, "Model::build(Sample) RGB route");
      Graph rgb_graph = model.graph(route_opt);
      const runtime::ExecutionGraphPlan rgb_plan =
          runtime::compile_public_graph(rgb_graph, RunOptions{}, rgb_seed);
      const PreprocOptions& graph_rgb =
          require_single_segment_preproc_options(rgb_plan, "model.graph().build(RGB Sample)");
      require(expected_rgb.input_img_type == "RGB",
              "RGB seed should remain explicit RGB after specialization");
      require(expected_rgb.input_shape == std::vector<int>({199, 257, 3}),
              "RGB seed shape should be propagated to model-managed Preproc");
      require_same_preproc_seed_contract(expected_rgb, graph_rgb,
                                         "RGB Model::build(Sample) vs graph seed route");

#if defined(SIMA_WITH_OPENCV)
      cv::Mat image_bgr(77, 91, CV_8UC3, cv::Scalar(1, 2, 3));
      const Sample cv_seed =
          Sample::from_image(image_bgr, ImageSpec::PixelFormat::BGR, TensorMemory::CPU);
      const auto expected_cv_nodes =
          internal::ModelAccess::build_public_route_nodes_for_seed(model, route_opt, cv_seed);
      const PreprocOptions& expected_cv =
          require_model_preproc_options(expected_cv_nodes, "Model::build(cv::Mat) route");
      Graph cv_graph = model.graph(route_opt);
      const runtime::ExecutionGraphPlan cv_plan =
          runtime::compile_public_graph(cv_graph, RunOptions{}, cv_seed);
      const PreprocOptions& graph_cv =
          require_single_segment_preproc_options(cv_plan, "model.graph().build(cv::Mat seed)");
      require(expected_cv.input_img_type == "BGR",
              "cv::Mat route seed should preserve OpenCV BGR semantics");
      require(expected_cv.input_shape == std::vector<int>({77, 91, 3}),
              "cv::Mat route seed should preserve image geometry");
      require_same_preproc_seed_contract(expected_cv, graph_cv,
                                         "Model::build(cv::Mat) vs graph cv::Mat seed route");
#endif

      const auto fixture_a = make_seed_preproc_fixture("model_graph_seed_preproc_fanout_a");
      const auto fixture_b = make_seed_preproc_fixture("model_graph_seed_preproc_fanout_b");
      Model model_a(fixture_a.tar_path, image_model_options());
      Model model_b(fixture_b.tar_path, image_model_options());

      Graph fanout_graph;
      auto frame_input = nodes::Input("frame");
      Graph route_a = model_a.graph(route_opt);
      route_a.set_name("route_a");
      Graph route_b = model_b.graph(route_opt);
      route_b.set_name("route_b");
      fanout_graph.connect(frame_input, route_a);
      fanout_graph.connect(frame_input, route_b);

      const runtime::ExecutionGraphPlan fanout_plan =
          runtime::compile_public_graph(fanout_graph, RunOptions{}, bgr_seed);
      const auto fanout_preprocs = collect_model_preproc_options(fanout_plan);
      require(fanout_preprocs.size() == 2U,
              "connected fanout should specialize both model-managed Preproc routes");
      require(contains_preproc_contract(fanout_preprocs, "BGR", {123, 321, 3}),
              "connected fanout route A should receive the raw-image seed contract");
      require(std::all_of(fanout_preprocs.begin(), fanout_preprocs.end(),
                          [](const PreprocOptions& opt) {
                            return opt.input_img_type == "BGR" &&
                                   opt.input_shape == std::vector<int>({123, 321, 3});
                          }),
              "connected fanout should apply the same single raw-image seed to every root route");

      Graph multi_input_graph;
      auto left_input = nodes::Input("left");
      auto right_input = nodes::Input("right");
      Graph left_route = model_a.graph(route_opt);
      left_route.set_name("left_route");
      Graph right_route = model_b.graph(route_opt);
      right_route.set_name("right_route");
      multi_input_graph.connect(left_input, left_route);
      multi_input_graph.connect(right_input, right_route);

      Sample multi_seed;
      multi_seed.push_back(make_image_tensor_sample(111, 222, ImageSpec::PixelFormat::BGR));
      multi_seed.fields.back().port_name = "left";
      multi_seed.push_back(make_image_tensor_sample(333, 444, ImageSpec::PixelFormat::RGB));
      multi_seed.fields.back().port_name = "right";
      const runtime::ExecutionGraphPlan multi_input_plan =
          runtime::compile_public_graph(multi_input_graph, RunOptions{}, multi_seed);
      const auto multi_input_preprocs = collect_model_preproc_options(multi_input_plan);
      require(multi_input_preprocs.size() == 2U,
              "connected multi-input graph should specialize both model-managed Preproc routes");
      require(contains_preproc_contract(multi_input_preprocs, "BGR", {222, 111, 3}),
              "multi-input route should preserve first image seed geometry/color");
      require(contains_preproc_contract(multi_input_preprocs, "RGB", {444, 333, 3}),
              "multi-input route should preserve second image seed geometry/color");
    }));

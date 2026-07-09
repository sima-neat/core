#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "neat.h"
#include "test_utils.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require_true(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

void require_contains(const std::string& text, const std::string& needle,
                      const std::string& message) {
  require_true(text.find(needle) != std::string::npos, message + ": missing '" + needle + "'");
}

simaai::neat::Model::Options make_options(bool set_input_max) {
  simaai::neat::Model::Options options;
  options.preprocess.kind = simaai::neat::InputKind::Image;
  options.preprocess.enable = simaai::neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  if (set_input_max) {
    options.preprocess.input_max_width = 1280;
    options.preprocess.input_max_height = 960;
    options.preprocess.input_max_depth = 3;
  }
  options.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  options.preprocess.resize.width = 640;
  options.preprocess.resize.height = 640;
  options.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  return options;
}

void check_model(const std::filesystem::path& model_path, bool explicit_input_max,
                 int expected_width, int expected_height, int expected_depth) {
  simaai::neat::Model model(model_path.string(), make_options(explicit_input_max));
  const auto plan = model.resolved_preprocess_plan();
  require_true(plan.enabled, "preprocess plan must be enabled");
  require_true(plan.resolved_kind == simaai::neat::InputKind::Image,
               "preprocess plan must resolve to image input");

  if (explicit_input_max) {
    require_true(plan.requested.input_max_width == expected_width &&
                     plan.requested.input_max_height == expected_height &&
                     plan.requested.input_max_depth == expected_depth,
                 "requested input_max must preserve the PCIe image envelope");
    require_true(plan.effective.input_max_width == expected_width &&
                     plan.effective.input_max_height == expected_height &&
                     plan.effective.input_max_depth == expected_depth,
                 "effective input_max must preserve the PCIe image envelope");
  }

  const simaai::neat::PreprocOptions preproc_options(model);
  require_true(preproc_options.input_height() == expected_height &&
                   preproc_options.input_width() == expected_width &&
                   preproc_options.input_channels() == expected_depth,
               "public PreprocOptions(Model) must use the static processcvu capacity envelope "
               "for PCIe-style image preproc; observed input_shape=[" +
                   std::to_string(preproc_options.input_height()) + ", " +
                   std::to_string(preproc_options.input_width()) + ", " +
                   std::to_string(preproc_options.input_channels()) + "]");

  simaai::neat::Graph graph(explicit_input_max ? "pcie-like-yolov8-explicit-input-max"
                                               : "pcie-like-yolov8-default-input-max");
  simaai::neat::PCIeSrcOptions src_options;
  simaai::neat::PCIeSinkOptions sink_options;
  sink_options.queue = 0;
  graph.add(simaai::neat::nodes::PCIeSrc(src_options));
  graph.add(model.graph());
  graph.add(simaai::neat::nodes::PCIeSink(sink_options));

  const std::string backend = graph.describe_backend(false);
  std::cout << "[pcie-preproc-max] case=" << (explicit_input_max ? "explicit" : "default")
            << " model=" << model_path << "\n";
  std::cout << "[pcie-preproc-max] expected envelope=[" << expected_height << ", " << expected_width
            << ", " << expected_depth << "]\n";
  std::cout << "[pcie-preproc-max] public PreprocOptions input_shape=["
            << preproc_options.input_height() << ", " << preproc_options.input_width() << ", "
            << preproc_options.input_channels() << "]\n";
  std::cout << "[pcie-preproc-max] backend:\n" << backend << "\n";

  require_contains(backend, "neatpciesrc", "PCIe-style graph must start with neatpciesrc");
  require_contains(backend, "neatprocesscvu", "PCIe-style graph must include processcvu");
  require_contains(backend, "neatprocessmla", "PCIe-style graph must include processmla");
  require_contains(backend, "neatpciesink", "PCIe-style graph must end with neatpciesink");

  const std::size_t src_pos = backend.find("neatpciesrc");
  const std::size_t cvu_pos = backend.find("neatprocesscvu");
  const std::size_t mla_pos = backend.find("neatprocessmla");
  const std::size_t sink_pos = backend.find("neatpciesink");
  require_true(src_pos < cvu_pos && cvu_pos < mla_pos && mla_pos < sink_pos,
               "PCIe-style graph must order src -> preproc/processcvu -> processmla -> sink");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path root =
        (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
    std::error_code ec;
    std::filesystem::create_directories(root / "tmp", ec);
    std::filesystem::current_path(root, ec);

    const std::filesystem::path model_path = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);

    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatpciesrc") ||
        !simaai::neat::element_exists("neatprocesscvu") ||
        !simaai::neat::element_exists("neatprocessmla") ||
        !simaai::neat::element_exists("neatpciesink")) {
      return skip_long_test("missing PCIe/model GStreamer plugins");
    }

    check_model(model_path, /*explicit_input_max=*/true, 1280, 960, 3);
    check_model(model_path, /*explicit_input_max=*/false, 1920, 1080, 3);

    std::cout << "RESULT pcie_style_public_preproc_input_max_graph_ok\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "UNEXPECTED_ERROR_BEGIN\n" << e.what() << "\nUNEXPECTED_ERROR_END\n";
    return 1;
  }
}

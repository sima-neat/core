#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * Device acceptance for the surgically prepared SSD300 model used by the tools application.
 *
 * The prepared heads encode the canonical 81-way SSD300 confidence layout, while the application
 * reports only background + seven tool classes. This test runs that exact 81 -> 8 class-prefix
 * contract through Model's model-managed on-device BoxDecode route. It intentionally validates
 * wire integrity and class bounds rather than claiming semantic accuracy on the generic COCO
 * image; a surgical-video golden belongs with the application fixture.
 */
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string resolve_ssd300_archive(const fs::path& root) {
  for (const char* name : {"SIMA_SSD300_TAR", "SIMA_SSD_SURGICAL_TAR"}) {
    if (const char* value = std::getenv(name); value && *value) {
      std::error_code error;
      if (fs::is_regular_file(value, error) && !error) {
        return value;
      }
    }
  }

  for (const auto& candidate :
       {root / "tmp" / "ssd300_heads_mpk.tar.gz", root / "ssd300_heads_mpk.tar.gz"}) {
    std::error_code error;
    if (fs::is_regular_file(candidate, error) && !error) {
      return candidate.string();
    }
  }

  skip_long_test_exception(
      "SSD300 surgical archive not staged; set SIMA_SSD300_TAR to ssd300_heads_mpk.tar.gz");
  throw std::runtime_error("unreachable after skip_long_test_exception");
}

simaai::neat::Model::Options surgical_model_options() {
  using namespace simaai::neat;
  Model::Options options;
  options.preprocess.kind = InputKind::Image;
  options.preprocess.enable = AutoFlag::On;
  options.preprocess.resize.enable = AutoFlag::On;
  options.preprocess.resize.mode = ResizeMode::Stretch;
  options.preprocess.resize.width = 300;
  options.preprocess.resize.height = 300;
  options.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  options.preprocess.color_convert.output_format = PreprocessColorFormat::RGB;
  options.preprocess.normalize.enable = AutoFlag::On;
  options.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  options.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
  options.preprocess.normalize.has_explicit_stats = true;
  options.decode_type = BoxDecodeType::Ssd;
  options.num_classes = 8; // background + seven surgical tools; heads physically encode 81.
  options.score_threshold = 0.001f;
  options.nms_iou_threshold = 0.45f;
  options.top_k = 24; // Must match the pack's baked object-decode output capacity.
  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    using namespace simaai::neat;

    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    const std::string archive = resolve_ssd300_archive(root);
    const fs::path image_path = sima_e2e::ensure_coco_sample(root);
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    require(!image.empty(), "SSD300 surgical acceptance could not load its input image");

    Model model(archive, surgical_model_options());
    Graph graph;
    graph.add(nodes::Input());
    graph.add(model); // Same model-managed postprocess path as the teammate application.
    graph.add(nodes::Output());

    auto make_sample = [&]() {
      return Sample{Sample::from_image(image, ImageSpec::PixelFormat::BGR, TensorMemory::EV74)};
    };
    Run run = graph.build_seeded_internal(make_sample(), RunMode::Sync);

    std::size_t decoded_boxes = 0U;
    for (int frame = 0; frame < 3; ++frame) {
      const Sample outputs = run.run(make_sample(), 30000);
      require(!outputs.empty(), "SSD300 surgical acceptance returned no output");

      std::vector<std::uint8_t> payload;
      std::string error;
      require(objdet::extract_bbox_payload(outputs.front(), frame, payload, error),
              "SSD300 surgical BBOX extraction failed: " + error);
      const auto boxes = objdet::parse_boxes_strict(payload, image.cols, image.rows,
                                                    /*expected_topk=*/24, false);
      for (const auto& box : boxes) {
        require(box.class_id >= 1 && box.class_id < 8,
                "SSD300 surgical decoder returned a class outside the selected [1,7] prefix");
      }
      decoded_boxes += boxes.size();
    }
    run.stop();

    require(decoded_boxes > 0U,
            "SSD300 surgical acceptance produced no boxes; the 81->8 score path was not exercised");
    require(graph.last_pipeline().find("neatobjectdecode") != std::string::npos,
            "SSD300 surgical model did not materialize the on-device object decoder");
    std::cout << "[OK] SSD300 surgical model-managed decode boxes=" << decoded_boxes << "\n";
    return 0;
  } catch (const SkipTest& error) {
    std::cout << "[SKIP] " << error.what() << "\n";
    return skip_long_test(error.what());
  } catch (const std::exception& error) {
    if (is_dispatcher_unavailable(error.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}

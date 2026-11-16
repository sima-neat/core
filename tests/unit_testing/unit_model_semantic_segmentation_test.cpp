// Model::Options::semantic_segmentation admission. The class-map decode replaces a dequantize
// or cast postprocess; every other route must fail at construction rather than silently return
// scores. Self-contained: synthetic route plans plus the checked-in strict seed archive.
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/RoutePlanner.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace simaai::neat;
using Kind = pipeline_internal::sima::RouteGraphKernelKind;
using Region = internal::RouteRegionKind;

internal::SessionRoutePlan route(const std::vector<std::pair<Region, Kind>>& regions) {
  internal::SessionRoutePlan plan;
  for (const auto& [region_kind, op_kind] : regions) {
    internal::RouteRegion region;
    region.kind = region_kind;
    region.op_kind = op_kind;
    plan.post_regions.push_back(region);
  }
  plan.include_post_stage = !plan.post_regions.empty();
  return plan;
}

Model::Options enabled() {
  Model::Options options;
  options.semantic_segmentation.enable = true;
  return options;
}

void expect_rejected(const std::function<void()>& fn, const std::string& needle,
                     const std::string& what) {
  try {
    fn();
  } catch (const std::invalid_argument& error) {
    require_contains(error.what(), "Model::Options::semantic_segmentation: ", what + " prefix");
    require_contains(error.what(), needle, what);
    return;
  }
  throw std::runtime_error(what + ": expected std::invalid_argument");
}

void route_rules() {
  const auto dequantize = route({{Region::Linear, Kind::Dequantize}});
  internal::validate_semantic_segmentation(enabled(), dequantize, false);
  internal::validate_semantic_segmentation(enabled(), route({{Region::Linear, Kind::Cast}}), false);
  internal::validate_semantic_segmentation(
      enabled(), route({{Region::Linear, Kind::Dequantize}, {Region::Linear, Kind::Cast}}), false);
  auto sized = enabled();
  sized.semantic_segmentation.output_width = 1280;
  sized.semantic_segmentation.output_height = 720;
  internal::validate_semantic_segmentation(sized, dequantize, false);

  expect_rejected(
      [] {
        internal::validate_semantic_segmentation(
            enabled(), route({{Region::FanoutMap, Kind::DetessDequant}}), false);
      },
      "must only dequantize or cast its class scores, but it runs 'detessdequant'",
      "a detection postprocess is not a dtype-only tail");
  expect_rejected(
      [] {
        internal::validate_semantic_segmentation(enabled(), route({{Region::Linear, Kind::Detess}}),
                                                 false);
      },
      "it runs 'detess'", "a tiled MLA output needs a detessellating postprocess");
  expect_rejected(
      [] {
        internal::validate_semantic_segmentation(
            enabled(), route({{Region::FanoutMap, Kind::Dequantize}}), false);
      },
      "it runs 'dequantize'", "a fan-out dequantize feeds several outputs, not one class map");
  expect_rejected([] { internal::validate_semantic_segmentation(enabled(), route({}), false); },
                  "has no dequantize or cast postprocess", "a model without a postprocess");
  expect_rejected([&] { internal::validate_semantic_segmentation(enabled(), dequantize, true); },
                  "inference_terminal stops the model before its postprocess",
                  "an MLA-only terminal policy");
  expect_rejected(
      [&] {
        auto options = enabled();
        options.decode_type = BoxDecodeType::YoloV8;
        internal::validate_semantic_segmentation(options, dequantize, false);
      },
      "cannot be combined with a BoxDecode decode_type", "BoxDecode and the class-map decode");
  for (const auto& [width, height] :
       std::vector<std::pair<int, int>>{{640, 0}, {0, 480}, {-1, -1}}) {
    expect_rejected(
        [&] {
          auto options = enabled();
          options.semantic_segmentation.output_width = width;
          options.semantic_segmentation.output_height = height;
          internal::validate_semantic_segmentation(options, dequantize, false);
        },
        "(got " + std::to_string(width) + "x" + std::to_string(height) + ")",
        "an incomplete output size");
  }
}

// The constructor applies the same rules: a detection model's archive is rejected, an MLA-only
// terminal policy is rejected before its route is considered, and the option off changes nothing.
void model_construction() {
  const auto fixture = sima_test::make_strict_model_archive_fixture("semantic_segmentation", {},
                                                                    true, "yolo_v9c_seg", true);
  Model::Options plain;
  plain.preprocess.kind = InputKind::Tensor;
  Model model(fixture.tar_path, plain);
  require(model.info().selection.selected_post_kind != "resize_argmax",
          "semantic_segmentation off must keep the model's own postprocess");

  auto options = plain;
  options.semantic_segmentation.enable = true;
  expect_rejected([&] { Model rejected(fixture.tar_path, options); }, "",
                  "a YOLOv9-seg archive has no dense class-score postprocess");
  options.inference_terminal.mla_only = true;
  expect_rejected([&] { Model rejected(fixture.tar_path, options); },
                  "inference_terminal stops the model before its postprocess",
                  "semantic_segmentation with an MLA-only terminal policy");
}
} // namespace

RUN_TEST("unit_model_semantic_segmentation_test", ([] {
           route_rules();
           model_construction();
         }));

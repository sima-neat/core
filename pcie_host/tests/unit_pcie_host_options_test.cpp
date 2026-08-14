#include "ModelOptionsJsonWriter.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace pcie = simaai::neat::pcie;
namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

} // namespace

int main() {
  try {
    {
      pcie::ModelOptions opt;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(!json.json.has_value(), "default tensor route must not emit JSON");
      require(!json.has_boxdecode, "default tensor route must not expect BBOX output");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "image route must emit JSON");
      require(contains(*json.json, "\"schema\": 1"), "schema missing");
      require(contains(*json.json, "\"preprocess\": {}"), "empty preprocess object missing");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.resize.enable = pcie::AutoFlag::On;
      opt.preprocess.resize.mode = pcie::ResizeMode::Stretch;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "image resize route must emit JSON");
      require(contains(*json.json, "\"enable\": true"), "resize enable state missing");
      require(contains(*json.json, "\"mode\": \"stretch\""), "resize mode missing");
      require(!contains(*json.json, "\"width\""), "resize width must be core-inferred");
      require(!contains(*json.json, "\"height\""), "resize height must be core-inferred");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.resize.enable = pcie::AutoFlag::Off;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "disabled image resize route must emit JSON");
      require(contains(*json.json, "\"enable\": false"), "resize disabled state missing");
    }

    for (const auto [width, height] : {std::pair{640, 0}, std::pair{0, 640}}) {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.resize.width = width;
      opt.preprocess.resize.height = height;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "explicit resize dimensions must throw");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.color_convert.enable = pcie::AutoFlag::Off;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "disabled color conversion route must emit image JSON");
      require(contains(*json.json, "\"color_convert\""), "disabled color conversion state missing");
      require(contains(*json.json, "\"enable\": false"), "disabled color conversion flag missing");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.color_convert.enable = pcie::AutoFlag::Off;
      opt.preprocess.color_convert.input_format = pcie::ColorFormat::BGR;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "disabled color conversion with explicit formats must throw");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.normalize.enable = pcie::AutoFlag::Off;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "disabled normalization route must emit image JSON");
      require(contains(*json.json, "\"normalize\""), "disabled normalization state missing");
      require(contains(*json.json, "\"enable\": false"), "disabled normalization flag missing");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.normalize.enable = pcie::AutoFlag::Off;
      opt.preprocess.normalize.preset = pcie::NormalizePreset::ImageNet;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "disabled normalization with a preset must throw");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.normalize.enable = pcie::AutoFlag::Off;
      opt.preprocess.normalize.has_explicit_stats = true;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "disabled normalization with explicit stats must throw");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.color_convert.input_format = pcie::ColorFormat::NV12;
      opt.preprocess.color_convert.output_format = pcie::ColorFormat::RGB;
      opt.decode_type = pcie::BoxDecodeType::YoloV8;
      opt.score_threshold = 0.25f;
      const auto json = pcie_internal::write_model_options_json(opt);
      require(json.json.has_value(), "boxdecode route must emit JSON");
      require(json.has_boxdecode, "boxdecode route must expect BBOX output");
      require(contains(*json.json, "\"input_format\": \"nv12\""), "NV12 input missing");
      require(contains(*json.json, "\"output_format\": \"rgb\""), "RGB output missing");
      require(contains(*json.json, "\"decode_type\": \"yolov8\""), "decode type missing");
      require(contains(*json.json, "\"score_threshold\""), "score threshold missing");
    }

    {
      pcie::ModelOptions opt;
      opt.decode_type = pcie::BoxDecodeType::YoloV8;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "boxdecode without image preprocess must throw");
    }

    {
      pcie::ModelOptions opt;
      opt.preprocess.kind = pcie::InputKind::Image;
      opt.preprocess.color_convert.output_format = pcie::ColorFormat::NV12;
      bool threw = false;
      try {
        (void)pcie_internal::write_model_options_json(opt);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "NV12 output must throw");
    }

    std::cout << "[PASS] model options JSON writer\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

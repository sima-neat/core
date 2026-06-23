#include "ModelOptionsJsonWriter.h"

#include <iostream>
#include <stdexcept>
#include <string>

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
      require(contains(*json.json, "\"mode\": \"stretch\""), "resize mode missing");
      require(!contains(*json.json, "\"width\""), "resize width must be core-inferred");
      require(!contains(*json.json, "\"height\""), "resize height must be core-inferred");
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

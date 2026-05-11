#include "gst/GstHelpers.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/Tess.h"
#include "test_main.h"

#include <nlohmann/json.hpp>

#include <string>

namespace {

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (found unexpected: " + needle + ")");
  }
}

nlohmann::json make_single_io_cfg(const char* graph_name) {
  return nlohmann::json{
      {"graph_name", graph_name},
      {"input_width", 640},
      {"input_height", 640},
      {"input_depth", 3},
      {"output_width", 640},
      {"output_height", 640},
      {"output_depth", 3},
  };
}

} // namespace

RUN_TEST("unit_processcvu_family_fragment_test_quant", ([] {
  using namespace simaai::neat;
  require(element_exists("neatprocesscvu"), "required plugin missing (neatprocesscvu)");

  QuantOptions opt;
  opt.config_json = make_single_io_cfg("quantize");
  const auto node = nodes::Quant(opt);
  const std::string frag = node->backend_fragment(2);
  require_contains(frag, "neatprocesscvu name=n2_quant", "Quant fragment name mismatch");
  require_contains(frag, "stage-id=n2_quant", "Quant fragment stage-id missing");
  require_not_contains(frag, "config=", "Quant fragment must not emit legacy config path");
}));

RUN_TEST("unit_processcvu_family_fragment_test_tess", ([] {
  using namespace simaai::neat;
  TessOptions opt;
  opt.config_json = make_single_io_cfg("tessellate");
  const auto node = nodes::Tess(opt);
  const std::string frag = node->backend_fragment(3);
  require_contains(frag, "neatprocesscvu name=n3_tess", "Tess fragment name mismatch");
  require_contains(frag, "stage-id=n3_tess", "Tess fragment stage-id missing");
  require_not_contains(frag, "config=", "Tess fragment must not emit legacy config path");
}));

RUN_TEST("unit_processcvu_family_fragment_test_quanttess", ([] {
  using namespace simaai::neat;
  QuantTessOptions opt;
  opt.config_json = make_single_io_cfg("quanttess");
  const auto node = nodes::QuantTess(opt);
  const std::string frag = node->backend_fragment(4);
  require_contains(frag, "neatprocesscvu name=n4_quanttess", "QuantTess fragment name mismatch");
  require_contains(frag, "stage-id=n4_quanttess", "QuantTess fragment stage-id missing");
  require_not_contains(frag, "config=", "QuantTess fragment must not emit legacy config path");
}));

RUN_TEST("unit_processcvu_family_fragment_test_detessdequant", ([] {
  using namespace simaai::neat;
  DetessDequantOptions opt;
  opt.num_buffers_locked = true;
  opt.num_buffers_model = 1;
  opt.num_buffers = 1;
  opt.config_json = make_single_io_cfg("detessdequant");
  const auto node = nodes::DetessDequant(opt);
  const std::string frag = node->backend_fragment(5);
  require_contains(frag, "neatprocesscvu name=n5_detessdequant",
                   "DetessDequant fragment name mismatch");
  require_contains(frag, "stage-id=n5_detessdequant",
                   "DetessDequant fragment stage-id missing");
  require_not_contains(frag, "config=",
                       "DetessDequant fragment must not emit legacy config path");
}));

#include "gst/GstHelpers.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/DetessCast.h"
#include "nodes/sima/Detess.h"
#include "nodes/sima/Dequant.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/Tess.h"
#include "test_main.h"

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include <string>

namespace {

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (found unexpected: " + needle + ")");
  }
}

// Parse the actual registered factory, not a stand-in, without starting hardware.
void require_runner_fragment(const std::string& fragment) {
  for (const char* forbidden : {"stage-id=", "num-buffers=", "config=", "q-scale=", "q-zp="}) {
    require_not_contains(fragment, forbidden,
                         "runner must receive typed context, not legacy properties");
  }
  GError* error = nullptr;
  const std::string launch = "fakesrc ! " + fragment + " ! fakesink";
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  const std::string message = error ? error->message : "";
  if (error) {
    g_error_free(error);
  }
  if (!pipeline || !message.empty()) {
    if (pipeline) {
      gst_object_unref(pipeline);
    }
    throw std::runtime_error("runner fragment must parse with request sink pad: " + message);
  }
  gst_object_unref(pipeline);
}

nlohmann::json make_single_io_cfg(const char* graph_name) {
  return nlohmann::json{
      {"graph_name", graph_name}, {"input_width", 640},   {"input_height", 640}, {"input_depth", 3},
      {"output_width", 640},      {"output_height", 640}, {"output_depth", 3},
  };
}

} // namespace

void unit_processcvu_family_fragment_test_quant() {
  using namespace simaai::neat;
  require(element_exists("neatkernelrunner"), "required plugin missing (neatkernelrunner)");

  QuantOptions opt;
  opt.config_json = make_single_io_cfg("quantize");
  const auto node = nodes::Quant(opt);
  const std::string frag = node->backend_fragment(2);
  require_contains(frag, "neatkernelrunner name=n2_quant", "Quant fragment name mismatch");
  require_runner_fragment(frag);
  require_not_contains(frag, "config=", "Quant fragment must not emit legacy config path");
}

void unit_processcvu_family_fragment_test_tess() {
  using namespace simaai::neat;
  TessOptions opt;
  opt.config_json = make_single_io_cfg("tessellate");
  const auto node = nodes::Tess(opt);
  const std::string frag = node->backend_fragment(3);
  require_contains(frag, "neatkernelrunner name=n3_tess", "Tess fragment name mismatch");
  require_runner_fragment(frag);
  require_not_contains(frag, "config=", "Tess fragment must not emit legacy config path");
}

void unit_processcvu_family_fragment_test_quanttess() {
  using namespace simaai::neat;
  QuantTessOptions opt;
  opt.config_json = make_single_io_cfg("quanttess");
  const auto node = nodes::QuantTess(opt);
  const std::string frag = node->backend_fragment(4);
  require_contains(frag, "neatkernelrunner name=n4_quanttess", "QuantTess fragment name mismatch");
  require_runner_fragment(frag);
  require_not_contains(frag, "config=", "QuantTess fragment must not emit legacy config path");
}

void unit_processcvu_family_fragment_test_detessdequant() {
  using namespace simaai::neat;
  DetessDequantOptions opt;
  opt.num_buffers_locked = true;
  opt.num_buffers_model = 1;
  opt.num_buffers = 1;
  opt.config_json = make_single_io_cfg("detessdequant");
  const auto node = nodes::DetessDequant(opt);
  const std::string frag = node->backend_fragment(5);
  require_contains(frag, "neatkernelrunner name=n5_detessdequant",
                   "DetessDequant fragment name mismatch");
  require_runner_fragment(frag);
  require_not_contains(frag, "config=", "DetessDequant fragment must not emit legacy config path");
}

void unit_processcvu_family_fragment_test_remaining_transforms() {
  using namespace simaai::neat;
  CastTessOptions casttess;
  DetessCastOptions detesscast;
  detesscast.num_buffers_locked = true;
  detesscast.num_buffers_model = 1;
  detesscast.num_buffers = 1;
  DetessOptions detess;
  // Rendering dispatch follows the typed payload, not optional execution flags.
  detess.compiled_contract = std::make_shared<const CompiledProcessCvuContract>();
  DequantOptions dequant;
  dequant.processcvu_compiled_contract = std::make_shared<const CompiledProcessCvuContract>();
  for (const auto& node : {nodes::CastTess(casttess), nodes::DetessCast(detesscast),
                           nodes::Detess(detess), nodes::Dequant(dequant)}) {
    const std::string fragment = node->backend_fragment(6);
    require_contains(fragment, "neatkernelrunner name=", "all transforms use Runtime");
    require_runner_fragment(fragment);
  }
}

RUN_TEST("unit_processcvu_family_fragment_test", ([] {
           unit_processcvu_family_fragment_test_quant();
           unit_processcvu_family_fragment_test_tess();
           unit_processcvu_family_fragment_test_quanttess();
           unit_processcvu_family_fragment_test_detessdequant();
           unit_processcvu_family_fragment_test_remaining_transforms();
         }));

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using simaai::neat::SimaDecodeOptions;
using simaai::neat::SimaDecodeType;

SimaDecodeOptions decoder_options(SimaDecodeType type = SimaDecodeType::H264) {
  SimaDecodeOptions options;
  options.type = type;
  options.dec_width = 1280;
  options.dec_height = 720;
  options.dec_fps = 60;
  options.raw_output = true;
  return options;
}

void require_direct_output_property(const simaai::neat::Node& decoder) {
  const std::string fragment = decoder.backend_fragment(0);
  const std::string property = "zero-copy-output=true";
  const auto pos = fragment.find(property);
  require(pos != std::string::npos &&
              fragment.find(property, pos + property.size()) == std::string::npos,
          "decoder must emit its direct-output property exactly once");

  // Parse only the decoder element: property readback must not depend on an
  // adapter tail's caps negotiation or start a hardware session.
  const std::string element_fragment = fragment.substr(0, fragment.find(" ! "));
  GError* error = nullptr;
  GstElement* element = gst_parse_launch(element_fragment.c_str(), &error);
  const std::string detail = error ? error->message : "";
  if (error) {
    g_error_free(error);
  }
  if (!element || !detail.empty()) {
    if (element) {
      gst_object_unref(element);
    }
    throw std::runtime_error("decoder property readback parse failed: " + detail);
  }
  gboolean direct = FALSE;
  g_object_get(element, "zero-copy-output", &direct, nullptr);
  gst_object_unref(element);
  require(direct == TRUE, "generated decoder element must use direct DMA-BUF output");
  require(fragment.find("admission-") == std::string::npos,
          "direct decoder must not render daemon admission properties");
}

void check_zero_copy_policy_is_producer_owned() {
  for (const auto codec :
       {SimaDecodeType::H264, SimaDecodeType::H265, SimaDecodeType::JPEG, SimaDecodeType::MJPEG}) {
    for (const std::string next : {"", "CVU", "APU"}) {
      auto options = decoder_options(codec);
      options.next_element = next;
      const simaai::neat::SimaDecode node(options);
      require_direct_output_property(node);
      // OutputSpec describes the exposed boundary. The generic MemoryContract
      // also controls graph metadata probes, not the decoder's allocation policy.
      const auto output = node.output_spec({});
      require(output.memory == "SimaAI" && output.format == "NV12",
              "native decoder output must preserve its device-memory NV12 boundary");
    }
  }

  auto options = decoder_options();
  options.raw_output = false;
  options.out_format = simaai::neat::FormatTag::RGB;
  const simaai::neat::SimaDecode adapted(options);
  require_direct_output_property(adapted);
  const auto adapted_output = adapted.output_spec({});
  require(adapted_output.memory == "SystemMemory" && adapted_output.format == "RGB",
          "an explicit adapter tail must report its exposed storage, not the decoder pool");
  require_contains(adapted.backend_fragment(0), "videoconvert",
                   "an explicit compatibility request must retain its adapter");

  options.raw_output = true;
  try {
    (void)simaai::neat::SimaDecode(options).backend_fragment(0);
  } catch (const std::invalid_argument& error) {
    require_contains(error.what(), "raw_output supports only NV12 or I420",
                     "direct output must preserve format validation");
    return;
  }
  throw std::runtime_error("direct output accepted an unsupported RGB format");
}

void check_output_pool_and_tuning_options() {
  auto options = decoder_options();
  options.num_buffers = -1;
  const auto unspecified = simaai::neat::SimaDecode(options).backend_fragment(0);
  require(unspecified.find("num-buffers=") == std::string::npos,
          "an unspecified output pool must preserve the decoder element default");

  // The codec owns its runtime minimum. Core forwards explicit settings rather
  // than imposing a second pool floor from an obsolete daemon lease.
  for (const int buffers : {4, 9}) {
    options.num_buffers = buffers;
    require_contains(simaai::neat::SimaDecode(options).backend_fragment(0),
                     "num-buffers=" + std::to_string(buffers),
                     "Core must preserve an explicit decoder output pool");
  }
  options.input_buffers = 3;
  options.decoder_tuning = "low-memory";
  options.memory_opt = true;
  const auto explicit_options = simaai::neat::SimaDecode(options).backend_fragment(0);
  require_contains(explicit_options, "dec-ip-cnt=3", "explicit input pool must be preserved");
  require_contains(explicit_options, "decoder-tuning=low-memory",
                   "explicit decoder tuning must be preserved");
  require_contains(explicit_options, "memory-opt=true", "explicit memory policy must be preserved");
}

void check_sync_cache_rebuild_order() {
  const std::vector<std::string> success =
      simaai::neat::session_test::sync_cache_rebuild_events_for_test(false);
  require(success == std::vector<std::string>({"old-release", "build", "new-release"}),
          "cache replacement must release old runner before building its replacement");

  const std::vector<std::string> failure =
      simaai::neat::session_test::sync_cache_rebuild_events_for_test(true);
  require(failure == std::vector<std::string>({"old-release", "build", "throw"}),
          "failed cache replacement must leave the old runner released and cache empty");
}

} // namespace

int main() {
  try {
    unsetenv("SIMA_ALLOW_GST_INIT");
    simaai::neat::gst_init_once();
    check_zero_copy_policy_is_producer_owned();
    check_output_pool_and_tuning_options();
    check_sync_cache_rebuild_order();
    std::cout << "[OK] unit_decoder_admission_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

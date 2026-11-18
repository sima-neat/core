#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/sima/SimaDecode.h"

#include "test_main.h"

#include <functional>
#include <string>
#include <vector>

namespace {

constexpr const char* kOptionOutOfRange = "misconfig.option_out_of_range";

// Refused values must raise a NeatError that names the field, the rejected value and the
// canonical presets, in both what() and repro_note.
void require_tuning_refused(const std::function<void()>& make, const std::string& value) {
  const std::string quoted = "'" + value + "'";
  for (const std::string needle :
       {std::string("decoder_tuning"), quoted, std::string("auto"), std::string("default"),
        std::string("low-memory"), std::string("throughput-low-latency")}) {
    require_neat_error(make, kOptionOutOfRange, needle, needle);
  }
}

simaai::neat::SimaDecodeOptions decode_options(const std::string& tuning) {
  simaai::neat::SimaDecodeOptions opt;
  opt.decoder_tuning = tuning;
  return opt;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_options(const std::string& tuning) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = "rtsp://example.local/h265";
  opt.codec = simaai::neat::nodes::groups::RtspCodec::H265;
  opt.dec_width = 1280;
  opt.dec_height = 720;
  opt.source_fps = 30;
  opt.decoder_tuning = tuning;
  return opt;
}

const std::vector<std::string>& refused_values() {
  static const std::vector<std::string> values = {"bogus_value",
                                                  "lowmemory",
                                                  "none",
                                                  " low-memory",
                                                  "low-memory ",
                                                  "low-memory\n",
                                                  "auto ! fakesink",
                                                  "low-memory ! fakesink"};
  return values;
}

const std::vector<std::string>& accepted_values() {
  static const std::vector<std::string> values = {"auto",
                                                  "Auto",
                                                  "default",
                                                  "low-memory",
                                                  "LOW-MEMORY",
                                                  "low_memory",
                                                  "memory",
                                                  "throughput-low-latency",
                                                  "throughput_low_latency",
                                                  "throughput",
                                                  "low-latency-throughput"};
  return values;
}

void check_class_constructor_refuses_unknown() {
  for (const auto& value : refused_values()) {
    require_tuning_refused([&] { simaai::neat::SimaDecode node(decode_options(value)); }, value);
  }
}

void check_factory_refuses_unknown() {
  for (const auto& value : refused_values()) {
    require_tuning_refused([&] { (void)simaai::neat::nodes::SimaDecode(decode_options(value)); },
                           value);
  }
}

void check_rtsp_decoded_input_refuses_unknown() {
  for (const auto& value : refused_values()) {
    require_tuning_refused(
        [&] { (void)simaai::neat::nodes::groups::RtspDecodedInput(rtsp_options(value)); }, value);
  }
}

void check_accepted_and_empty_still_construct() {
  for (const auto& value : accepted_values()) {
    simaai::neat::SimaDecode node(decode_options(value));
    require(node.options().decoder_tuning == value, "accepted tuning must be kept unchanged");
    (void)simaai::neat::nodes::SimaDecode(decode_options(value));
    (void)simaai::neat::nodes::groups::RtspDecodedInput(rtsp_options(value));
  }
  simaai::neat::SimaDecode empty_node(decode_options(""));
  require(empty_node.options().decoder_tuning.empty(), "empty tuning must stay empty");
  (void)simaai::neat::nodes::groups::RtspDecodedInput(rtsp_options(""));
}

} // namespace

RUN_TEST("unit_sima_decode_tuning_validation_regression_test", [] {
  check_class_constructor_refuses_unknown();
  check_factory_refuses_unknown();
  check_rtsp_decoded_input_refuses_unknown();
  check_accepted_and_empty_still_construct();
});

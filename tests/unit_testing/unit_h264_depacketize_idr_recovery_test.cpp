// Runtime regression for H264Depacketize's default config-interval (-1).
//
// The source carries SPS/PPS only ahead of its first access unit, as an MP4 republished over RTSP
// without re-encoding does. Downstream of the depacketizer the first access units are dropped, as
// the leaky egress queue of a fused encoded branch can do at startup. A consumer then sees
// parameter sets again only if the depacketizer's h264parse repeats them before later IDRs.

#include "gst/GstHelpers.h"
#include "gst_test_utils.h"
#include "nodes/rtp/H264Depacketize.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kFrames = 30;
constexpr int kGopFrames = 10;
constexpr int kIdrCount = kFrames / kGopFrames;
constexpr int kDroppedAccessUnits = 3;
constexpr int kPayloadType = 97;
constexpr int kNalIdr = 5;
constexpr int kNalSps = 7;
constexpr int kNalPps = 8;

// A software H.264 encoder this host provides, set to an IDR every kGopFrames frames without
// B-frames, or "" if it has none. avenc_h264 is left out: no test image here provides it, so its
// settings could not be checked.
std::string software_h264_encoder() {
  if (simaai::neat::element_exists("x264enc")) {
    return "x264enc tune=zerolatency speed-preset=ultrafast bframes=0 byte-stream=true "
           "key-int-max=" +
           std::to_string(kGopFrames);
  }
  if (simaai::neat::element_exists("openh264enc")) {
    return "openh264enc gop-size=" + std::to_string(kGopFrames);
  }
  return {};
}

// A software H.264 decoder this host provides, or "" if it has none. avdec_h264 is kept from
// outputting frames it decoded without their references, so only cleanly decoded frames count.
std::string software_h264_decoder() {
  const std::string decoder = sima_test::first_available_element({"avdec_h264", "openh264dec"});
  return decoder == "avdec_h264" ? decoder + " output-corrupt=false" : decoder;
}

struct NalUnit {
  std::size_t begin = 0; // start of the start code
  std::size_t end = 0;   // one past the last byte of the NAL unit
  int type = 0;
};

// Splits an Annex B access unit at its start codes (00 00 01 or 00 00 00 01).
std::vector<NalUnit> annexb_nal_units(const guint8* data, std::size_t size) {
  std::vector<NalUnit> units;
  std::size_t i = 0;
  while (i + 3 <= size) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
      const std::size_t begin = (i > 0 && data[i - 1] == 0) ? i - 1 : i;
      if (!units.empty()) {
        units.back().end = begin;
      }
      if (i + 3 < size) {
        units.push_back({begin, size, data[i + 3] & 0x1f});
      }
      i += 3;
    } else {
      ++i;
    }
  }
  return units;
}

bool has_nal_type(const std::vector<NalUnit>& units, int type) {
  for (const NalUnit& unit : units) {
    if (unit.type == type) {
      return true;
    }
  }
  return false;
}

struct Counters {
  std::atomic<int> source_access_units{0};
  std::atomic<int> source_access_units_with_sps{0};
  std::atomic<int> parsed_access_units{0};
  std::atomic<int> parsed_idrs{0};
  std::atomic<int> parsed_idrs_with_sps{0};
  std::atomic<int> decoded_frames{0};
  int drop_access_units = 0;
};

// Encoder output: keeps SPS/PPS only in the first access unit.
GstPadProbeReturn strip_repeated_parameter_sets(GstPad* /*pad*/, GstPadProbeInfo* info,
                                                gpointer user_data) {
  auto* counters = static_cast<Counters*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const int index = counters->source_access_units.fetch_add(1);
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    return GST_PAD_PROBE_OK;
  }
  const std::vector<NalUnit> units = annexb_nal_units(map.data, map.size);
  if (index == 0) {
    counters->source_access_units_with_sps.fetch_add(has_nal_type(units, kNalSps) ? 1 : 0);
    gst_buffer_unmap(buffer, &map);
    return GST_PAD_PROBE_OK;
  }
  std::vector<guint8> kept;
  kept.reserve(map.size);
  for (const NalUnit& unit : units) {
    if (unit.type != kNalSps && unit.type != kNalPps) {
      kept.insert(kept.end(), map.data + unit.begin, map.data + unit.end);
    }
  }
  gst_buffer_unmap(buffer, &map);
  if (kept.size() == gst_buffer_get_size(buffer)) {
    return GST_PAD_PROBE_OK;
  }
  GstBuffer* stripped = gst_buffer_new_allocate(nullptr, kept.size(), nullptr);
  gst_buffer_fill(stripped, 0, kept.data(), kept.size());
  gst_buffer_copy_into(
      stripped, buffer,
      static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS), 0,
      static_cast<gsize>(-1));
  gst_buffer_unref(buffer);
  GST_PAD_PROBE_INFO_DATA(info) = stripped;
  return GST_PAD_PROBE_OK;
}

// Depacketizer output: counts IDRs with and without SPS, then drops the first access units.
GstPadProbeReturn count_and_drop_initial(GstPad* /*pad*/, GstPadProbeInfo* info,
                                         gpointer user_data) {
  auto* counters = static_cast<Counters*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const int index = counters->parsed_access_units.fetch_add(1);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const std::vector<NalUnit> units = annexb_nal_units(map.data, map.size);
    if (has_nal_type(units, kNalIdr)) {
      counters->parsed_idrs.fetch_add(1);
      counters->parsed_idrs_with_sps.fetch_add(has_nal_type(units, kNalSps) ? 1 : 0);
    }
    gst_buffer_unmap(buffer, &map);
  }
  return index < counters->drop_access_units ? GST_PAD_PROBE_DROP : GST_PAD_PROBE_OK;
}

GstPadProbeReturn count_decoded(GstPad* /*pad*/, GstPadProbeInfo* /*info*/, gpointer user_data) {
  static_cast<Counters*>(user_data)->decoded_frames.fetch_add(1);
  return GST_PAD_PROBE_OK;
}

void add_buffer_probe(GstElement* pipeline, const char* element, const char* pad_name,
                      GstPadProbeCallback callback, Counters* counters) {
  GstElement* el = gst_bin_get_by_name(GST_BIN(pipeline), element);
  require(el != nullptr, std::string("missing element ") + element);
  GstPad* pad = gst_element_get_static_pad(el, pad_name);
  gst_object_unref(el);
  require(pad != nullptr, std::string("missing pad ") + element + "." + pad_name);
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, callback, counters, nullptr);
  gst_object_unref(pad);
}

// The element factory at the start of a fragment such as "x264enc key-int-max=10".
std::string factory_name(const std::string& fragment) {
  return fragment.substr(0, fragment.find(' '));
}

struct ArmResult {
  int source_access_units = 0;
  int source_access_units_with_sps = 0;
  int parsed_idrs = 0;
  int parsed_idrs_with_sps = 0;
  int decoded_frames = 0;
  std::string end;    // "eos", or the bus error the pipeline stopped with
  std::string codecs; // encoder and decoder factories, for failure messages
};

// Encodes kFrames, sends them through RTP and the depacketizer, and decodes what is left after
// dropping the first `drop_access_units`. `config_interval` unset uses the node default.
ArmResult run_arm(const std::string& encoder, const std::string& decoder,
                  std::optional<int> config_interval, int drop_access_units) {
  const auto depacketize =
      config_interval ? simaai::neat::nodes::H264Depacketize(kPayloadType, *config_interval)
                      : simaai::neat::nodes::H264Depacketize(kPayloadType);
  const std::string launch =
      "videotestsrc num-buffers=" + std::to_string(kFrames) +
      " ! video/x-raw,format=I420,width=160,height=120,framerate=30/1"
      " ! " +
      encoder +
      " ! capsfilter name=source_caps caps=\"video/x-h264,stream-format=byte-stream,alignment=au\""
      " ! rtph264pay pt=" +
      std::to_string(kPayloadType) + " ! " + depacketize->backend_fragment(1) +
      " ! h264parse name=consumer_parse ! " + decoder + " ! fakesink name=decoded_sink sync=false";

  GError* error = nullptr;
  std::unique_ptr<GstElement, void (*)(gpointer)> owned(gst_parse_launch(launch.c_str(), &error),
                                                        gst_object_unref);
  if (error != nullptr) {
    const std::string message = error->message ? error->message : "unknown parse error";
    g_error_free(error);
    throw std::runtime_error("gst_parse_launch failed: " + message + "\n" + launch);
  }
  GstElement* pipeline = owned.get();

  Counters counters;
  counters.drop_access_units = drop_access_units;
  add_buffer_probe(pipeline, "source_caps", "src", strip_repeated_parameter_sets, &counters);
  add_buffer_probe(pipeline, "n1_h264_caps", "src", count_and_drop_initial, &counters);
  add_buffer_probe(pipeline, "decoded_sink", "sink", count_decoded, &counters);

  std::string bus_error;
  const bool started =
      gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE;
  const bool eos = started && sima_test::wait_for_bus_eos_or_error(pipeline, 10000, bus_error);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  require(started, "pipeline failed to start\n" + launch);
  require(eos || !bus_error.empty(), "pipeline timed out before EOS\n" + launch);

  return {counters.source_access_units.load(),
          counters.source_access_units_with_sps.load(),
          counters.parsed_idrs.load(),
          counters.parsed_idrs_with_sps.load(),
          counters.decoded_frames.load(),
          eos ? "eos" : bus_error,
          factory_name(encoder) + "/" + factory_name(decoder)};
}

std::string describe(const char* arm, const ArmResult& r) {
  return std::string(arm) + " (" + r.codecs +
         "): source_aus=" + std::to_string(r.source_access_units) +
         " source_aus_with_sps=" + std::to_string(r.source_access_units_with_sps) +
         " parsed_idrs=" + std::to_string(r.parsed_idrs) +
         " parsed_idrs_with_sps=" + std::to_string(r.parsed_idrs_with_sps) +
         " decoded=" + std::to_string(r.decoded_frames) + " end=" + r.end;
}

// The fixture itself: one access unit per frame, SPS/PPS only in the first, kIdrCount IDRs.
void require_source_shape(const char* arm, const ArmResult& r) {
  require(r.source_access_units == kFrames,
          describe(arm, r) + ": encoder should emit one access unit per frame");
  require(r.source_access_units_with_sps == 1,
          describe(arm, r) + ": source should carry SPS/PPS in its first access unit");
  require(r.parsed_idrs == kIdrCount, describe(arm, r) + ": stream should carry one IDR per GOP");
}

} // namespace

RUN_TEST("unit_h264_depacketize_idr_recovery_test", ([] {
           simaai::neat::gst_init_once();
           for (const char* element : {"videotestsrc", "rtph264pay", "rtph264depay", "h264parse"}) {
             require(simaai::neat::element_exists(element),
                     std::string("missing GStreamer element: ") + element);
           }
           const std::string encoder = software_h264_encoder();
           const std::string decoder = software_h264_decoder();
           require(!encoder.empty(), "missing software H.264 encoder (x264enc or openh264enc)");
           require(!decoder.empty(), "missing software H.264 decoder (avdec_h264 or openh264dec)");

           // Control: nothing dropped, so every frame decodes without repetition.
           const ArmResult control = run_arm(encoder, decoder, /*config_interval=*/0,
                                             /*drop_access_units=*/0);
           require_source_shape("control", control);
           require(control.decoded_frames == kFrames && control.end == "eos",
                   describe("control", control) + ": every frame should decode");

           // Default: the consumer misses the first IDR and resumes at the next one.
           const ArmResult repeated = run_arm(encoder, decoder, std::nullopt, kDroppedAccessUnits);
           require_source_shape("default", repeated);
           require(repeated.parsed_idrs_with_sps == kIdrCount,
                   describe("default", repeated) + ": SPS/PPS should precede every IDR");
           require(repeated.decoded_frames == kFrames - kGopFrames && repeated.end == "eos",
                   describe("default", repeated) +
                       ": decoding should resume at the next IDR and keep every later frame");

           // Negative control: without repetition the consumer never sees SPS/PPS again, decodes
           // nothing, and its h264parse fails at EOS for lack of valid frames.
           const ArmResult off =
               run_arm(encoder, decoder, /*config_interval=*/0, kDroppedAccessUnits);
           require_source_shape("off", off);
           require(off.parsed_idrs_with_sps == 1,
                   describe("off", off) + ": SPS/PPS should not repeat");
           require(off.decoded_frames == 0 &&
                       off.end.find("No valid frames found") != std::string::npos,
                   describe("off", off) + ": nothing should decode without SPS/PPS");
         }));

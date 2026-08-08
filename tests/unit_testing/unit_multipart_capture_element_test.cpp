/**
 * Functional test for the private `neatmultipartjpegdemux` element.
 *
 * Runs a real GStreamer pipeline and asserts that each emitted JPEG buffer carries exactly
 * the headers its own part was sent with. This covers the association guarantee up to the
 * decoder boundary: missing headers stay missing, changed values change, and a value never
 * leaks onto the following frame.
 */
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "gst/GstSampleAttributes.h"

#include "test_main.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace {

using simaai::neat::SampleAttributes;

/// Minimal JPEG carrying a real SOF0 so the element can advertise output caps, plus a
/// one-byte marker in the scan data that identifies which frame this is.
std::string make_marked_jpeg(uint16_t width, uint16_t height, uint8_t marker) {
  std::string jpeg;
  auto push = [&jpeg](std::initializer_list<uint8_t> bytes) {
    for (const uint8_t b : bytes) {
      jpeg.push_back(static_cast<char>(b));
    }
  };
  push({0xFF, 0xD8});                   // SOI
  push({0xFF, 0xC0, 0x00, 0x11, 0x08}); // SOF0, len 17, 8-bit
  push({static_cast<uint8_t>(height >> 8), static_cast<uint8_t>(height & 0xFF)});
  push({static_cast<uint8_t>(width >> 8), static_cast<uint8_t>(width & 0xFF)});
  push({0x03}); // 3 components
  push({0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01});
  push({0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00}); // SOS
  jpeg.push_back(static_cast<char>(marker));                    // frame marker inside scan data
  push({0xFF, 0xD9});                                           // EOI
  return jpeg;
}

std::string part(const std::string& boundary, const std::vector<std::string>& header_lines,
                 const std::string& body) {
  std::string out = "--" + boundary + "\r\n";
  out += "Content-Type: image/jpeg\r\n";
  for (const std::string& line : header_lines) {
    out += line + "\r\n";
  }
  out += "\r\n";
  out += body;
  out += "\r\n";
  return out;
}

struct Received {
  uint8_t marker = 0;
  int width = -1;
  int height = -1;
  SampleAttributes attributes;
};

void read_sample_caps(GstSample* sample, Received* received) {
  GstCaps* caps = sample ? gst_sample_get_caps(sample) : nullptr;
  const GstStructure* structure = caps ? gst_caps_get_structure(caps, 0U) : nullptr;
  if (structure && received) {
    (void)gst_structure_get_int(structure, "width", &received->width);
    (void)gst_structure_get_int(structure, "height", &received->height);
  }
}

Received pull_received(GstElement* sink, GstClockTime timeout = 2 * GST_SECOND) {
  GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), timeout);
  require(sample != nullptr, "expected multipart output sample");

  Received received;
  read_sample_caps(sample, &received);
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (buffer) {
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ) == TRUE) {
      if (map.size >= 3U) {
        received.marker = map.data[map.size - 3U];
      }
      gst_buffer_unmap(buffer, &map);
    }
    simaai::neat::gst_internal::read_attributes(buffer, &received.attributes);
  }
  gst_sample_unref(sample);
  return received;
}

void run_pipeline(const std::string& stream, const std::string& boundary,
                  const std::string& capture_headers, std::vector<Received>* out,
                  bool single_stream = false) {
  simaai::neat::gst_init_once();
  require(simaai::neat::element_exists("neatmultipartjpegdemux"),
          "neatmultipartjpegdemux must be registered by Neat's GStreamer init");

  GstElement* pipeline = gst_pipeline_new("capture-test");
  GstElement* src = gst_element_factory_make("appsrc", "src");
  GstElement* demux = gst_element_factory_make("neatmultipartjpegdemux", "demux");
  GstElement* sink = gst_element_factory_make("appsink", "sink");
  require(pipeline && src && demux && sink, "capture-test pipeline elements must be creatable");

  g_object_set(demux, "boundary", boundary.c_str(), "capture-headers", capture_headers.c_str(),
               "single-stream", single_stream ? TRUE : FALSE, nullptr);
  g_object_set(sink, "sync", FALSE, "emit-signals", FALSE, "max-buffers", 64, nullptr);

  gst_bin_add_many(GST_BIN(pipeline), src, demux, sink, nullptr);
  require(gst_element_link_many(src, demux, sink, nullptr) == TRUE,
          "capture-test pipeline must link");

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "capture-test pipeline must reach PLAYING");

  // Feed the whole stream in small chunks so the element's incremental path is exercised.
  constexpr std::size_t kChunk = 13U;
  for (std::size_t off = 0; off < stream.size(); off += kChunk) {
    const std::size_t n = std::min(kChunk, stream.size() - off);
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, n, nullptr);
    GstMapInfo map;
    require(gst_buffer_map(buffer, &map, GST_MAP_WRITE) == TRUE, "chunk buffer must map");
    std::memcpy(map.data, stream.data() + off, n);
    gst_buffer_unmap(buffer, &map);
    require(gst_app_src_push_buffer(GST_APP_SRC(src), buffer) == GST_FLOW_OK,
            "appsrc push must succeed");
  }
  gst_app_src_end_of_stream(GST_APP_SRC(src));

  for (;;) {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    if (!sample) {
      break;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    Received received;
    read_sample_caps(sample, &received);
    if (buffer) {
      GstMapInfo map;
      if (gst_buffer_map(buffer, &map, GST_MAP_READ) == TRUE) {
        // The marker sits immediately before the trailing EOI.
        if (map.size >= 3U) {
          received.marker = map.data[map.size - 3U];
        }
        gst_buffer_unmap(buffer, &map);
      }
      simaai::neat::gst_internal::read_attributes(buffer, &received.attributes);
    }
    out->push_back(received);
    gst_sample_unref(sample);
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

struct ReentrantReset {
  GstElement* demux = nullptr;
  bool changed = false;
};

GstPadProbeReturn reset_capture_from_downstream(GstPad*, GstPadProbeInfo* info,
                                                gpointer user_data) {
  auto* reset = static_cast<ReentrantReset*>(user_data);
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) != 0U && !reset->changed) {
    reset->changed = true;
    // This runs synchronously inside gst_pad_push(). It reproduces the reentrant property reset
    // that used to delete MultipartParser while MultipartParser::feed() was still on the stack.
    g_object_set(reset->demux, "capture-headers", "image-time", nullptr);
  }
  return GST_PAD_PROBE_OK;
}

void push_stream(GstElement* src, const std::string& stream) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, stream.size(), nullptr);
  require(buffer != nullptr, "stream buffer must allocate");
  GstMapInfo map;
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE) == TRUE, "stream buffer must map");
  std::memcpy(map.data, stream.data(), stream.size());
  gst_buffer_unmap(buffer, &map);
  require(gst_app_src_push_buffer(GST_APP_SRC(src), buffer) == GST_FLOW_OK,
          "whole-stream appsrc push must succeed");
}

void test_reentrant_property_reset_is_safe() {
  simaai::neat::gst_init_once();

  GstElement* pipeline = gst_pipeline_new("capture-reentrant-reset-test");
  GstElement* src = gst_element_factory_make("appsrc", "src");
  GstElement* demux = gst_element_factory_make("neatmultipartjpegdemux", "demux");
  GstElement* sink = gst_element_factory_make("appsink", "sink");
  require(pipeline && src && demux && sink, "reentrant-reset elements must be creatable");

  g_object_set(demux, "boundary", "frame", "capture-headers", "image-index", nullptr);
  g_object_set(sink, "sync", FALSE, "emit-signals", FALSE, "max-buffers", 8, nullptr);
  gst_bin_add_many(GST_BIN(pipeline), src, demux, sink, nullptr);
  require(gst_element_link_many(src, demux, sink, nullptr) == TRUE,
          "reentrant-reset pipeline must link");

  ReentrantReset reset{demux, false};
  GstPad* srcpad = gst_element_get_static_pad(demux, "src");
  require(srcpad != nullptr, "demux src pad must exist");
  gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, reset_capture_from_downstream, &reset,
                    nullptr);
  gst_object_unref(srcpad);

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "reentrant-reset pipeline must reach PLAYING");

  // Both parts complete in one parser feed. The first downstream push resets the element while
  // the second completed part is waiting to be pushed.
  std::string first_stream;
  first_stream +=
      part("frame", {"Image-Index: 1", "Image-Time: old-1"}, make_marked_jpeg(32, 32, 0x71));
  first_stream +=
      part("frame", {"Image-Index: 2", "Image-Time: old-2"}, make_marked_jpeg(32, 32, 0x72));
  first_stream += "--frame--\r\n";
  push_stream(src, first_stream);

  const Received first = pull_received(sink);
  const Received second = pull_received(sink);
  require(reset.changed, "downstream probe must perform the reentrant reset");
  require(first.marker == 0x71 && first.attributes == SampleAttributes{{"image-index", "1"}},
          "first already-parsed part must keep its original capture policy");
  require(second.marker == 0x72 && second.attributes == SampleAttributes{{"image-index", "2"}},
          "second already-parsed part must survive the reentrant reset");

  // The reset applies to subsequent input, proving the replacement parser remains usable.
  std::string next_stream;
  next_stream +=
      part("frame", {"Image-Index: 3", "Image-Time: new-3"}, make_marked_jpeg(32, 32, 0x73));
  next_stream += "--frame--\r\n";
  push_stream(src, next_stream);
  gst_app_src_end_of_stream(GST_APP_SRC(src));

  const Received third = pull_received(sink);
  require(third.marker == 0x73 && third.attributes == SampleAttributes{{"image-time", "new-3"}},
          "replacement parser must use the reentrant capture policy");

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

void test_per_frame_association() {
  const std::string boundary = "frame";
  // Frame 1: both headers. Frame 2: index changes, continuation absent. Frame 3: index
  // changes again, continuation present with a different value. Frame 4: index only.
  std::string stream;
  stream += part(boundary, {"Image-Index: 100", "Image-Continuation: true"},
                 make_marked_jpeg(64, 48, 0x11));
  stream += part(boundary, {"Image-Index: 101"}, make_marked_jpeg(64, 48, 0x22));
  stream += part(boundary, {"Image-Index: 102", "Image-Continuation: false"},
                 make_marked_jpeg(64, 48, 0x33));
  stream += part(boundary, {"Image-Index: 103"}, make_marked_jpeg(64, 48, 0x44));
  stream += "--" + boundary + "--\r\n";

  std::vector<Received> received;
  run_pipeline(stream, boundary, "image-index,image-continuation", &received);

  require(received.size() == 4U, "expected 4 frames, got " + std::to_string(received.size()));

  require(received[0].marker == 0x11, "frame 0 marker");
  require(received[0].attributes.at("image-index") == "100", "frame 0 index");
  require(received[0].attributes.at("image-continuation") == "true", "frame 0 continuation");

  require(received[1].marker == 0x22, "frame 1 marker");
  require(received[1].attributes.at("image-index") == "101", "frame 1 index changed");
  require(received[1].attributes.count("image-continuation") == 0U,
          "an absent header must not be inherited from the previous frame");

  require(received[2].marker == 0x33, "frame 2 marker");
  require(received[2].attributes.at("image-index") == "102", "frame 2 index changed");
  require(received[2].attributes.at("image-continuation") == "false",
          "frame 2 continuation must be its own value");

  require(received[3].marker == 0x44, "frame 3 marker");
  require(received[3].attributes.at("image-index") == "103", "frame 3 index changed");
  require(received[3].attributes.count("image-continuation") == 0U,
          "frame 3 must not retain frame 2's continuation value");
}

void test_capture_only_selected_headers() {
  const std::string boundary = "b";
  std::string stream;
  stream +=
      part(boundary, {"Image-Index: 5", "X-Not-Selected: secret"}, make_marked_jpeg(32, 32, 0x55));
  stream += "--" + boundary + "--\r\n";

  std::vector<Received> received;
  run_pipeline(stream, boundary, "image-index", &received);
  require(received.size() == 1U, "expected 1 frame");
  require(received[0].attributes.size() == 1U, "only allowlisted headers may be captured");
  require(received[0].attributes.at("image-index") == "5", "selected header value");
  require(received[0].attributes.count("x-not-selected") == 0U,
          "unselected header must not appear");
}

void test_no_capture_configured_emits_no_attributes() {
  const std::string boundary = "b";
  std::string stream;
  stream += part(boundary, {"Image-Index: 9"}, make_marked_jpeg(32, 32, 0x66));
  stream += "--" + boundary + "--\r\n";

  std::vector<Received> received;
  run_pipeline(stream, boundary, "", &received);
  require(received.size() == 1U, "expected 1 frame");
  require(received[0].attributes.empty(),
          "with no allowlist the element must not attach attributes");
}

void test_dimension_change_renegotiates_caps() {
  const std::string boundary = "frame";
  std::string stream;
  stream += part(boundary, {"Image-Index: 1"}, make_marked_jpeg(64, 48, 0x41));
  stream += part(boundary, {"Image-Index: 2"}, make_marked_jpeg(96, 72, 0x42));
  stream += "--" + boundary + "--\r\n";

  std::vector<Received> received;
  run_pipeline(stream, boundary, "image-index", &received, false);

  require(received.size() == 2U, "dimension-change stream must emit two frames");
  require(received[0].width == 64 && received[0].height == 48,
          "first frame must advertise its own JPEG dimensions");
  require(received[1].width == 96 && received[1].height == 72,
          "single-stream=false must renegotiate changed JPEG dimensions");
}

void test_attribute_mutation_rejects_shared_buffers() {
  simaai::neat::gst_init_once();

  GstBuffer* buffer = gst_buffer_new();
  require(buffer != nullptr, "attribute test buffer must allocate");
  const SampleAttributes original{{"image-index", "9"}};
  require(simaai::neat::gst_internal::write_attributes(buffer, original),
          "initial attributes must be writable");

  GstBuffer* shared_ref = gst_buffer_ref(buffer);
  require(!simaai::neat::gst_internal::clear_attributes(buffer),
          "clearing attributes on a shared buffer must fail without mutating it");
  SampleAttributes observed;
  simaai::neat::gst_internal::read_attributes(buffer, &observed);
  require(observed == original, "failed clear must leave shared-buffer attributes unchanged");

  require(!simaai::neat::gst_internal::write_attributes(buffer, {}),
          "writing an empty map on a shared buffer must not bypass writability");
  simaai::neat::gst_internal::read_attributes(buffer, &observed);
  require(observed == original, "failed empty write must leave shared-buffer attributes unchanged");

  gst_buffer_unref(shared_ref);
  require(simaai::neat::gst_internal::clear_attributes(buffer),
          "attribute clear must succeed after the buffer becomes writable");
  simaai::neat::gst_internal::read_attributes(buffer, &observed);
  require(observed.empty(), "successful clear must remove all attributes");
  gst_buffer_unref(buffer);
}

} // namespace

RUN_TEST("unit_multipart_capture_element", [] {
  test_per_frame_association();
  test_capture_only_selected_headers();
  test_no_capture_configured_emits_no_attributes();
  test_dimension_change_renegotiates_caps();
  test_reentrant_property_reset_is_safe();
  test_attribute_mutation_rejects_shared_buffers();
})

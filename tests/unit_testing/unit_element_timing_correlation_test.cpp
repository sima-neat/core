#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "pipeline/graph/GraphDetail.h"
#include "pipeline/internal/Diagnostics.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

namespace {

GstBuffer* buffer(std::int64_t id = -1, const char* stream = "stream0",
                  GstClockTime pts = GST_CLOCK_TIME_NONE) {
  GstBuffer* result = gst_buffer_new_allocate(nullptr, 16, nullptr);
  require(result != nullptr, "buffer allocation failed");
  GST_BUFFER_PTS(result) = pts;
  if (id >= 0) {
    auto* meta = gst_buffer_add_custom_meta(result, "GstSimaMeta");
    require(meta != nullptr, "GstSimaMeta is not registered");
    gst_structure_set(gst_custom_meta_get_structure(meta), "frame-id", G_TYPE_INT64,
                      static_cast<gint64>(id), "stream-id", G_TYPE_STRING, stream, nullptr);
  }
  return result;
}

GstBuffer* request_buffer(std::int64_t id, std::int64_t sequence, std::uint64_t timestamp,
                          GstClockTime pts = GST_CLOCK_TIME_NONE) {
  GstBuffer* result = buffer(id, "stream0", pts);
  auto* meta = gst_buffer_get_custom_meta(result, "GstSimaMeta");
  gst_structure_set(gst_custom_meta_get_structure(meta), "input-seq", G_TYPE_INT64,
                    static_cast<gint64>(sequence), "timestamp", G_TYPE_UINT64,
                    static_cast<guint64>(timestamp), nullptr);
  return result;
}

struct Fixture {
  GstElement* pipeline = gst_pipeline_new(nullptr);
  GstPad* sink = nullptr;
  GstPad* src = nullptr;
  std::shared_ptr<simaai::neat::pipeline_internal::DiagCtx> diag =
      std::make_shared<simaai::neat::pipeline_internal::DiagCtx>();
  simaai::neat::pipeline_internal::ElementTimingCounters* timing = nullptr;

  explicit Fixture(const char* factory = "capsfilter") {
    auto* element = gst_element_factory_make(factory, "timed");
    auto* output = gst_element_factory_make("fakesink", "output");
    require(pipeline && element && output, "test elements unavailable");
    g_object_set(output, "sync", FALSE, "async", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), element, output, nullptr);
    require(gst_element_link(element, output), "test elements did not link");
    simaai::neat::attach_element_timing_probes(pipeline, diag, true);
    for (auto& row : diag->element_timings)
      if (row->element_name == "timed")
        timing = row.get();
    require(timing != nullptr, "element timing probes were not attached");
    sink = gst_element_get_static_pad(element, "sink");
    src = gst_element_get_static_pad(element, "src");
    gst_pad_add_probe(
        sink, GST_PAD_PROBE_TYPE_BUFFER,
        [](GstPad*, GstPadProbeInfo*, gpointer) { return GST_PAD_PROBE_DROP; }, nullptr, nullptr);
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "test pipeline failed to start");
    gst_pad_send_event(sink, gst_event_new_stream_start("test"));
    GstCaps* caps = gst_caps_new_empty_simple("application/octet-stream");
    gst_pad_send_event(sink, gst_event_new_caps(caps));
    gst_caps_unref(caps);
    GstSegment segment;
    gst_segment_init(&segment, GST_FORMAT_TIME);
    gst_pad_send_event(sink, gst_event_new_segment(&segment));
  }

  ~Fixture() {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    if (sink)
      gst_object_unref(sink);
    if (src)
      gst_object_unref(src);
    gst_object_unref(pipeline);
  }

  void input(GstBuffer* value) {
    require(gst_pad_chain(sink, value) == GST_FLOW_OK, "input probe failed");
  }

  void output(GstBuffer* value) {
    g_usleep(1000); // Ensure a measurable interval with the microsecond timing clock.
    require(gst_pad_push(src, value) == GST_FLOW_OK, "output probe failed");
  }

  std::uint64_t samples() const {
    return timing->samples.load();
  }
};

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    const char* original = g_getenv("SIMA_GST_ELEMENT_TIMINGS");
    const bool had_original = original != nullptr;
    const std::string saved = original ? original : "";
    auto* pipeline = gst_pipeline_new(nullptr);
    auto* element = gst_element_factory_make("capsfilter", "option-test");
    require(pipeline && element, "option test elements unavailable");
    gst_bin_add(GST_BIN(pipeline), element);
    auto diag = std::make_shared<simaai::neat::pipeline_internal::DiagCtx>();
    g_setenv("SIMA_GST_ELEMENT_TIMINGS", "0", TRUE);
    simaai::neat::attach_element_timing_probes(pipeline, diag, true);
    require(diag->element_timings.empty() && diag->element_pad_timings.empty(),
            "explicit timing disable did not override enabled options");
    g_setenv("SIMA_GST_ELEMENT_TIMINGS", "1", TRUE);
    simaai::neat::attach_element_timing_probes(pipeline, diag, false);
    require(!diag->element_timings.empty(),
            "explicit timing enable did not override disabled options");
    gst_object_unref(pipeline);

    Fixture test;
    for (int i = 0; i < 3; ++i)
      test.input(buffer(-1, "stream0", 100 * (i + 1)));
    test.output(buffer(-1, "stream0", 300));
    require(test.samples() == 0, "many-to-one output borrowed an unrelated input timestamp");

    auto* same = buffer();
    test.input(gst_buffer_ref(same));
    test.output(same);
    require(test.samples() == 1, "same-buffer correlation was lost");

    test.input(buffer(10));
    test.input(buffer(11));
    test.output(buffer(11));
    test.output(buffer(10));
    require(test.samples() == 3, "reordered exact identities were not correlated");
    test.output(buffer(12));
    test.output(buffer(11));
    require(test.samples() == 3, "unmatched or duplicate output borrowed another timestamp");

    test.input(buffer(22, "a"));
    test.output(buffer(22, "b"));
    require(test.samples() == 3, "same frame number from another stream was correlated");
    test.output(buffer(22, "a"));
    require(test.samples() == 4, "matching stream identity was not correlated");

    same = buffer(100);
    test.input(gst_buffer_ref(same));
    test.output(same);
    require(test.samples() == 5, "same-buffer metadata path lost its timing");
    require(test.timing->pending.empty(), "same-buffer success left an obsolete identity");

    test.input(buffer(200, "stream0", 1000));
    test.output(buffer(200, "stream0", 2000));
    require(test.samples() == 5, "reused frame ID with a different PTS was correlated");
    test.output(buffer(200, "stream0", 1000));
    require(test.samples() == 6, "matching frame ID and PTS were not correlated");

    std::uint64_t input_count = 0, output_count = 0;
    for (const auto& pad : test.diag->element_pad_timings) {
      if (pad->element_name != "timed")
        continue;
      (pad->is_sink ? input_count : output_count) += pad->samples.load();
    }
    require(input_count == 9 && output_count == 11,
            "unequal input/output cadence counts changed with latency correlation");
    Fixture convert("videoconvert");
    convert.input(buffer(-1, "stream0", 10));
    convert.input(buffer(-1, "stream0", 20));
    convert.output(buffer(-1, "stream0", 20));
    convert.output(buffer(-1, "stream0", 10));
    require(convert.samples() == 2, "replacement buffers lost unique PTS correlation");
    convert.input(buffer(-1, "stream0", 30));
    convert.input(buffer(-1, "stream0", 30));
    convert.output(buffer(-1, "stream0", 30));
    convert.input(buffer(-1, "stream0", 30));
    convert.output(buffer(-1, "stream0", 30));
    convert.input(buffer());
    convert.output(buffer());
    require(convert.samples() == 2, "ambiguous or absent PTS produced a timing sample");
    convert.input(buffer(40, "stream0", 40));
    convert.output(buffer(-1, "stream0", 40));
    require(convert.samples() == 3 && convert.timing->pending.empty(),
            "replacement PTS success left obsolete metadata correlation");
    convert.input(buffer(-1, "stream0", 50));
    GstSegment segment;
    gst_segment_init(&segment, GST_FORMAT_TIME);
    gst_pad_send_event(convert.sink, gst_event_new_segment(&segment));
    convert.output(buffer(-1, "stream0", 50));
    require(convert.samples() == 3 && convert.timing->pending_pts.empty(),
            "segment change retained an earlier timing identity");
    Fixture request;
    request.input(request_buffer(1, 10, 100, 100));
    request.output(request_buffer(1, 10, 100));
    require(request.samples() == 1, "request metadata did not survive missing native PTS");
    request.input(request_buffer(2, 11, 200, 200));
    request.output(request_buffer(2, 11, 200, 900));
    require(request.samples() == 2, "native PTS overrode preserved request metadata");

    request.input(request_buffer(3, 12, 300, 300));
    request.output(request_buffer(3, 13, 300, 300));
    request.output(request_buffer(3, 12, 301, 300));
    request.output(buffer(3, "stream0", 300));
    require(request.samples() == 2, "conflicting or missing request metadata was correlated");
    request.output(request_buffer(3, 12, 300));
    require(request.samples() == 3, "matching request was lost after unrelated outputs");

    request.input(request_buffer(4, 14, 400));
    request.input(request_buffer(4, 15, 400));
    request.output(request_buffer(4, 15, 400));
    request.output(request_buffer(4, 14, 400));
    require(request.samples() == 5, "reordered requests with repeated frame IDs were not matched");
    request.input(request_buffer(5, 16, 500));
    request.input(request_buffer(5, 16, 500));
    request.output(request_buffer(5, 16, 500));
    request.input(request_buffer(5, 16, 500));
    request.output(request_buffer(5, 16, 500));
    require(request.samples() == 5,
            "delayed ambiguous output borrowed a newer request's start time");
    gst_pad_send_event(request.sink, gst_event_new_segment(&segment));
    require(request.timing->pending.empty(), "segment change retained ambiguous requests");
    request.input(request_buffer(5, 16, 500));
    request.output(request_buffer(5, 16, 500));
    require(request.samples() == 6, "new segment could not reuse a prior request identity");
    Fixture convert_request("videoconvert");
    convert_request.input(request_buffer(1, 1, 10, 10));
    convert_request.input(request_buffer(1, 1, 10, 10));
    convert_request.input(request_buffer(1, 1, 10, 20));
    convert_request.output(request_buffer(1, 1, 10, 10));
    require(convert_request.samples() == 0, "duplicate PTS revived an ambiguous metadata request");

    Fixture shared_pts("videoconvert");
    shared_pts.input(request_buffer(7, 20, 600, 600));
    shared_pts.input(request_buffer(8, 21, 600, 600));
    shared_pts.output(request_buffer(8, 21, 600, 600));
    shared_pts.output(request_buffer(7, 20, 600, 600));
    require(shared_pts.samples() == 2 && shared_pts.timing->pending.empty(),
            "repeated PTS discarded distinct exact request identities");
    shared_pts.output(buffer(-1, "stream0", 600));
    require(shared_pts.samples() == 2, "ambiguous PTS fallback became usable after exact matches");

    for (const char* factory : {"capsfilter", "videoconvert"}) {
      Fixture bounded(factory);
      bounded.timing->max_pending = 1;
      const bool metadata = std::string(factory) == "capsfilter";
      auto identity = [metadata](int id) {
        return metadata ? request_buffer(id, id, id) : buffer(-1, "stream0", id);
      };
      bounded.input(identity(1));
      bounded.input(identity(2)); // Exceeds the bounded correlation history.
      bounded.output(identity(1));
      bounded.input(identity(2));
      bounded.output(identity(2));
      require(bounded.samples() == 0, "overflow allowed a delayed output to match a newer input");
      gst_pad_send_event(bounded.sink, gst_event_new_segment(&segment));
      bounded.input(identity(2));
      bounded.output(identity(2));
      require(bounded.samples() == 1, "segment change did not recover bounded timing history");
    }
    if (had_original)
      g_setenv("SIMA_GST_ELEMENT_TIMINGS", saved.c_str(), TRUE);
    else
      g_unsetenv("SIMA_GST_ELEMENT_TIMINGS");
    std::cout << "[OK] unit_element_timing_correlation_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}

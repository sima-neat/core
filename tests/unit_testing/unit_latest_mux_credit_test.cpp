#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/LatestByStreamFrameTap.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kLoanValidField = "neat-latest-mux-loan-valid";
constexpr const char* kLoanNamespaceField = "neat-latest-mux-loan-namespace";
constexpr const char* kLoanStreamIdField = "neat-latest-mux-loan-stream-id";
constexpr const char* kLoanFrameIdField = "neat-latest-mux-loan-frame-id";

struct MuxLoanKey {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;
};

simaai::neat::Sample make_gst_sample_backed_sample(std::optional<MuxLoanKey> key) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

  if (key.has_value()) {
    GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    require(meta != nullptr, "failed to attach GstSimaMeta");
    GstStructure* s = gst_custom_meta_get_structure(meta);
    require(s != nullptr, "failed to access GstSimaMeta structure");
    gst_structure_set(s, kLoanValidField, G_TYPE_BOOLEAN, TRUE, kLoanNamespaceField, G_TYPE_UINT64,
                      static_cast<guint64>(key->namespace_id), kLoanStreamIdField, G_TYPE_STRING,
                      key->stream_id.c_str(), kLoanFrameIdField, G_TYPE_INT64,
                      static_cast<gint64>(key->frame_id), nullptr);
  }

  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, 4, "height", G_TYPE_INT, 4, nullptr);
  require(caps != nullptr, "failed to allocate GstCaps");
  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(gst_sample != nullptr, "failed to allocate GstSample");

  simaai::neat::Tensor tensor;
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(gst_sample);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.shape = {16};
  gst_sample_unref(gst_sample);

  simaai::neat::Sample sample = simaai::neat::sample_from_tensors({tensor});
  sample.stream_id = "fallback-stream";
  sample.frame_id = 7;
  sample.media_type = "video/x-raw";
  sample.payload_tag = "NV12";
  sample.format = "NV12";
  return sample;
}

GstBuffer* make_mux_input_buffer(std::int64_t frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate latest-mux input buffer");
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "failed to attach latest-mux input metadata");
  GstStructure* structure = gst_custom_meta_get_structure(meta);
  require(structure != nullptr, "failed to access latest-mux input metadata");
  gst_structure_set(structure, "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id), "input-seq",
                    G_TYPE_INT64, static_cast<gint64>(frame_id), "orig-input-seq", G_TYPE_INT64,
                    static_cast<gint64>(frame_id), nullptr);
  GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_id) * GST_MSECOND;
  GST_BUFFER_DURATION(buffer) = GST_MSECOND;
  return buffer;
}

GstBuffer* make_mux_input_buffer_with_timing(std::int64_t frame_id, GstClockTime pts,
                                             GstClockTime dts, GstClockTime duration) {
  GstBuffer* buffer = make_mux_input_buffer(frame_id);
  GST_BUFFER_PTS(buffer) = pts;
  GST_BUFFER_DTS(buffer) = dts;
  GST_BUFFER_DURATION(buffer) = duration;
  return buffer;
}

GstBuffer* make_encoded_tap_buffer(const std::vector<std::uint8_t>& bytes, GstClockTime pts,
                                   GstClockTime dts, GstClockTime duration) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes.size(), nullptr);
  require(buffer != nullptr, "failed to allocate encoded-tap buffer");
  GstMapInfo map = GST_MAP_INFO_INIT;
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE) == TRUE, "failed to map encoded-tap buffer");
  std::copy(bytes.begin(), bytes.end(), map.data);
  gst_buffer_unmap(buffer, &map);
  GST_BUFFER_PTS(buffer) = pts;
  GST_BUFFER_DTS(buffer) = dts;
  GST_BUFFER_DURATION(buffer) = duration;
  return buffer;
}

GstBuffer* make_terminal_identity_buffer(const char* stream_id,
                                         std::optional<std::int64_t> frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate terminal identity buffer");
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "failed to attach terminal identity metadata");
  GstStructure* structure = gst_custom_meta_get_structure(meta);
  require(structure != nullptr, "failed to access terminal identity metadata");
  gst_structure_set(structure, "stream-id", G_TYPE_STRING, stream_id, "orig-stream-id",
                    G_TYPE_STRING, stream_id, nullptr);
  if (frame_id.has_value()) {
    gst_structure_set(structure, "frame-id", G_TYPE_INT64, static_cast<gint64>(*frame_id), nullptr);
  }
  return buffer;
}

GstBuffer*
make_terminal_sequence_buffer(const char* stream_id, std::optional<std::int64_t> frame_id,
                              std::optional<std::int64_t> input_seq,
                              std::optional<std::int64_t> orig_input_seq = std::nullopt) {
  GstBuffer* buffer = make_terminal_identity_buffer(stream_id, frame_id);
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  require(structure != nullptr, "failed to access terminal sequence metadata");
  if (input_seq.has_value()) {
    gst_structure_set(structure, "input-seq", G_TYPE_INT64, static_cast<gint64>(*input_seq),
                      nullptr);
  }
  if (orig_input_seq.has_value()) {
    gst_structure_set(structure, "orig-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(*orig_input_seq), nullptr);
  }
  return buffer;
}

GstBuffer* make_terminal_stale_private_key_buffer(const char* public_stream_id,
                                                  std::int64_t public_frame_id,
                                                  std::uint64_t mux_namespace,
                                                  const char* stale_private_stream_id,
                                                  std::int64_t stale_private_frame_id) {
  GstBuffer* buffer = make_terminal_identity_buffer(public_stream_id, public_frame_id);
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  require(structure != nullptr, "failed to access stale private-key terminal metadata");
  gst_structure_set(structure, kLoanValidField, G_TYPE_BOOLEAN, TRUE, kLoanNamespaceField,
                    G_TYPE_UINT64, static_cast<guint64>(mux_namespace), kLoanStreamIdField,
                    G_TYPE_STRING, stale_private_stream_id, kLoanFrameIdField, G_TYPE_INT64,
                    static_cast<gint64>(stale_private_frame_id), nullptr);
  return buffer;
}

void require_canonical_terminal_loan(GstBuffer* terminal, const std::string& expected_stream,
                                     std::int64_t expected_frame_id,
                                     std::uint64_t expected_namespace, GstClockTime expected_pts,
                                     GstClockTime expected_dts, GstClockTime expected_duration) {
  require(terminal && GST_BUFFER_PTS(terminal) == expected_pts &&
              GST_BUFFER_DTS(terminal) == expected_dts &&
              GST_BUFFER_DURATION(terminal) == expected_duration,
          "stream fallback should restore canonical loan timing");
  GstCustomMeta* meta = gst_buffer_get_custom_meta(terminal, "GstSimaMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  require(structure != nullptr, "stream fallback should restore canonical terminal metadata");
  const char* stream_id = gst_structure_get_string(structure, "stream-id");
  const char* orig_stream_id = gst_structure_get_string(structure, "orig-stream-id");
  const char* private_stream_id = gst_structure_get_string(structure, kLoanStreamIdField);
  gint64 frame_id = -1;
  gint64 input_seq = -1;
  gint64 orig_input_seq = -1;
  gint64 private_frame_id = -1;
  gint64 private_input_seq = -1;
  gint64 private_orig_input_seq = -1;
  guint64 private_namespace = 0;
  gboolean private_valid = TRUE;
  require(
      stream_id && std::string(stream_id) == expected_stream && orig_stream_id &&
          std::string(orig_stream_id) == expected_stream && private_stream_id &&
          std::string(private_stream_id) == expected_stream &&
          gst_structure_get_int64(structure, "frame-id", &frame_id) == TRUE &&
          gst_structure_get_int64(structure, "input-seq", &input_seq) == TRUE &&
          gst_structure_get_int64(structure, "orig-input-seq", &orig_input_seq) == TRUE &&
          gst_structure_get_int64(structure, kLoanFrameIdField, &private_frame_id) == TRUE &&
          gst_structure_get_int64(structure, "neat-latest-mux-loan-input-seq",
                                  &private_input_seq) == TRUE &&
          gst_structure_get_int64(structure, "neat-latest-mux-loan-orig-input-seq",
                                  &private_orig_input_seq) == TRUE &&
          gst_structure_get_uint64(structure, kLoanNamespaceField, &private_namespace) == TRUE &&
          gst_structure_get_boolean(structure, kLoanValidField, &private_valid) == TRUE &&
          frame_id == expected_frame_id && input_seq == expected_frame_id &&
          orig_input_seq == expected_frame_id && private_frame_id == expected_frame_id &&
          private_input_seq == expected_frame_id && private_orig_input_seq == expected_frame_id &&
          private_namespace == expected_namespace && private_valid == FALSE,
      "stream fallback should replace stale public/private identity with the selected loan");
}

void push_mux_input(GstElement* appsrc, std::int64_t frame_id) {
  require(gst_app_src_push_buffer(GST_APP_SRC(appsrc), make_mux_input_buffer(frame_id)) ==
              GST_FLOW_OK,
          "latest-mux appsrc push failed");
}

GstSample* pull_mux_output(GstElement* appsink, GstClockTime timeout) {
  return gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), timeout);
}

struct LatestMuxPipeline {
  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  GstElement* mux = nullptr;
  GstElement* dropper = nullptr;
  GstElement* queue = nullptr;
  GstElement* appsink = nullptr;
  GstPad* mux_sink = nullptr;
  std::uint64_t mux_namespace = 0;
};

LatestMuxPipeline make_latest_mux_pipeline(const char* pipeline_name, const char* stream_id,
                                           int stream_inflight_limit = 1,
                                           bool lifetime_guard_enabled = true,
                                           int max_inflight_total = 0) {
  LatestMuxPipeline fixture;
  fixture.pipeline = gst_pipeline_new(pipeline_name);
  fixture.appsrc = gst_element_factory_make("appsrc", nullptr);
  fixture.mux = gst_element_factory_make("neatlatestbystreammux", nullptr);
  fixture.dropper = gst_element_factory_make("identity", nullptr);
  fixture.queue = gst_element_factory_make("queue", nullptr);
  fixture.appsink = gst_element_factory_make("appsink", nullptr);
  require(fixture.pipeline && fixture.appsrc && fixture.mux && fixture.dropper && fixture.queue &&
              fixture.appsink,
          "failed to construct queued latest-mux test pipeline");

  g_object_set(fixture.appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);
  const std::string inflight_limit = std::to_string(stream_inflight_limit);
  g_object_set(fixture.mux, "stream-ids", stream_id, "stream-inflight-limits",
               inflight_limit.c_str(), "max-inflight-total", max_inflight_total, nullptr);
  require(simaai::neat::pipeline_internal::set_latest_by_stream_mux_lifetime_guard_enabled(
              fixture.mux, lifetime_guard_enabled),
          "failed to configure latest-mux lifetime guard before start");
  g_object_set(fixture.queue, "max-size-buffers", 1U, "max-size-bytes", 0U, "max-size-time",
               static_cast<guint64>(0), "leaky", 0, nullptr);
  g_object_set(fixture.appsink, "sync", FALSE, "max-buffers", 4U, "drop", FALSE,
               "enable-last-sample", FALSE, nullptr);

  GstCaps* caps =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8", "width", G_TYPE_INT, 4,
                          "height", G_TYPE_INT, 4, "framerate", GST_TYPE_FRACTION, 1, 1, nullptr);
  require(caps != nullptr, "failed to allocate queued latest-mux input caps");
  gst_app_src_set_caps(GST_APP_SRC(fixture.appsrc), caps);
  gst_caps_unref(caps);

  gst_bin_add_many(GST_BIN(fixture.pipeline), fixture.appsrc, fixture.mux, fixture.dropper,
                   fixture.queue, fixture.appsink, nullptr);
  fixture.mux_sink = gst_element_request_pad_simple(fixture.mux, "sink_0");
  GstPad* appsrc_src = gst_element_get_static_pad(fixture.appsrc, "src");
  require(fixture.mux_sink && appsrc_src &&
              gst_pad_link(appsrc_src, fixture.mux_sink) == GST_PAD_LINK_OK,
          "failed to link appsrc to queued latest-mux request pad");
  gst_object_unref(appsrc_src);
  require(gst_element_link_many(fixture.mux, fixture.dropper, fixture.queue, fixture.appsink,
                                nullptr) == TRUE,
          "failed to link queued latest-mux output path");

  fixture.mux_namespace = simaai::neat::latest_by_stream_mux_namespace(fixture.mux);
  require(fixture.mux_namespace != 0U, "latest-mux should expose a nonzero loan namespace");
  require(gst_element_set_state(fixture.pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start queued latest-mux test pipeline");
  return fixture;
}

void stop_latest_mux_pipeline(LatestMuxPipeline* fixture) {
  if (!fixture || !fixture->pipeline) {
    return;
  }
  (void)gst_app_src_end_of_stream(GST_APP_SRC(fixture->appsrc));
  require(gst_element_set_state(fixture->pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "failed to stop queued latest-mux test pipeline");
  gst_element_release_request_pad(fixture->mux, fixture->mux_sink);
  gst_object_unref(fixture->mux_sink);
  gst_object_unref(fixture->pipeline);
  fixture->pipeline = nullptr;
  fixture->mux_sink = nullptr;
}

void release_terminal_loan(const LatestMuxPipeline& fixture, const char* stream_id,
                           std::int64_t frame_id) {
  GstBuffer* terminal = make_terminal_identity_buffer(stream_id, frame_id);
  const bool released =
      simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
          terminal, fixture.mux_namespace);
  gst_buffer_unref(terminal);
  require(released, "namespace-qualified terminal output should release the mux loan");
}

void test_encoded_frame_tap_owns_au_and_timing() {
  constexpr GstClockTime kPts = 9001001;
  constexpr GstClockTime kDts = 8001001;
  constexpr GstClockTime kDuration = 50000000;
  const std::vector<std::uint8_t> expected = {0x00, 0x00, 0x00, 0x01, 0x65, 0x12, 0x34};
  GstCaps* caps =
      gst_caps_from_string("video/x-h264,stream-format=(string)byte-stream,alignment=(string)au");
  require(caps != nullptr, "failed to allocate encoded-tap caps");
  GstBuffer* source = make_encoded_tap_buffer(expected, kPts, kDts, kDuration);

  std::size_t callback_count = 0;
  simaai::neat::Sample captured;
  simaai::neat::set_latest_by_stream_encoded_frame_callback([&](simaai::neat::Sample sample) {
    ++callback_count;
    captured = std::move(sample);
  });
  require(simaai::neat::pipeline_internal::dispatch_latest_by_stream_encoded_frame_for_buffer(
              source, caps, "stream17"),
          "valid encoded tap dispatch should succeed");
  simaai::neat::clear_latest_by_stream_encoded_frame_callback();

  GstMapInfo overwrite = GST_MAP_INFO_INIT;
  require(gst_buffer_map(source, &overwrite, GST_MAP_WRITE) == TRUE,
          "failed to remap encoded-tap source buffer");
  std::fill(overwrite.data, overwrite.data + overwrite.size, 0xEE);
  gst_buffer_unmap(source, &overwrite);
  gst_buffer_unref(source);
  gst_caps_unref(caps);

  require(callback_count == 1U, "encoded tap should deliver exactly one callback");
  require(captured.stream_id == "stream17", "encoded tap should preserve stream identity");
  require(captured.pts_ns == static_cast<std::int64_t>(kPts), "encoded tap should preserve PTS");
  require(captured.dts_ns == static_cast<std::int64_t>(kDts), "encoded tap should preserve DTS");
  require(captured.duration_ns == static_cast<std::int64_t>(kDuration),
          "encoded tap should preserve duration");
  require(captured.caps_string.find("video/x-h264") != std::string::npos,
          "encoded tap should preserve caps");
  require(captured.tensors.size() == 1U && captured.tensors.front().storage,
          "encoded tap should return one owned payload tensor");
  const simaai::neat::Mapping payload = captured.tensors.front().map_read();
  require(payload.data != nullptr && payload.size_bytes == expected.size(),
          "encoded tap payload has the wrong size");
  const auto* payload_bytes = static_cast<const std::uint8_t*>(payload.data);
  require(std::equal(expected.begin(), expected.end(), payload_bytes),
          "encoded tap payload must outlive and not alias the source GstBuffer");
}

void test_clear_encoded_frame_tap_waits_for_inflight_callback() {
  std::mutex mutex;
  std::condition_variable condition;
  bool callback_entered = false;
  bool release_callback = false;
  std::atomic<std::size_t> callback_count{0};

  simaai::neat::set_latest_by_stream_encoded_frame_callback([&](simaai::neat::Sample) {
    callback_count.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lock(mutex);
    callback_entered = true;
    condition.notify_all();
    condition.wait(lock, [&] { return release_callback; });
  });

  std::thread dispatch_thread([&] {
    GstCaps* caps =
        gst_caps_from_string("video/x-h264,stream-format=(string)byte-stream,alignment=(string)au");
    GstBuffer* buffer = make_encoded_tap_buffer({0x00, 0x00, 0x01, 0x65}, 101, 99, 33);
    simaai::neat::pipeline_internal::dispatch_latest_by_stream_encoded_frame_for_buffer(
        buffer, caps, "stream0");
    gst_buffer_unref(buffer);
    gst_caps_unref(caps);
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    require(condition.wait_for(lock, std::chrono::seconds(2), [&] { return callback_entered; }),
            "encoded tap callback did not enter");
  }

  std::thread release_thread([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
      std::lock_guard<std::mutex> lock(mutex);
      release_callback = true;
    }
    condition.notify_all();
  });
  const auto clear_start = std::chrono::steady_clock::now();
  simaai::neat::clear_latest_by_stream_encoded_frame_callback();
  const auto clear_elapsed = std::chrono::steady_clock::now() - clear_start;
  dispatch_thread.join();
  release_thread.join();

  require(clear_elapsed >= std::chrono::milliseconds(50),
          "clear must wait until every admitted encoded callback has returned");

  GstCaps* caps =
      gst_caps_from_string("video/x-h264,stream-format=(string)byte-stream,alignment=(string)au");
  GstBuffer* buffer = make_encoded_tap_buffer({0x00, 0x00, 0x01, 0x41}, 201, 199, 33);
  simaai::neat::pipeline_internal::dispatch_latest_by_stream_encoded_frame_for_buffer(buffer, caps,
                                                                                      "stream0");
  gst_buffer_unref(buffer);
  gst_caps_unref(caps);
  require(callback_count.load(std::memory_order_relaxed) == 1U,
          "clear must prevent callbacks from being admitted afterward");
}

void test_encoded_frame_tap_copy_failure_posts_pipeline_error() {
  GstElement* pipeline = gst_pipeline_new("encoded-tap-failure-pipeline");
  GstElement* appsrc = gst_element_factory_make("appsrc", "encoded-tap-failure-source");
  GstElement* capsfilter = gst_element_factory_make("capsfilter", "encoded-tap-failure-caps");
  GstElement* fakesink = gst_element_factory_make("fakesink", "encoded-tap-failure-sink");
  require(pipeline && appsrc && capsfilter && fakesink,
          "failed to construct encoded-tap failure pipeline");

  GstCaps* caps =
      gst_caps_from_string("video/x-h264,stream-format=(string)byte-stream,alignment=(string)au");
  require(caps != nullptr, "failed to allocate encoded-tap failure caps");
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  g_object_set(capsfilter, "caps", caps, nullptr);
  g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);
  g_object_set(fakesink, "sync", FALSE, nullptr);
  gst_bin_add_many(GST_BIN(pipeline), appsrc, capsfilter, fakesink, nullptr);
  require(gst_element_link_many(appsrc, capsfilter, fakesink, nullptr) == TRUE,
          "failed to link encoded-tap failure pipeline");

  std::atomic<std::size_t> callback_count{0};
  simaai::neat::set_latest_by_stream_encoded_frame_callback(
      [&](simaai::neat::Sample) { callback_count.fetch_add(1, std::memory_order_relaxed); });
  GstPad* tap_pad = gst_element_get_static_pad(capsfilter, "src");
  require(tap_pad != nullptr, "failed to get encoded-tap failure pad");
  require(simaai::neat::pipeline_internal::attach_latest_by_stream_encoded_frame_tap_probe(
              tap_pad, "failure-stream") != 0,
          "failed to attach production encoded-tap probe");
  gst_object_unref(tap_pad);

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start encoded-tap failure pipeline");
  // An empty AU exercises the same exception-contained copy failure path used
  // for GstBuffer map failures and allocation failures, without relying on a
  // platform allocator's interpretation of GST_MEMORY_FLAG_NOT_MAPPABLE.
  GstBuffer* buffer = gst_buffer_new();
  require(buffer != nullptr, "failed to allocate encoded-tap failure buffer");
  GST_BUFFER_PTS(buffer) = 1;
  GST_BUFFER_DURATION(buffer) = GST_MSECOND;
  (void)gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);

  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* message = gst_bus_timed_pop_filtered(bus, 2 * GST_SECOND, GST_MESSAGE_ERROR);
  require(message != nullptr,
          "an encoded-AU copy/map failure must post a fatal pipeline error, not silently drop");
  GError* gst_error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &gst_error, &debug);
  require(gst_error != nullptr &&
              std::string(gst_error->message).find("Encoded-frame tap failed") != std::string::npos,
          "encoded-tap failure should carry an actionable bus error");
  if (gst_error) {
    g_error_free(gst_error);
  }
  g_free(debug);
  gst_message_unref(message);
  gst_object_unref(bus);
  require(callback_count.load(std::memory_order_relaxed) == 0U,
          "invalid encoded AU must not invoke the subscriber with partial data");

  simaai::neat::clear_latest_by_stream_encoded_frame_callback();
  (void)gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_caps_unref(caps);
  gst_object_unref(pipeline);
}

void test_terminal_release_restores_original_pts() {
  constexpr const char* kStreamId = "timing-stream";
  constexpr GstClockTime kPts = 123456789;
  constexpr GstClockTime kDts = 120000000;
  constexpr GstClockTime kDuration = 50000000;
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-timing-restore-pipeline", kStreamId);

  GstBuffer* input = make_mux_input_buffer_with_timing(21, kPts, kDts, kDuration);
  require(gst_app_src_push_buffer(GST_APP_SRC(fixture.appsrc), input) == GST_FLOW_OK,
          "failed to push latest-mux timing fixture");
  GstSample* output = pull_mux_output(fixture.appsink, GST_SECOND);
  require(output != nullptr, "latest-mux should emit the timing fixture");

  GstBuffer* terminal = make_terminal_identity_buffer(kStreamId, 21);
  GST_BUFFER_PTS(terminal) = 999;
  GST_BUFFER_DTS(terminal) = GST_CLOCK_TIME_NONE;
  GST_BUFFER_DURATION(terminal) = 1;
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              terminal, fixture.mux_namespace),
          "terminal completion should resolve the timing fixture loan");
  require(GST_BUFFER_PTS(terminal) == kPts,
          "terminal completion should restore the decoder input PTS");
  require(GST_BUFFER_DTS(terminal) == kDts,
          "terminal completion should restore the decoder input DTS");
  require(GST_BUFFER_DURATION(terminal) == kDuration,
          "terminal completion should restore the decoder input duration");
  simaai::neat::Sample restored;
  simaai::neat::restore_sample_timing_from_gst_buffer(terminal, &restored);
  require(restored.pts_ns == static_cast<std::int64_t>(kPts),
          "restored Sample should carry the original PTS");
  require(restored.dts_ns == static_cast<std::int64_t>(kDts),
          "restored Sample should carry the original DTS");
  require(restored.duration_ns == static_cast<std::int64_t>(kDuration),
          "restored Sample should carry the original duration");
  gst_buffer_unref(terminal);
  gst_sample_unref(output);

  push_mux_input(fixture.appsrc, 22);
  GstSample* next = pull_mux_output(fixture.appsink, GST_SECOND);
  require(next != nullptr, "timing restoration should also release mux admission credit");
  release_terminal_loan(fixture, kStreamId, 22);
  gst_sample_unref(next);
  stop_latest_mux_pipeline(&fixture);
}

void test_replacing_chain_uses_namespace_fifo_timing() {
  constexpr const char* kStreamId = "replacing-terminal-stream";
  LatestMuxPipeline fixture = make_latest_mux_pipeline("latest-mux-replacing-terminal-pipeline",
                                                       kStreamId, /*stream_inflight_limit=*/4,
                                                       /*lifetime_guard_enabled=*/false);

  const std::vector<std::pair<std::int64_t, GstClockTime>> inputs{
      {101, 100 * GST_MSECOND},
      {102, 200 * GST_MSECOND},
      {103, 300 * GST_MSECOND},
      {104, 400 * GST_MSECOND},
  };
  for (const auto& [frame_id, pts] : inputs) {
    require(gst_app_src_push_buffer(GST_APP_SRC(fixture.appsrc),
                                    make_mux_input_buffer_with_timing(
                                        frame_id, pts, GST_CLOCK_TIME_NONE, 50 * GST_MSECOND)) ==
                GST_FLOW_OK,
            "failed to push replacing-chain timing input");
    GstSample* decoded = pull_mux_output(fixture.appsink, GST_SECOND);
    require(decoded != nullptr, "replacing-chain mux should admit the bounded input");
    // Models CVU/MLA consuming the decoded buffer and emitting a different
    // GstBuffer which does not carry arbitrary lifecycle GstMeta.
    gst_sample_unref(decoded);
  }

  require(gst_app_src_push_buffer(
              GST_APP_SRC(fixture.appsrc),
              make_mux_input_buffer_with_timing(105, 500 * GST_MSECOND, GST_CLOCK_TIME_NONE,
                                                50 * GST_MSECOND)) == GST_FLOW_OK,
          "failed to push replacing-chain pending input");
  GstSample* blocked = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  require(blocked == nullptr,
          "destroying replaced decoded inputs must not release terminal admission early");

  const auto release_and_require_pts = [&](GstBuffer* terminal, GstClockTime expected_pts,
                                           const char* reason) {
    GST_BUFFER_PTS(terminal) = 999 * GST_MSECOND;
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                terminal, fixture.mux_namespace),
            reason);
    require(GST_BUFFER_PTS(terminal) == expected_pts,
            "terminal selection restored timing from the wrong outstanding stream loan");
    gst_buffer_unref(terminal);
  };

  // Public frame/sequence fields are not freshness tokens on replacement
  // buffers. Even though they name frame 104, the ordered namespace contract
  // must complete frame 101.
  release_and_require_pts(make_terminal_sequence_buffer(kStreamId, 101, 104, 104),
                          100 * GST_MSECOND, "recycled public identity should use namespace FIFO");
  GstSample* fifth = pull_mux_output(fixture.appsink, GST_SECOND);
  require(fifth != nullptr, "one terminal completion should admit the pending fifth frame");
  gst_sample_unref(fifth);

  // Every replacement metadata shape uses the same enforced
  // ordered/non-dropping namespace completion order.
  release_and_require_pts(make_terminal_identity_buffer(kStreamId, std::nullopt), 200 * GST_MSECOND,
                          "stream-only terminal should use the bounded FIFO fallback");
  release_and_require_pts(make_terminal_sequence_buffer(kStreamId, std::nullopt, 103, 103),
                          300 * GST_MSECOND, "sequence-only identity should use namespace FIFO");
  release_and_require_pts(make_terminal_sequence_buffer(kStreamId, 101, 102, 102),
                          400 * GST_MSECOND,
                          "stale public sequence should not select a completed frame");
  release_and_require_pts(make_terminal_sequence_buffer(kStreamId, 9999, 105, 105),
                          500 * GST_MSECOND,
                          "pending frame should retain its registered FIFO timing");

  stop_latest_mux_pipeline(&fixture);
}

void test_replacing_chain_stale_live_private_collision_cannot_exhaust_total_gate() {
  constexpr std::size_t kStreamCount = 4;
  constexpr int kTotalLimit = 8;
  constexpr int kIterations = 2048;

  GstElement* pipeline = gst_pipeline_new("latest-mux-replacing-multistream-long-pipeline");
  GstElement* mux = gst_element_factory_make("neatlatestbystreammux", nullptr);
  GstElement* dropper = gst_element_factory_make("identity", nullptr);
  GstElement* queue = gst_element_factory_make("queue", nullptr);
  GstElement* appsink = gst_element_factory_make("appsink", nullptr);
  require(pipeline && mux && dropper && queue && appsink,
          "failed to construct multi-stream replacing mux fixture");

  const std::vector<std::string> stream_ids{"replacing-long-stream0", "replacing-long-stream1",
                                            "replacing-long-stream2", "replacing-long-stream3"};
  g_object_set(mux, "stream-ids",
               "replacing-long-stream0,replacing-long-stream1,replacing-long-stream2,"
               "replacing-long-stream3",
               "stream-inflight-limits", "8,8,8,8", "max-inflight-total", kTotalLimit, nullptr);
  require(simaai::neat::pipeline_internal::set_latest_by_stream_mux_lifetime_guard_enabled(
              mux, /*enabled=*/false),
          "failed to disable lifecycle guards for multi-stream replacing fixture");
  g_object_set(queue, "max-size-buffers", 1U, "max-size-bytes", 0U, "max-size-time",
               static_cast<guint64>(0), "leaky", 0, nullptr);
  g_object_set(appsink, "sync", FALSE, "max-buffers", 16U, "drop", FALSE, "enable-last-sample",
               FALSE, nullptr);
  gst_bin_add_many(GST_BIN(pipeline), mux, dropper, queue, appsink, nullptr);
  require(gst_element_link_many(mux, dropper, queue, appsink, nullptr) == TRUE,
          "failed to link multi-stream replacing output path");

  std::vector<GstElement*> appsrcs;
  std::vector<GstPad*> mux_pads;
  appsrcs.reserve(kStreamCount);
  mux_pads.reserve(kStreamCount);
  GstCaps* caps =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8", "width", G_TYPE_INT, 4,
                          "height", G_TYPE_INT, 4, "framerate", GST_TYPE_FRACTION, 1, 1, nullptr);
  require(caps != nullptr, "failed to allocate multi-stream replacing caps");
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    GstElement* appsrc = gst_element_factory_make("appsrc", nullptr);
    require(appsrc != nullptr, "failed to create multi-stream replacing appsrc");
    g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);
    gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
    gst_bin_add(GST_BIN(pipeline), appsrc);
    const std::string pad_name = "sink_" + std::to_string(i);
    GstPad* mux_pad = gst_element_request_pad_simple(mux, pad_name.c_str());
    GstPad* source_pad = gst_element_get_static_pad(appsrc, "src");
    require(mux_pad && source_pad && gst_pad_link(source_pad, mux_pad) == GST_PAD_LINK_OK,
            "failed to link multi-stream replacing appsrc");
    gst_object_unref(source_pad);
    appsrcs.push_back(appsrc);
    mux_pads.push_back(mux_pad);
  }
  gst_caps_unref(caps);

  const std::uint64_t mux_namespace = simaai::neat::latest_by_stream_mux_namespace(mux);
  require(mux_namespace != 0U, "multi-stream replacing mux should expose a namespace");
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start multi-stream replacing mux fixture");

  struct Outstanding {
    std::size_t stream_index = 0;
    std::int64_t sequence = -1;
    GstClockTime pts = GST_CLOCK_TIME_NONE;
  };
  std::deque<Outstanding> outstanding;
  std::vector<std::int64_t> next_sequence(kStreamCount, 1);

  const auto push_and_pull = [&](std::size_t stream_index, std::int64_t sequence) {
    const GstClockTime pts =
        static_cast<GstClockTime>(sequence * 100 + static_cast<std::int64_t>(stream_index)) *
        GST_MSECOND;
    require(gst_app_src_push_buffer(GST_APP_SRC(appsrcs[stream_index]),
                                    make_mux_input_buffer_with_timing(sequence, pts,
                                                                      GST_CLOCK_TIME_NONE,
                                                                      GST_MSECOND)) == GST_FLOW_OK,
            "failed to push multi-stream replacing input");
    GstSample* decoded = pull_mux_output(appsink, GST_SECOND);
    require(decoded != nullptr,
            "total-cap=8 replacing mux stalled after stale/missing terminal metadata");
    gst_sample_unref(decoded);
    outstanding.push_back(Outstanding{stream_index, sequence, pts});
  };

  // Fill the shared total gate with duplicate per-stream frame/input sequences.
  // Registration is serialized by pulling each mux output before the next push,
  // so this deque also records the exact namespace-wide completion order.
  for (int i = 0; i < kTotalLimit; ++i) {
    const std::size_t stream_index = static_cast<std::size_t>(i) % kStreamCount;
    const std::int64_t sequence = next_sequence[stream_index]++;
    push_and_pull(stream_index, sequence);
  }

  const auto require_canonical_terminal = [&](GstBuffer* terminal, const Outstanding& expected) {
    require(GST_BUFFER_PTS(terminal) == expected.pts,
            "terminal fallback restored timing from the wrong replacing loan");
    GstCustomMeta* meta = gst_buffer_get_custom_meta(terminal, "GstSimaMeta");
    GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    require(structure != nullptr,
            "terminal fallback should restore canonical GstSimaMeta when metadata was missing");
    const std::string& expected_stream = stream_ids[expected.stream_index];
    const char* stream_id = gst_structure_get_string(structure, "stream-id");
    const char* orig_stream_id = gst_structure_get_string(structure, "orig-stream-id");
    const char* private_stream_id = gst_structure_get_string(structure, kLoanStreamIdField);
    require(stream_id && std::string(stream_id) == expected_stream && orig_stream_id &&
                std::string(orig_stream_id) == expected_stream && private_stream_id &&
                std::string(private_stream_id) == expected_stream,
            "terminal completion should restore canonical public/private stream identity");
    gint64 frame_id = -1;
    gint64 input_seq = -1;
    gint64 orig_input_seq = -1;
    gint64 private_frame_id = -1;
    gint64 private_input_seq = -1;
    gint64 private_orig_input_seq = -1;
    guint64 private_namespace = 0;
    gboolean private_valid = TRUE;
    require(gst_structure_get_int64(structure, "frame-id", &frame_id) == TRUE &&
                gst_structure_get_int64(structure, "input-seq", &input_seq) == TRUE &&
                gst_structure_get_int64(structure, "orig-input-seq", &orig_input_seq) == TRUE &&
                gst_structure_get_int64(structure, kLoanFrameIdField, &private_frame_id) == TRUE &&
                gst_structure_get_int64(structure, "neat-latest-mux-loan-input-seq",
                                        &private_input_seq) == TRUE &&
                gst_structure_get_int64(structure, "neat-latest-mux-loan-orig-input-seq",
                                        &private_orig_input_seq) == TRUE &&
                gst_structure_get_uint64(structure, kLoanNamespaceField, &private_namespace) ==
                    TRUE &&
                gst_structure_get_boolean(structure, kLoanValidField, &private_valid) == TRUE,
            "terminal completion should restore complete canonical frame/sequence identity");
    require(frame_id == expected.sequence && input_seq == expected.sequence &&
                orig_input_seq == expected.sequence && private_frame_id == expected.sequence &&
                private_input_seq == expected.sequence &&
                private_orig_input_seq == expected.sequence && private_namespace == mux_namespace &&
                private_valid == FALSE,
            "terminal completion restored canonical identity from the wrong replacing loan");
  };

  {
    // Per-source sequences overlap by design. A recycled current stream-id can
    // therefore combine with the current sequence and accidentally name a
    // different live stream. The replacing-chain fallback must honor global
    // completion order rather than swapping the two channels' identity/PTS.
    const Outstanding expected = outstanding.front();
    const auto collision = std::find_if(std::next(outstanding.begin()), outstanding.end(),
                                        [&](const Outstanding& candidate) {
                                          return candidate.stream_index != expected.stream_index &&
                                                 candidate.sequence == expected.sequence;
                                        });
    require(collision != outstanding.end(),
            "multi-stream fixture should contain a live duplicate sequence collision");
    GstBuffer* terminal =
        make_terminal_stale_private_key_buffer(stream_ids[expected.stream_index].c_str(), 910000,
                                               mux_namespace, "stale-private-stream", 810000);
    GstStructure* structure =
        gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
    require(structure != nullptr, "failed to create valid-other-stream collision metadata");
    gst_structure_set(structure, "stream-id", G_TYPE_STRING,
                      stream_ids[collision->stream_index].c_str(), "input-seq", G_TYPE_INT64,
                      static_cast<gint64>(expected.sequence), "orig-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(expected.sequence), nullptr);
    GST_BUFFER_PTS(terminal) = 999 * GST_MSECOND;
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                terminal, mux_namespace),
            "valid stale stream/sequence collision should release the globally oldest loan");
    require_canonical_terminal(terminal, expected);
    gst_buffer_unref(terminal);
    outstanding.pop_front();
    push_and_pull(expected.stream_index, next_sequence[expected.stream_index]++);
  }

  for (int i = 0; i < kIterations; ++i) {
    std::size_t selected_index = 0;
    GstBuffer* terminal = nullptr;
    const int variant = i % 6;
    if (variant == 0) {
      // Even a current stream-id + sequence pair can be a coincidental mix of
      // recycled scalars. Point it at a non-oldest live loan and require FIFO.
      const auto& target = outstanding.back();
      terminal = make_terminal_stale_private_key_buffer(
          "stale-original-stream", 900000 + i, mux_namespace, "stale-private-stream", 800000 + i);
      GstStructure* structure =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
      require(structure != nullptr, "failed to create current-stream terminal metadata");
      gst_structure_set(structure, "stream-id", G_TYPE_STRING,
                        stream_ids[target.stream_index].c_str(), "input-seq", G_TYPE_INT64,
                        static_cast<gint64>(target.sequence), "orig-input-seq", G_TYPE_INT64,
                        static_cast<gint64>(target.sequence), nullptr);
    } else if (variant == 1) {
      // A pooled terminal buffer can retain a complete private key which
      // exactly names another live, non-oldest loan. The renderer's global
      // ordered/non-dropping contract is authoritative, so even this
      // syntactically perfect collision must not jump the namespace FIFO.
      const auto& target = outstanding.back();
      terminal = make_terminal_stale_private_key_buffer(
          "stale-public-stream", 900000 + i, mux_namespace, stream_ids[target.stream_index].c_str(),
          target.sequence);
      GstStructure* structure =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
      require(structure != nullptr, "failed to create live-private collision metadata");
      gst_structure_set(structure, "input-seq", G_TYPE_INT64, static_cast<gint64>(700000 + i),
                        "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(700000 + i),
                        "neat-latest-mux-loan-input-seq", G_TYPE_INT64,
                        static_cast<gint64>(target.sequence), "neat-latest-mux-loan-orig-input-seq",
                        G_TYPE_INT64, static_cast<gint64>(target.sequence), nullptr);
    } else if (variant == 2) {
      // InputStreamPull treats a current stream-id equal to buffer-name as a
      // stage label and exposes orig-stream-id, but neither public scalar is a
      // freshness token. Point both at a non-oldest loan and still require FIFO.
      const auto& target = outstanding.back();
      terminal = make_terminal_stale_private_key_buffer(stream_ids[target.stream_index].c_str(),
                                                        900000 + i, mux_namespace,
                                                        "stale-private-stream", 800000 + i);
      GstStructure* structure =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
      require(structure != nullptr, "failed to create stage-label terminal metadata");
      gst_structure_set(structure, "stream-id", G_TYPE_STRING, "boxdecode-output", "buffer-name",
                        G_TYPE_STRING, "boxdecode-output", "input-seq", G_TYPE_INT64,
                        static_cast<gint64>(target.sequence), "orig-input-seq", G_TYPE_INT64,
                        static_cast<gint64>(target.sequence), nullptr);
    } else if (variant == 3) {
      // The current stream scalar is stale while orig-stream-id looks correct.
      // Do not trust the older orig field: the ordered namespace fallback must
      // consume the globally oldest loan and rewrite both fields canonically.
      const auto& oldest = outstanding.front();
      terminal = make_terminal_stale_private_key_buffer(stream_ids[oldest.stream_index].c_str(),
                                                        900000 + i, mux_namespace,
                                                        "stale-private-stream", 800000 + i);
      GstStructure* structure =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
      require(structure != nullptr, "failed to create stale-current terminal metadata");
      gst_structure_set(structure, "stream-id", G_TYPE_STRING, "stale-current-stream", "input-seq",
                        G_TYPE_INT64, static_cast<gint64>(oldest.sequence), "orig-input-seq",
                        G_TYPE_INT64, static_cast<gint64>(oldest.sequence), nullptr);
    } else if (variant == 4) {
      // Both stream identities/private fields are stale, and this duplicate
      // per-stream sequence may name loans in other streams. It must not be
      // treated as namespace-global identity.
      const auto duplicate_sequence = outstanding.back().sequence;
      terminal = make_terminal_stale_private_key_buffer(
          "stale-public-stream", 900000 + i, mux_namespace, "stale-private-stream", 800000 + i);
      GstStructure* structure =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(terminal, "GstSimaMeta"));
      require(structure != nullptr, "failed to create fully stale terminal metadata");
      gst_structure_set(structure, "input-seq", G_TYPE_INT64,
                        static_cast<gint64>(duplicate_sequence), "orig-input-seq", G_TYPE_INT64,
                        static_cast<gint64>(duplicate_sequence), nullptr);
    } else {
      // No GstSimaMeta at all: one ordered terminal output still completes one
      // exact-namespace loan and must synthesize canonical metadata.
      terminal = gst_buffer_new_allocate(nullptr, 16U, nullptr);
      require(terminal != nullptr, "failed to allocate missing-metadata terminal buffer");
    }

    GST_BUFFER_PTS(terminal) = 999 * GST_MSECOND;
    const Outstanding expected = outstanding[selected_index];
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                terminal, mux_namespace),
            "stale/missing terminal metadata must release one replacing loan");
    require_canonical_terminal(terminal, expected);
    gst_buffer_unref(terminal);
    outstanding.erase(outstanding.begin() + static_cast<std::ptrdiff_t>(selected_index));

    const std::size_t stream_index = static_cast<std::size_t>(i) % kStreamCount;
    push_and_pull(stream_index, next_sequence[stream_index]++);
  }

  while (!outstanding.empty()) {
    GstBuffer* terminal = gst_buffer_new_allocate(nullptr, 16U, nullptr);
    require(terminal != nullptr, "failed to allocate final missing-metadata terminal buffer");
    const Outstanding expected = outstanding.front();
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                terminal, mux_namespace),
            "final ordered completion should release its replacing loan");
    require_canonical_terminal(terminal, expected);
    gst_buffer_unref(terminal);
    outstanding.pop_front();
  }

  for (GstElement* appsrc : appsrcs) {
    (void)gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
  }
  require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "failed to stop multi-stream replacing mux fixture");
  for (GstPad* pad : mux_pads) {
    gst_element_release_request_pad(mux, pad);
    gst_object_unref(pad);
  }
  gst_object_unref(pipeline);
}

void test_keyed_release_cannot_restore_finalized_timing() {
  constexpr const char* kStreamId = "keyed-timing-stream";
  constexpr GstClockTime kOriginalPts = 123 * GST_MSECOND;
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-keyed-timing-pipeline", kStreamId);

  require(gst_app_src_push_buffer(GST_APP_SRC(fixture.appsrc),
                                  make_mux_input_buffer_with_timing(31, kOriginalPts,
                                                                    GST_CLOCK_TIME_NONE,
                                                                    GST_MSECOND)) == GST_FLOW_OK,
          "failed to push keyed timing fixture");
  GstSample* output = pull_mux_output(fixture.appsink, GST_SECOND);
  require(output != nullptr, "keyed timing fixture should reach the mux output");

  simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan(kStreamId, 31);

  GstBuffer* stale_probe = make_terminal_identity_buffer(kStreamId, 31);
  constexpr GstClockTime kSentinelPts = 999 * GST_MSECOND;
  GST_BUFFER_PTS(stale_probe) = kSentinelPts;
  require(!simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              stale_probe, fixture.mux_namespace),
          "a finalized keyed loan must no longer be registered");
  require(GST_BUFFER_PTS(stale_probe) == kSentinelPts,
          "keyed final release must not restore finalized timing on a later terminal buffer");
  gst_buffer_unref(stale_probe);
  gst_sample_unref(output);
  stop_latest_mux_pipeline(&fixture);
}

void test_public_per_stream_limit_reaches_mux_gate() {
  constexpr const char* kStreamId = "limit-two-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-explicit-limit-pipeline", kStreamId, 2);

  push_mux_input(fixture.appsrc, 1);
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "explicit limit=2 should admit the first frame");
  push_mux_input(fixture.appsrc, 2);
  GstSample* second = pull_mux_output(fixture.appsink, GST_SECOND);
  require(second != nullptr, "explicit limit=2 should admit a second outstanding frame");

  push_mux_input(fixture.appsrc, 3);
  GstSample* blocked = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  require(blocked == nullptr, "explicit limit=2 must hold a third frame until terminal completion");
  release_terminal_loan(fixture, kStreamId, 1);
  GstSample* third = pull_mux_output(fixture.appsink, GST_SECOND);
  require(third != nullptr, "terminal completion must reopen the explicit per-stream gate");

  release_terminal_loan(fixture, kStreamId, 2);
  release_terminal_loan(fixture, kStreamId, 3);
  gst_sample_unref(third);
  gst_sample_unref(second);
  gst_sample_unref(first);
  stop_latest_mux_pipeline(&fixture);
}

void test_output_buffer_finalize_does_not_release_terminal_credit() {
  constexpr const char* kStreamId = "lifecycle-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-terminal-lifecycle-pipeline", kStreamId);

  push_mux_input(fixture.appsrc, 1);
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "latest-mux should emit the first lifecycle frame");

  // A normal model transform copies the lifecycle meta to its output before
  // releasing the decoded input. The propagated carrier keeps terminal credit
  // charged even though the original decoder-backed buffer is recyclable.
  GstBuffer* terminal = gst_buffer_copy_deep(gst_sample_get_buffer(first));
  require(terminal != nullptr, "failed to copy the lifecycle carrier to a terminal buffer");
  gst_sample_unref(first);
  push_mux_input(fixture.appsrc, 2);
  GstSample* early_second = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  const bool emitted_before_terminal = early_second != nullptr;
  if (early_second) {
    gst_sample_unref(early_second);
  }
  require(!emitted_before_terminal,
          "finalizing the decoded output buffer must not release terminal-bound mux credit");

  require(!simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              terminal, fixture.mux_namespace + 1U),
          "a wrong namespace hint must reject the lifecycle carrier without claiming it");
  GstCustomMeta* terminal_meta = gst_buffer_get_custom_meta(terminal, "GstSimaMeta");
  GstStructure* terminal_structure =
      terminal_meta ? gst_custom_meta_get_structure(terminal_meta) : nullptr;
  require(terminal_structure != nullptr, "propagated terminal carrier should retain GstSimaMeta");
  gst_structure_set(terminal_structure, "stream-id", G_TYPE_STRING, "rewritten-stream",
                    "orig-stream-id", G_TYPE_STRING, "rewritten-stream", nullptr);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              terminal, fixture.mux_namespace),
          "the immutable guard must survive rejected scope and rewritten public metadata");
  gst_buffer_unref(terminal);
  GstSample* second = pull_mux_output(fixture.appsink, GST_SECOND);
  require(second != nullptr, "terminal completion should admit the pending lifecycle frame");
  release_terminal_loan(fixture, kStreamId, 2);
  gst_sample_unref(second);
  stop_latest_mux_pipeline(&fixture);
}

void test_drop_before_terminal_releases_lifetime_credit() {
  constexpr const char* kStreamId = "drop-before-terminal-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-drop-before-terminal-pipeline", kStreamId);
  g_object_set(fixture.dropper, "drop-probability", 1.0, nullptr);

  // A limit-one mux used to stop after the first buffer when a downstream
  // element discarded it before the terminal appsink probe. Each destroyed
  // lifecycle carrier must now return exactly that frame's credit.
  for (std::int64_t frame_id = 1; frame_id <= 8; ++frame_id) {
    push_mux_input(fixture.appsrc, frame_id);
    GstSample* dropped = pull_mux_output(fixture.appsink, 50 * GST_MSECOND);
    require(dropped == nullptr, "drop fixture must discard every pre-terminal frame");
  }

  g_object_set(fixture.dropper, "drop-probability", 0.0, nullptr);
  push_mux_input(fixture.appsrc, 9);
  GstSample* terminal = pull_mux_output(fixture.appsink, GST_SECOND);
  require(terminal != nullptr,
          "pre-terminal drops beyond the inflight limit must not stall mux admission");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(terminal), fixture.mux_namespace),
          "terminal frame should release through the normal terminal probe path");
  gst_sample_unref(terminal);

  // Finalizing a terminal buffer after its probe must not release twice and
  // corrupt the limit-one gate; the next frame still has exactly one credit.
  push_mux_input(fixture.appsrc, 10);
  GstSample* next = pull_mux_output(fixture.appsink, GST_SECOND);
  require(next != nullptr, "terminal release plus buffer finalization must not double-release");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(next), fixture.mux_namespace),
          "next terminal frame should retain a valid lifecycle guard");
  gst_sample_unref(next);
  stop_latest_mux_pipeline(&fixture);
}

void test_fanout_terminal_and_drop_release_retained_credit() {
  constexpr const char* kStreamId = "fanout-drop-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-fanout-drop-pipeline", kStreamId);

  const auto retain_one_fanout_ref = [&](std::int64_t frame_id) {
    const std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit> credits{
        {fixture.mux_namespace, kStreamId, frame_id, frame_id, frame_id}};
    require(simaai::neat::pipeline_internal::retain_realtime_frame_credits(
                credits, 1U, "unit-latest-mux-fanout"),
            "fan-out fixture must retain one logical mux-loan reference");
  };

  // One branch completes at the terminal while its copied sibling is dropped.
  // The first completion must not disarm the shared guard before the last
  // carrier accounts for the remaining retained reference.
  push_mux_input(fixture.appsrc, 1);
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "fan-out fixture should emit the first frame");
  GstBuffer* dropped_sibling = gst_buffer_copy_deep(gst_sample_get_buffer(first));
  require(dropped_sibling != nullptr, "failed to copy a dropped fan-out carrier");
  retain_one_fanout_ref(1);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(first), fixture.mux_namespace),
          "one fan-out branch should complete at the terminal");
  require(!simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(first), fixture.mux_namespace),
          "one physical terminal carrier must not consume the retained sibling reference twice");
  gst_sample_unref(first);
  gst_buffer_unref(dropped_sibling);

  push_mux_input(fixture.appsrc, 2);
  GstSample* after_mixed_outcome = pull_mux_output(fixture.appsink, GST_SECOND);
  require(after_mixed_outcome != nullptr,
          "one terminal plus one dropped carrier must return limit-one credit");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(after_mixed_outcome), fixture.mux_namespace),
          "post-drop progress frame should release normally");
  gst_sample_unref(after_mixed_outcome);

  // When every retained branch drops before the terminal, destruction of the
  // last physical meta carrier must finalize all logical refs, not just
  // decrement one of them. Model that at the dropper sink so no appsink/preroll
  // reference can make the lifetime assertion implementation-dependent.
  struct AllDropProbeContext {
    std::uint64_t mux_namespace = 0;
    const char* stream_id = nullptr;
    std::int64_t frame_id = -1;
    std::atomic<bool> ran{false};
    std::atomic<bool> copied{false};
    std::atomic<bool> retained{false};
  } drop_context{fixture.mux_namespace, kStreamId, 3};
  GstPad* dropper_sink = gst_element_get_static_pad(fixture.dropper, "sink");
  require(dropper_sink != nullptr, "failed to get fan-out dropper sink pad");
  const gulong drop_probe = gst_pad_add_probe(
      dropper_sink, GST_PAD_PROBE_TYPE_BUFFER,
      +[](GstPad*, GstPadProbeInfo* info, gpointer user_data) -> GstPadProbeReturn {
        auto* context = static_cast<AllDropProbeContext*>(user_data);
        GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
        GstBuffer* sibling = buffer ? gst_buffer_copy_deep(buffer) : nullptr;
        context->copied.store(sibling != nullptr, std::memory_order_release);
        const std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit> credits{
            {context->mux_namespace, context->stream_id, context->frame_id, context->frame_id,
             context->frame_id}};
        context->retained.store(simaai::neat::pipeline_internal::retain_realtime_frame_credits(
                                    credits, 1U, "unit-latest-mux-all-drop"),
                                std::memory_order_release);
        if (sibling) {
          gst_buffer_unref(sibling);
        }
        context->ran.store(true, std::memory_order_release);
        return GST_PAD_PROBE_REMOVE;
      },
      &drop_context, nullptr);
  require(drop_probe != 0, "failed to attach all-drop fan-out probe");
  g_object_set(fixture.dropper, "drop-probability", 1.0, nullptr);
  push_mux_input(fixture.appsrc, 3);
  GstSample* all_drop = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  require(all_drop == nullptr, "all-drop fan-out frame must not reach appsink");
  require(drop_context.ran.load(std::memory_order_acquire) &&
              drop_context.copied.load(std::memory_order_acquire) &&
              drop_context.retained.load(std::memory_order_acquire),
          "all-drop probe must copy a carrier and retain its logical fan-out reference");
  gst_object_unref(dropper_sink);

  g_object_set(fixture.dropper, "drop-probability", 0.0, nullptr);
  push_mux_input(fixture.appsrc, 4);
  GstSample* after_all_drop = pull_mux_output(fixture.appsink, GST_SECOND);
  require(after_all_drop != nullptr,
          "last-carrier destruction must release every retained logical reference");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(after_all_drop), fixture.mux_namespace),
          "all-drop progress frame should release normally");
  gst_sample_unref(after_all_drop);

  // Exercise both orderings of the non-final terminal/drop race. Carrier
  // retirement is serialized by the shared guard, so whichever branch retires
  // last must force the remaining retained ref and wake the limit-one mux.
  for (std::int64_t iteration = 0; iteration < 8; ++iteration) {
    const std::int64_t raced_frame = 100 + iteration * 2;
    push_mux_input(fixture.appsrc, raced_frame);
    GstSample* raced_terminal = pull_mux_output(fixture.appsink, GST_SECOND);
    require(raced_terminal != nullptr, "race fixture should emit its retained frame");
    GstBuffer* raced_drop = gst_buffer_copy_deep(gst_sample_get_buffer(raced_terminal));
    require(raced_drop != nullptr, "race fixture should copy a dropped sibling carrier");
    retain_one_fanout_ref(raced_frame);

    std::atomic<bool> start{false};
    std::atomic<bool> terminal_released{false};
    std::thread terminal_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      terminal_released.store(
          simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(raced_terminal), fixture.mux_namespace),
          std::memory_order_release);
    });
    std::thread drop_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      gst_buffer_unref(raced_drop);
    });
    start.store(true, std::memory_order_release);
    terminal_thread.join();
    drop_thread.join();
    require(terminal_released.load(std::memory_order_acquire),
            "raced terminal branch should consume one retained reference");
    gst_sample_unref(raced_terminal);

    const std::int64_t progress_frame = raced_frame + 1;
    push_mux_input(fixture.appsrc, progress_frame);
    GstSample* raced_progress = pull_mux_output(fixture.appsink, GST_SECOND);
    require(raced_progress != nullptr,
            "concurrent terminal/drop retirement must return limit-one credit");
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                gst_sample_get_buffer(raced_progress), fixture.mux_namespace),
            "race progress frame should release normally");
    gst_sample_unref(raced_progress);
  }
  stop_latest_mux_pipeline(&fixture);
}

void test_stale_guard_sequence_cannot_release_reused_key() {
  constexpr const char* kStreamId = "stale-guard-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-stale-guard-pipeline", kStreamId);

  push_mux_input(fixture.appsrc, 42);
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "stale-guard fixture should emit its first frame");
  GstBuffer* stale_carrier = gst_buffer_copy_deep(gst_sample_get_buffer(first));
  require(stale_carrier != nullptr, "failed to retain stale lifecycle carrier");
  // Finalize the first registration without touching either lifecycle carrier.
  // This deliberately leaves the old guard armed so the stale terminal below
  // reaches the registry sequence check rather than failing only because the
  // old guard was already disarmed.
  simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan(kStreamId, 42);
  gst_sample_unref(first);

  // Reuse the same public/private key under a new registry sequence while an
  // old copied carrier still exists.
  push_mux_input(fixture.appsrc, 42);
  GstSample* replacement = pull_mux_output(fixture.appsink, GST_SECOND);
  require(replacement != nullptr, "same key should register again after first completion");
  require(!simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              stale_carrier, fixture.mux_namespace),
          "stale guard sequence must not release a newer loan with the same scalar key");
  gst_buffer_unref(stale_carrier);

  push_mux_input(fixture.appsrc, 43);
  GstSample* blocked = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  require(blocked == nullptr, "new same-key loan must remain charged after stale guard rejection");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              gst_sample_get_buffer(replacement), fixture.mux_namespace),
          "current same-key guard should release its own registration");
  gst_sample_unref(replacement);

  GstSample* progress = pull_mux_output(fixture.appsink, GST_SECOND);
  require(progress != nullptr, "current guard completion should admit the pending frame");
  release_terminal_loan(fixture, kStreamId, 43);
  gst_sample_unref(progress);
  stop_latest_mux_pipeline(&fixture);
}

void test_teardown_releases_unresolved_terminal_credit() {
  constexpr const char* kStreamId = "teardown-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-unresolved-teardown-pipeline", kStreamId);

  push_mux_input(fixture.appsrc, 7);
  GstSample* output = pull_mux_output(fixture.appsink, GST_SECOND);
  require(output != nullptr, "latest-mux should emit the unresolved teardown frame");

  const std::uint64_t mux_namespace = fixture.mux_namespace;
  stop_latest_mux_pipeline(&fixture);
  gst_sample_unref(output);

  GstBuffer* late_terminal = make_terminal_identity_buffer(kStreamId, 7);
  const bool remained_registered =
      simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(late_terminal,
                                                                                    mux_namespace);
  gst_buffer_unref(late_terminal);
  require(!remained_registered, "latest-mux teardown must release every unresolved terminal loan");
}

struct ReentrantBufferFinalizeState {
  GstElement* mux = nullptr;
  std::atomic<bool>* callback_called = nullptr;
};

void query_mux_property_on_buffer_finalize(gpointer data) {
  std::unique_ptr<ReentrantBufferFinalizeState> state(
      static_cast<ReentrantBufferFinalizeState*>(data));
  gchar* stream_ids = nullptr;
  g_object_get(state->mux, "stream-ids", &stream_ids, nullptr);
  g_free(stream_ids);
  state->callback_called->store(true, std::memory_order_release);
}

void test_pending_buffer_unref_is_reentrant() {
  constexpr const char* kStreamId = "reentrant-stream";
  LatestMuxPipeline fixture =
      make_latest_mux_pipeline("latest-mux-reentrant-unref-pipeline", kStreamId);

  push_mux_input(fixture.appsrc, 11);
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "latest-mux should emit the first reentrant-unref frame");

  // Use a fresh single-owner buffer so this lock-safety regression does not
  // depend on appsink's implementation-specific retention of output refs.
  GstBuffer* pending = make_mux_input_buffer(12);

  std::atomic<bool> callback_called{false};
  auto* callback_state = new ReentrantBufferFinalizeState{fixture.mux, &callback_called};
  gst_mini_object_set_qdata(
      GST_MINI_OBJECT_CAST(pending),
      g_quark_from_static_string("sima-unit-latest-mux-reentrant-buffer-finalize"), callback_state,
      query_mux_property_on_buffer_finalize);

  // Credit for frame 11 is still in flight, so this buffer remains in the mux's
  // pending slot. Replacing it drops its final ref. The qdata callback
  // re-enters the mux property getter and deterministically deadlocks if the
  // unref is performed while the mux mutex is held.
  require(gst_pad_chain(fixture.mux_sink, pending) == GST_FLOW_OK,
          "failed to chain synthetic buffer to latest-mux pending slot");
  require(!callback_called.load(std::memory_order_acquire),
          "pending buffer should remain owned by latest-mux until replacement");

  GstBuffer* replacement = make_mux_input_buffer(13);
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  bool replacement_done = false;
  GstFlowReturn replacement_flow = GST_FLOW_ERROR;
  std::thread replacement_thread([&] {
    replacement_flow = gst_pad_chain(fixture.mux_sink, replacement);
    {
      std::lock_guard<std::mutex> lock(completion_mutex);
      replacement_done = true;
    }
    completion_cv.notify_one();
  });
  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    if (!completion_cv.wait_for(lock, std::chrono::seconds(2), [&] { return replacement_done; })) {
      std::cerr << "[FAIL] replacing a pending buffer deadlocked during buffer finalization\n"
                << std::flush;
      std::_Exit(EXIT_FAILURE);
    }
  }
  replacement_thread.join();
  require(replacement_flow == GST_FLOW_OK, "latest-mux pending replacement should succeed");
  require(callback_called.load(std::memory_order_acquire),
          "pending replacement should finalize the displaced synthetic buffer");

  GstSample* early_replacement = pull_mux_output(fixture.appsink, 100 * GST_MSECOND);
  const bool emitted_before_terminal = early_replacement != nullptr;
  if (early_replacement) {
    gst_sample_unref(early_replacement);
  }
  require(!emitted_before_terminal,
          "pending-buffer finalization must not release terminal-bound mux credit");

  release_terminal_loan(fixture, kStreamId, 11);
  gst_sample_unref(first);
  GstSample* second = pull_mux_output(fixture.appsink, GST_SECOND);
  require(second != nullptr, "terminal completion should admit the replacement frame");
  release_terminal_loan(fixture, kStreamId, 13);
  gst_sample_unref(second);
  stop_latest_mux_pipeline(&fixture);
}

void test_namespace_bounded_terminal_loan_fallback() {
  GstElement* pipeline = gst_pipeline_new("latest-mux-credit-fallback-pipeline");
  GstElement* appsrc = gst_element_factory_make("appsrc", "latest-mux-credit-source");
  GstElement* mux = gst_element_factory_make("neatlatestbystreammux", "latest-mux-credit-mux");
  GstElement* appsink = gst_element_factory_make("appsink", "latest-mux-credit-sink");
  require(pipeline && appsrc && mux && appsink,
          "failed to construct latest-mux credit fallback pipeline");

  g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);
  g_object_set(mux, "stream-ids", "sole-stream", "stream-inflight-limits", "1", nullptr);
  g_object_set(appsink, "sync", FALSE, "max-buffers", 4U, "drop", FALSE, "enable-last-sample",
               FALSE, nullptr);
  GstCaps* caps =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8", "width", G_TYPE_INT, 4,
                          "height", G_TYPE_INT, 4, "framerate", GST_TYPE_FRACTION, 1, 1, nullptr);
  require(caps != nullptr, "failed to allocate latest-mux input caps");
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  gst_caps_unref(caps);

  gst_bin_add_many(GST_BIN(pipeline), appsrc, mux, appsink, nullptr);
  GstPad* mux_sink = gst_element_request_pad_simple(mux, "sink_0");
  GstPad* appsrc_src = gst_element_get_static_pad(appsrc, "src");
  require(mux_sink && appsrc_src && gst_pad_link(appsrc_src, mux_sink) == GST_PAD_LINK_OK,
          "failed to link appsrc to latest-mux request pad");
  gst_object_unref(appsrc_src);
  require(gst_element_link(mux, appsink) == TRUE, "failed to link latest-mux to appsink");

  const std::uint64_t mux_namespace = simaai::neat::latest_by_stream_mux_namespace(mux);
  require(mux_namespace != 0U, "latest-mux should expose a nonzero loan namespace");
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start latest-mux credit fallback pipeline");

  const auto require_canonical_terminal = [&](GstBuffer* terminal, std::int64_t frame_id) {
    require_canonical_terminal_loan(terminal, "sole-stream", frame_id, mux_namespace,
                                    static_cast<GstClockTime>(frame_id) * GST_MSECOND,
                                    GST_CLOCK_TIME_NONE, GST_MSECOND);
  };

  push_mux_input(appsrc, 101);
  GstSample* first = pull_mux_output(appsink, GST_SECOND);
  require(first != nullptr, "latest-mux should emit the first admitted frame");

  // Simulate a terminal model result that preserved stream identity but
  // replaced frame/input identity and no longer carries the private mux key.
  // Without the exact mux namespace this must not guess across mux instances.
  GstBuffer* mismatched = make_terminal_identity_buffer("sole-stream", 9999);
  require(
      !simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(mismatched),
      "unqualified mismatched output must not release a mux loan");

  push_mux_input(appsrc, 102);
  GstSample* blocked = pull_mux_output(appsink, 100 * GST_MSECOND);
  require(blocked == nullptr,
          "max-inflight=1 should hold the next stream frame before terminal release");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              mismatched, mux_namespace),
          "namespace-qualified terminal output should release the sole stream loan even when "
          "frame identity changed");
  require_canonical_terminal(mismatched, 101);
  gst_buffer_unref(mismatched);

  GstSample* second = pull_mux_output(appsink, GST_SECOND);
  require(second != nullptr,
          "sole-loan fallback should wake the mux and admit the pending stream frame");

  // Identity may be absent altogether after a transform. The stream plus the
  // exact mux namespace is still sufficient when only one loan can be live.
  GstBuffer* stream_only = make_terminal_identity_buffer("sole-stream", std::nullopt);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              stream_only, mux_namespace),
          "namespace-qualified stream-only output should release the sole stream loan");
  require_canonical_terminal(stream_only, 102);
  gst_buffer_unref(stream_only);

  push_mux_input(appsrc, 103);
  GstSample* third = pull_mux_output(appsink, GST_SECOND);
  require(third != nullptr,
          "stream-only sole-loan fallback should keep the stream making progress");

  // A recycled transform output can retain a syntactically valid private key
  // for a completed frame. Exact-key lookup then misses; the terminal mux
  // namespace plus the stream must still release the sole current loan.
  GstBuffer* stale_private = make_terminal_stale_private_key_buffer(
      "sole-stream", 103, mux_namespace, "stale-private-stream", 99999);
  require(
      !simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(stale_private),
      "unqualified stale private key must not guess a namespace-bounded stream loan");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              stale_private, mux_namespace),
          "namespace-qualified stale private key should use the current public stream and release "
          "its sole loan");
  require_canonical_terminal(stale_private, 103);
  gst_buffer_unref(stale_private);

  push_mux_input(appsrc, 104);
  GstSample* fourth = pull_mux_output(appsink, GST_SECOND);
  require(fourth != nullptr,
          "stale-private-key fallback should wake the mux and admit the next stream frame");
  // Sequence matching is stronger than a stale frame-id and must rewrite the
  // recycled public identity to the selected registry entry as well.
  GstBuffer* final_terminal = make_terminal_sequence_buffer("sole-stream", 9999, 104, 104);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              final_terminal, mux_namespace),
          "sequence terminal output should release the final test loan");
  require_canonical_terminal(final_terminal, 104);
  gst_buffer_unref(final_terminal);

  // Keep earlier outputs alive until after their fallback releases. This proves
  // progress came from terminal completion rather than GstBuffer qdata cleanup.
  gst_sample_unref(fourth);
  gst_sample_unref(third);
  gst_sample_unref(second);
  gst_sample_unref(first);
  (void)gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
  require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "failed to stop latest-mux credit fallback pipeline");
  gst_element_release_request_pad(mux, mux_sink);
  gst_object_unref(mux_sink);
  gst_object_unref(pipeline);
}

void test_sequence_fallback_selects_between_live_loans() {
  constexpr const char* kStreamId = "sequence-fallback-stream";
  constexpr GstClockTime kFirstPts = 1201 * GST_MSECOND;
  constexpr GstClockTime kFirstDts = 1191 * GST_MSECOND;
  constexpr GstClockTime kSecondPts = 2202 * GST_MSECOND;
  constexpr GstClockTime kSecondDts = 2192 * GST_MSECOND;
  constexpr GstClockTime kDuration = 7 * GST_MSECOND;

  LatestMuxPipeline fixture = make_latest_mux_pipeline("latest-mux-sequence-fallback-pipeline",
                                                       kStreamId, /*stream_inflight_limit=*/2);
  require(gst_app_src_push_buffer(GST_APP_SRC(fixture.appsrc),
                                  make_mux_input_buffer_with_timing(201, kFirstPts, kFirstDts,
                                                                    kDuration)) == GST_FLOW_OK,
          "failed to push first sequence-fallback input");
  GstSample* first = pull_mux_output(fixture.appsink, GST_SECOND);
  require(first != nullptr, "sequence-fallback mux should emit the first live loan");

  require(gst_app_src_push_buffer(GST_APP_SRC(fixture.appsrc),
                                  make_mux_input_buffer_with_timing(202, kSecondPts, kSecondDts,
                                                                    kDuration)) == GST_FLOW_OK,
          "failed to push second sequence-fallback input");
  GstSample* second = pull_mux_output(fixture.appsink, GST_SECOND);
  require(second != nullptr, "inflight-limit=2 should keep two same-stream loans live");

  // With two candidates, the namespace-bounded sole-loan fallback is
  // ineligible. The preserved input sequence must select the second loan even
  // though a recycled public frame-id and timing name neither live frame.
  GstBuffer* second_terminal = make_terminal_sequence_buffer(kStreamId, 9999, 202, 202);
  GST_BUFFER_PTS(second_terminal) = 999 * GST_MSECOND;
  GST_BUFFER_DTS(second_terminal) = 998 * GST_MSECOND;
  GST_BUFFER_DURATION(second_terminal) = 99 * GST_MSECOND;
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              second_terminal, fixture.mux_namespace),
          "input-seq should select the second of two live stream loans");
  require_canonical_terminal_loan(second_terminal, kStreamId, 202, fixture.mux_namespace,
                                  kSecondPts, kSecondDts, kDuration);
  gst_buffer_unref(second_terminal);

  // The first loan must still be independently live and selectable after the
  // out-of-order sequence completion above.
  GstBuffer* first_terminal = make_terminal_sequence_buffer(kStreamId, 8888, 201, 201);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              first_terminal, fixture.mux_namespace),
          "sequence fallback should leave the non-selected first loan live");
  require_canonical_terminal_loan(first_terminal, kStreamId, 201, fixture.mux_namespace, kFirstPts,
                                  kFirstDts, kDuration);
  gst_buffer_unref(first_terminal);

  // Keeping both source samples alive until both terminal completions proves
  // that buffer-finalize guards did not provide the observed credit release.
  gst_sample_unref(second);
  gst_sample_unref(first);
  stop_latest_mux_pipeline(&fixture);
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    test_encoded_frame_tap_owns_au_and_timing();
    test_clear_encoded_frame_tap_waits_for_inflight_callback();
    test_encoded_frame_tap_copy_failure_posts_pipeline_error();
    test_terminal_release_restores_original_pts();
    test_replacing_chain_uses_namespace_fifo_timing();
    test_replacing_chain_stale_live_private_collision_cannot_exhaust_total_gate();
    test_keyed_release_cannot_restore_finalized_timing();
    test_public_per_stream_limit_reaches_mux_gate();
    test_output_buffer_finalize_does_not_release_terminal_credit();
    test_drop_before_terminal_releases_lifetime_credit();
    test_fanout_terminal_and_drop_release_retained_credit();
    test_stale_guard_sequence_cannot_release_reused_key();
    test_teardown_releases_unresolved_terminal_credit();
    test_pending_buffer_unref_is_reentrant();
    test_namespace_bounded_terminal_loan_fallback();
    test_sequence_fallback_selects_between_live_loans();

    const simaai::neat::Sample namespaced =
        make_gst_sample_backed_sample(MuxLoanKey{1234, "mux-stream", 42});
    std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit> credits =
        simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(namespaced);
    require(credits.size() == 1U,
            "stamped latest-mux samples must not add unsafe public fallback credits");
    require(credits[0].namespace_id == 1234U, "credit should preserve mux namespace");
    require(credits[0].stream_id == "mux-stream", "credit should use stamped mux stream id");
    require(credits[0].frame_id == 42, "credit should use stamped mux frame id");

    simaai::neat::Sample bundled;
    bundled.kind = simaai::neat::SampleKind::Bundle;
    bundled.stream_id = "fallback-stream";
    bundled.frame_id = 7;
    bundled.fields.push_back(namespaced);
    credits = simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(bundled);
    require(credits.size() == 1U,
            "bundle with stamped latest-mux child must not add public fallback credits");
    require(credits[0].namespace_id == 1234U, "bundle credit should preserve child mux namespace");
    require(credits[0].stream_id == "mux-stream", "bundle credit should use child mux stream id");
    require(credits[0].frame_id == 42, "bundle credit should use child mux frame id");

    const simaai::neat::Sample fallback = make_gst_sample_backed_sample(std::nullopt);
    credits = simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(fallback);
    require(credits.size() == 1U, "expected one fallback realtime credit");
    require(credits[0].namespace_id == 0U, "fallback credit should remain unqualified");
    require(credits[0].stream_id == "fallback-stream", "fallback should use sample stream id");
    require(credits[0].frame_id == 7, "fallback should use sample frame id");

    simaai::neat::Sample sidecar_credit_sample = make_gst_sample_backed_sample(std::nullopt);
    const simaai::neat::pipeline_internal::RealtimeFrameCredit attached_credit{
        5678, "attached-stream", 99};
    require(!simaai::neat::pipeline_internal::sample_has_attached_realtime_frame_credit(
                sidecar_credit_sample),
            "fresh sample should not carry an attached graph realtime credit");
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(sidecar_credit_sample,
                                                                            attached_credit);
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(sidecar_credit_sample,
                                                                            attached_credit);
    require(simaai::neat::pipeline_internal::sample_has_attached_realtime_frame_credit(
                sidecar_credit_sample),
            "attached graph realtime credit should be discoverable from TensorBuffer sidecar");
    credits =
        simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sidecar_credit_sample);
    require(credits.size() == 2U,
            "attached graph credit should be collected once plus the public fallback credit");
    require(credits[0].namespace_id == 5678U,
            "attached graph credit should preserve private namespace");
    require(credits[0].stream_id == "attached-stream",
            "attached graph credit should preserve stream id");
    require(credits[0].frame_id == 99, "attached graph credit should preserve frame id");
    require(credits[1].namespace_id == 0U,
            "attached-credit sample should still expose public fallback release key");

    simaai::neat::Sample stamped_sidecar_sample =
        make_gst_sample_backed_sample(MuxLoanKey{1235, "mux-stream-b", 43});
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(stamped_sidecar_sample,
                                                                            attached_credit);
    credits =
        simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(stamped_sidecar_sample);
    require(credits.size() == 2U,
            "stamped latest-mux samples should keep graph-private sidecar credit only plus the "
            "exact mux loan key");
    require(credits[0].graph_private, "sidecar graph credit should remain graph-private");
    require(credits[0].namespace_id == 5678U,
            "stamped sidecar graph credit should preserve private namespace");
    require(credits[1].namespace_id == 1235U,
            "stamped sidecar mux credit should preserve latest-mux namespace");
    require(!credits[1].graph_private, "stamped latest-mux credit should not be graph-private");

    const simaai::neat::pipeline_internal::RealtimeFrameCredit colliding_graph_credit{
        4321, "domain-collision-stream", 44};
    simaai::neat::Sample colliding_domains_sample =
        make_gst_sample_backed_sample(MuxLoanKey{4321, "domain-collision-stream", 44});
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(
        colliding_domains_sample, colliding_graph_credit);
    credits = simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(
        colliding_domains_sample);
    require(credits.size() == 2U,
            "graph-private sidecar credits and latest-mux loans are separate domains even when "
            "their numeric keys collide");
    require(credits[0].graph_private && !credits[1].graph_private,
            "domain-collision sample should preserve graph-private and mux credits separately");

    auto sidecar_lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    auto unrelated_lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    require(sidecar_lane->gate->try_acquire() && unrelated_lane->gate->try_acquire(),
            "sidecar/stamped release test should acquire both lanes");
    const std::uint64_t sidecar_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    const std::uint64_t unrelated_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                sidecar_ns, "mux-same-stream", 77, sidecar_lane),
            "sidecar/stamped release test should register exact sidecar credit");
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                unrelated_ns, "mux-same-stream", 77, unrelated_lane),
            "sidecar/stamped release test should register unrelated same public key credit");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{sidecar_ns, "mux-same-stream", 77, -1,
                                                              -1, true},
         simaai::neat::pipeline_internal::RealtimeFrameCredit{1236, "mux-same-stream", 77}},
        "unit-stamped-with-sidecar");
    require(sidecar_lane->gate->inflight() == 0,
            "exact graph-private sidecar credit should release normally");
    require(unrelated_lane->gate->inflight() == 1,
            "stamped latest-mux release must not also release an unrelated unqualified graph "
            "credit");
    simaai::neat::pipeline_internal::release_all_registered_realtime_frame_credits(
        unrelated_ns, "unit-stamped-with-sidecar-cleanup");

    {
      simaai::neat::RealtimeGraphLinkOptions bounded_options;
      bounded_options.queue_depth = 1;
      bounded_options.max_inflight_per_stream = 1;
      bounded_options.max_inflight_total = 1;
      simaai::neat::runtime::DownstreamTarget bounded_target{
          simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
          0U,
          simaai::neat::graph::kInvalidPort,
          0U,
      };
      simaai::neat::runtime::RealtimeLatestLink bounded_link(bounded_target, bounded_options,
                                                             "bounded-stream");
      std::mutex bounded_mu;
      std::condition_variable bounded_cv;
      std::vector<std::int64_t> bounded_frames;
      int bounded_attached = 0;
      std::string bounded_error;
      bounded_link.start(
          [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
              std::size_t) {
            const auto sample_credits =
                simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample);
            const bool has_graph_private_credit =
                std::find_if(sample_credits.begin(), sample_credits.end(), [](const auto& item) {
                  return item.graph_private;
                }) != sample_credits.end();
            simaai::neat::pipeline_internal::release_realtime_frame_credits(sample_credits,
                                                                            "unit-bounded-output");
            {
              std::lock_guard<std::mutex> lock(bounded_mu);
              bounded_frames.push_back(sample.frame_id);
              if (has_graph_private_credit) {
                ++bounded_attached;
              }
            }
            bounded_cv.notify_all();
            return true;
          },
          [] { return false; },
          [&](const std::string& msg) {
            std::lock_guard<std::mutex> lock(bounded_mu);
            bounded_error = msg;
            bounded_cv.notify_all();
          });

      simaai::neat::Sample bounded_first = make_gst_sample_backed_sample(std::nullopt);
      bounded_first.stream_id = "bounded-stream";
      bounded_first.frame_id = 501;
      require(bounded_link.offer(std::move(bounded_first), 0U),
              "bounded realtime credit link should accept the first frame");
      {
        std::unique_lock<std::mutex> lock(bounded_mu);
        require(bounded_cv.wait_for(lock, std::chrono::seconds(1),
                                    [&] { return bounded_frames.size() == 1U; }),
                "bounded first frame should dispatch promptly");
      }

      simaai::neat::Sample bounded_second = make_gst_sample_backed_sample(std::nullopt);
      bounded_second.stream_id = "bounded-stream";
      bounded_second.frame_id = 502;
      require(bounded_link.offer(std::move(bounded_second), 0U),
              "bounded realtime credit link should accept the second frame");
      {
        std::unique_lock<std::mutex> lock(bounded_mu);
        require(bounded_cv.wait_for(lock, std::chrono::seconds(1),
                                    [&] { return bounded_frames.size() == 2U; }),
                "bounded second frame should dispatch after the first output releases credit");
        require(bounded_error.empty(), "bounded realtime link should not report an error");
      }
      bounded_link.close();
      const auto bounded_stats = bounded_link.stats();
      require(bounded_attached == 2,
              "bounded admission must attach releasable graph credits to dispatched samples");
      require(bounded_stats.credit_registered == 2U,
              "bounded lane should register both dispatched frame credits");
      require(bounded_stats.credit_released_by_output == 4U,
              "public per-stream and total lanes should both release through output pulls");
      require(bounded_stats.credit_inflight == 0U,
              "bounded lane must not leak an acquired global credit");
    }

    int wake_count = 0;
    auto lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(
        1, [&wake_count] { ++wake_count; });
    require(lane != nullptr && lane->gate != nullptr, "graph credit lane should be created");
    require(lane->gate->try_acquire(), "graph credit test should acquire the only credit");
    const std::uint64_t graph_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                graph_ns, "graph-stream", 101, lane),
            "graph credit should register with a private namespace");
    require(lane->registered.load() == 1U, "graph credit lane should count registration");
    require(lane->gate->inflight() == 1, "graph credit should remain in-flight while registered");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "graph-stream", 101}},
        "unit-unqualified");
    require(lane->gate->inflight() == 0,
            "unqualified output release should resolve graph private credit");
    require(lane->released_by_output.load() == 1U,
            "graph credit lane should count output releases");
    require(wake_count == 1, "graph credit release should notify the lane wake hook");

    require(lane->gate->try_acquire(), "fanout retain test should acquire graph credit");
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                graph_ns, "graph-stream", 1001, lane),
            "fanout retain test should register graph credit");
    require(simaai::neat::pipeline_internal::retain_registered_realtime_frame_credits(
                {simaai::neat::pipeline_internal::RealtimeFrameCredit{graph_ns, "graph-stream",
                                                                      1001, -1, -1, true}},
                1U, "unit-fanout-retain"),
            "fanout retain test should add one branch reference");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{graph_ns, "graph-stream", 1001, -1,
                                                              -1, true}},
        "unit-fanout-release-first");
    require(lane->gate->inflight() == 1,
            "first fanout branch release must not free the graph credit lane");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{graph_ns, "graph-stream", 1001, -1,
                                                              -1, true}},
        "unit-fanout-release-second");
    require(lane->gate->inflight() == 0,
            "last fanout branch release should free the graph credit lane");

    require(lane->gate->try_acquire(), "graph credit test should acquire alias credit");
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                graph_ns, "graph-stream", 102, lane),
            "graph credit should register an aliasable frame");
    simaai::neat::Sample sanitized;
    sanitized.stream_id = "graph-stream";
    sanitized.frame_id = 102;
    sanitized.input_seq = 55;
    sanitized.orig_input_seq = 1001;
    require(simaai::neat::pipeline_internal::alias_registered_realtime_frame_credits(
                {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "graph-stream", 102}},
                sanitized, "unit-alias"),
            "graph credit should alias by sanitized input sequence");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "neatprocess0", -1, 55, -1}},
        "unit-alias-release");
    require(lane->gate->inflight() == 0,
            "input-seq alias release should resolve graph private credit");
    require(lane->released_by_output.load() == 3U,
            "alias release should count as an output release");

    auto lane_a = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    auto lane_b = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    require(lane_a->gate->try_acquire() && lane_b->gate->try_acquire(),
            "same-orig alias test should acquire both lanes");
    const std::uint64_t alias_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                alias_ns, "alias-stream-a", 201, lane_a),
            "same-orig alias test should register stream A");
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                alias_ns, "alias-stream-b", 202, lane_b),
            "same-orig alias test should register stream B");
    simaai::neat::Sample alias_a;
    alias_a.stream_id = "alias-stream-a";
    alias_a.frame_id = 201;
    alias_a.input_seq = 2001;
    alias_a.orig_input_seq = 7;
    simaai::neat::Sample alias_b;
    alias_b.stream_id = "alias-stream-b";
    alias_b.frame_id = 202;
    alias_b.input_seq = 2002;
    alias_b.orig_input_seq = 7;
    require(simaai::neat::pipeline_internal::alias_registered_realtime_frame_credits(
                {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "alias-stream-a", 201}},
                alias_a, "unit-same-orig-a"),
            "same-orig alias test should alias stream A");
    require(simaai::neat::pipeline_internal::alias_registered_realtime_frame_credits(
                {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "alias-stream-b", 202}},
                alias_b, "unit-same-orig-b"),
            "same-orig alias test should alias stream B");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "alias-stream-b", -1, -1, 7}},
        "unit-same-orig-release-b");
    require(lane_a->gate->inflight() == 1,
            "stream B orig alias release must not release stream A credit");
    require(lane_b->gate->inflight() == 0,
            "stream B orig alias release should release stream B credit");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{0, "alias-stream-a", -1, -1, 7}},
        "unit-same-orig-release-a");
    require(lane_a->gate->inflight() == 0,
            "stream A orig alias release should still release stream A credit");

    auto stale_alias_lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    require(stale_alias_lane->gate->try_acquire(),
            "stale namespace alias test should acquire its lane");
    const std::uint64_t stale_ns_a =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    const std::uint64_t stale_ns_b =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                stale_ns_b, "stale-stream", 303, stale_alias_lane),
            "stale namespace alias test should register namespace B");
    simaai::neat::Sample stale_alias_sample;
    stale_alias_sample.stream_id = "stale-stream";
    stale_alias_sample.frame_id = 303;
    stale_alias_sample.input_seq = 3003;
    require(
        !simaai::neat::pipeline_internal::alias_registered_realtime_frame_credits(
            {simaai::neat::pipeline_internal::RealtimeFrameCredit{stale_ns_a, "stale-stream", 303}},
            stale_alias_sample, "unit-stale-namespace-alias"),
        "nonzero namespace alias must not fall back to another namespace on exact miss");
    simaai::neat::pipeline_internal::release_realtime_frame_credits(
        {simaai::neat::pipeline_internal::RealtimeFrameCredit{stale_ns_a, "stale-stream", -1, 3003,
                                                              -1}},
        "unit-stale-namespace-release");
    require(stale_alias_lane->gate->inflight() == 1,
            "stale namespace release must not release namespace B through an alias");
    simaai::neat::pipeline_internal::release_all_registered_realtime_frame_credits(
        stale_ns_b, "unit-stale-namespace-cleanup");
    require(stale_alias_lane->gate->inflight() == 0,
            "stale namespace cleanup should release namespace B");

    auto pending_lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    require(pending_lane->gate->try_acquire(),
            "pending overwrite test should acquire graph credit");
    const std::uint64_t pending_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                pending_ns, "pending-stream", 1, pending_lane),
            "pending overwrite test should register graph credit");
    simaai::neat::Sample pending_sample = make_gst_sample_backed_sample(std::nullopt);
    pending_sample.stream_id = "pending-stream";
    pending_sample.frame_id = 1;
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(
        pending_sample,
        simaai::neat::pipeline_internal::RealtimeFrameCredit{pending_ns, "pending-stream", 1});
    simaai::neat::Sample replacement_sample = make_gst_sample_backed_sample(std::nullopt);
    replacement_sample.stream_id = "pending-stream";
    replacement_sample.frame_id = 2;
    simaai::neat::RealtimeGraphLinkOptions pending_options;
    pending_options.queue_depth = 1;
    simaai::neat::runtime::DownstreamTarget pending_target;
    simaai::neat::runtime::RealtimeLatestLink pending_link(pending_target, pending_options,
                                                           "pending-stream");
    require(
        pending_link.offer(std::move(pending_sample), simaai::neat::runtime::invalid_edge_index()),
        "pending overwrite test should accept first sample");
    require(pending_link.offer(std::move(replacement_sample),
                               simaai::neat::runtime::invalid_edge_index()),
            "pending overwrite test should accept replacement sample");
    require(pending_lane->gate->inflight() == 0,
            "overwriting a pending realtime sample must release its carried graph credit");
    require(pending_lane->released_without_output.load() == 1U,
            "pending overwrite release should be counted as non-output release");
    pending_link.close();

    auto close_lane = simaai::neat::pipeline_internal::make_realtime_frame_credit_lane(1);
    require(close_lane->gate->try_acquire(), "pending close test should acquire graph credit");
    const std::uint64_t close_ns =
        simaai::neat::pipeline_internal::next_realtime_frame_credit_namespace();
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                close_ns, "close-stream", 1, close_lane),
            "pending close test should register graph credit");
    simaai::neat::Sample close_sample = make_gst_sample_backed_sample(std::nullopt);
    close_sample.stream_id = "close-stream";
    close_sample.frame_id = 1;
    simaai::neat::pipeline_internal::attach_realtime_frame_credit_to_sample(
        close_sample,
        simaai::neat::pipeline_internal::RealtimeFrameCredit{close_ns, "close-stream", 1});
    simaai::neat::runtime::RealtimeLatestLink close_link(pending_target, pending_options,
                                                         "close-stream");
    require(close_link.offer(std::move(close_sample), simaai::neat::runtime::invalid_edge_index()),
            "pending close test should accept sample");
    close_link.close();
    require(close_lane->gate->inflight() == 0,
            "closing a realtime link must release pending carried graph credit");
    require(close_lane->released_without_output.load() == 1U,
            "pending close release should be counted as non-output release");

    require(lane->gate->try_acquire(), "graph credit test should acquire another credit");
    require(simaai::neat::pipeline_internal::register_realtime_frame_credit(
                graph_ns, "graph-stream", 103, lane),
            "graph credit should register a second frame");
    simaai::neat::pipeline_internal::release_all_registered_realtime_frame_credits(
        graph_ns, "unit-release-all");
    require(lane->gate->inflight() == 0, "release-all should release graph private credits");
    require(lane->released_without_output.load() == 1U,
            "release-all should count as a non-output release");

    std::cout << "[OK] unit_latest_mux_credit_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

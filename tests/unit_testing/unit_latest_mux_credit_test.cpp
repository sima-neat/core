#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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
  gst_structure_set(structure, "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id), nullptr);
  GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_id) * GST_MSECOND;
  GST_BUFFER_DURATION(buffer) = GST_MSECOND;
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

void push_mux_input(GstElement* appsrc, std::int64_t frame_id) {
  require(gst_app_src_push_buffer(GST_APP_SRC(appsrc), make_mux_input_buffer(frame_id)) ==
              GST_FLOW_OK,
          "latest-mux appsrc push failed");
}

GstSample* pull_mux_output(GstElement* appsink, GstClockTime timeout) {
  return gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), timeout);
}

void test_namespace_bounded_terminal_loan_fallback() {
  GstElement* pipeline = gst_pipeline_new("latest-mux-credit-fallback-pipeline");
  GstElement* appsrc = gst_element_factory_make("appsrc", "latest-mux-credit-source");
  GstElement* mux = gst_element_factory_make("neatlatestbystreammux", "latest-mux-credit-mux");
  GstElement* appsink = gst_element_factory_make("appsink", "latest-mux-credit-sink");
  require(pipeline && appsrc && mux && appsink,
          "failed to construct latest-mux credit fallback pipeline");

  g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);
  g_object_set(mux, "stream-ids", "sole-stream", nullptr);
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
  gst_buffer_unref(stale_private);

  push_mux_input(appsrc, 104);
  GstSample* fourth = pull_mux_output(appsink, GST_SECOND);
  require(fourth != nullptr,
          "stale-private-key fallback should wake the mux and admit the next stream frame");
  GstBuffer* final_terminal = make_terminal_identity_buffer("sole-stream", 104);
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              final_terminal, mux_namespace),
          "exact terminal output should release the final test loan");
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

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    ::setenv("SIMA_LATEST_MUX_MAX_INFLIGHT_PER_STREAM", "1", 1);
    ::setenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_PER_STREAM", "0", 1);
    ::setenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL", "1", 1);

    test_namespace_bounded_terminal_loan_fallback();

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
      simaai::neat::GraphLinkOptions global_only_options;
      global_only_options.queue_depth = 1;
      simaai::neat::runtime::DownstreamTarget global_only_target{
          simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
          0U,
          simaai::neat::graph::kInvalidPort,
          0U,
      };
      simaai::neat::runtime::RealtimeLatestLink global_only_link(
          global_only_target, global_only_options, "global-only-stream");
      std::mutex global_only_mu;
      std::condition_variable global_only_cv;
      std::vector<std::int64_t> global_only_frames;
      int global_only_attached = 0;
      std::string global_only_error;
      global_only_link.start(
          [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
              std::size_t) {
            const auto sample_credits =
                simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample);
            const bool has_graph_private_credit =
                std::find_if(sample_credits.begin(), sample_credits.end(), [](const auto& item) {
                  return item.graph_private;
                }) != sample_credits.end();
            simaai::neat::pipeline_internal::release_realtime_frame_credits(
                sample_credits, "unit-global-only-output");
            {
              std::lock_guard<std::mutex> lock(global_only_mu);
              global_only_frames.push_back(sample.frame_id);
              if (has_graph_private_credit) {
                ++global_only_attached;
              }
            }
            global_only_cv.notify_all();
            return true;
          },
          [] { return false; },
          [&](const std::string& msg) {
            std::lock_guard<std::mutex> lock(global_only_mu);
            global_only_error = msg;
            global_only_cv.notify_all();
          });

      simaai::neat::Sample global_first = make_gst_sample_backed_sample(std::nullopt);
      global_first.stream_id = "global-only-stream";
      global_first.frame_id = 501;
      require(global_only_link.offer(std::move(global_first), 0U),
              "global-only realtime credit link should accept the first frame");
      {
        std::unique_lock<std::mutex> lock(global_only_mu);
        require(global_only_cv.wait_for(lock, std::chrono::seconds(1),
                                        [&] { return global_only_frames.size() == 1U; }),
                "global-only first frame should dispatch promptly");
      }

      simaai::neat::Sample global_second = make_gst_sample_backed_sample(std::nullopt);
      global_second.stream_id = "global-only-stream";
      global_second.frame_id = 502;
      require(global_only_link.offer(std::move(global_second), 0U),
              "global-only realtime credit link should accept the second frame");
      {
        std::unique_lock<std::mutex> lock(global_only_mu);
        require(global_only_cv.wait_for(lock, std::chrono::seconds(1),
                                        [&] { return global_only_frames.size() == 2U; }),
                "global-only second frame should dispatch after the first output releases credit");
        require(global_only_error.empty(), "global-only realtime link should not report an error");
      }
      global_only_link.close();
      const auto global_only_stats = global_only_link.stats();
      require(global_only_attached == 2,
              "global-only admission must attach releasable graph credits to dispatched samples");
      require(global_only_stats.credit_registered == 2U,
              "global-only lane should register both dispatched frame credits");
      require(global_only_stats.credit_released_by_output == 2U,
              "global-only lane should release credits through output pulls");
      require(global_only_stats.credit_inflight == 0U,
              "global-only lane must not leak an acquired global credit");
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
    simaai::neat::GraphLinkOptions pending_options;
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

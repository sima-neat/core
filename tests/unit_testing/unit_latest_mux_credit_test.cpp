#include "gst/GstInit.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
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
  return sample;
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    const simaai::neat::Sample namespaced =
        make_gst_sample_backed_sample(MuxLoanKey{1234, "mux-stream", 42});
    std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit> credits =
        simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(namespaced);
    require(credits.size() == 2U, "expected stamped and public fallback realtime credits");
    require(credits[0].namespace_id == 1234U, "credit should preserve mux namespace");
    require(credits[0].stream_id == "mux-stream", "credit should use stamped mux stream id");
    require(credits[0].frame_id == 42, "credit should use stamped mux frame id");
    require(credits[1].namespace_id == 0U, "fallback credit should remain unqualified");
    require(credits[1].stream_id == "fallback-stream", "fallback should use sample stream id");
    require(credits[1].frame_id == 7, "fallback should use sample frame id");

    simaai::neat::Sample bundled;
    bundled.kind = simaai::neat::SampleKind::Bundle;
    bundled.stream_id = "fallback-stream";
    bundled.frame_id = 7;
    bundled.fields.push_back(namespaced);
    credits = simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(bundled);
    require(credits.size() == 2U, "bundle should keep stamped child and public fallback credits");
    require(credits[0].namespace_id == 1234U, "bundle credit should preserve child mux namespace");
    require(credits[0].stream_id == "mux-stream", "bundle credit should use child mux stream id");
    require(credits[0].frame_id == 42, "bundle credit should use child mux frame id");
    require(credits[1].namespace_id == 0U, "bundle fallback credit should remain unqualified");
    require(credits[1].stream_id == "fallback-stream",
            "bundle fallback should use public sample stream id");
    require(credits[1].frame_id == 7, "bundle fallback should use public sample frame id");

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
    require(lane->released_by_output.load() == 2U,
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

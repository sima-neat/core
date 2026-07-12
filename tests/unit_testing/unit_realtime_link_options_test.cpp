#include "gst/GstInit.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

simaai::neat::Sample make_raw_holder_sample(std::string stream_id, std::int64_t frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

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
  sample.stream_id = std::move(stream_id);
  sample.frame_id = frame_id;
  sample.media_type = "video/x-raw";
  sample.payload_tag = "NV12";
  sample.format = "NV12";
  return sample;
}

simaai::neat::runtime::RealtimeLatestLink::Stats
run_one_raw_sample(const simaai::neat::GraphLinkOptions& options, int stream_count = 1,
                   bool use_explicit_stream_ids = true) {
  simaai::neat::runtime::DownstreamTarget target{
      simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
      0U,
      simaai::neat::graph::kInvalidPort,
      0U,
  };
  simaai::neat::runtime::RealtimeLatestLink link(target, options,
                                                 use_explicit_stream_ids ? "stream0" : "");
  for (int i = 1; i < stream_count; ++i) {
    link.add_edge_stream_id(static_cast<std::size_t>(i),
                            use_explicit_stream_ids ? "stream" + std::to_string(i) : "");
  }

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  link.start(
      [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
          std::size_t) {
        simaai::neat::pipeline_internal::release_realtime_frame_credits(
            simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample),
            "unit-realtime-link-options");
        {
          std::lock_guard<std::mutex> lock(mu);
          done = true;
        }
        cv.notify_all();
        return true;
      },
      [] { return false; },
      [&](std::string msg) {
        std::lock_guard<std::mutex> lock(mu);
        error = std::move(msg);
        done = true;
        cv.notify_all();
      });

  require(
      link.offer(make_raw_holder_sample("stream0", 1), simaai::neat::runtime::invalid_edge_index()),
      "RealtimeLatestLink should accept the sample");
  {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, std::chrono::seconds(2), [&] { return done; }),
            "timed out waiting for realtime link dispatch");
  }
  require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
  const auto stats = link.stats();
  link.close();
  link.join();
  return stats;
}

void verify_raw_frame_cap_blocks_after(const simaai::neat::GraphLinkOptions& options,
                                       int stream_count, int admitted_frames,
                                       const std::string& label,
                                       bool use_explicit_link_stream_ids = true) {
  simaai::neat::runtime::DownstreamTarget target{
      simaai::neat::runtime::DownstreamTarget::Kind::PipelineInput,
      0U,
      simaai::neat::graph::kInvalidPort,
      0U,
  };

  simaai::neat::runtime::RealtimeLatestLink link(target, options,
                                                 use_explicit_link_stream_ids ? "stream0" : "");
  for (int i = 1; i < stream_count; ++i) {
    link.add_edge_stream_id(static_cast<std::size_t>(i),
                            use_explicit_link_stream_ids ? "stream" + std::to_string(i) : "");
  }

  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::int64_t> dispatched_frames;
  std::vector<std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit>> dispatched_credits;
  std::string error;

  link.start(
      [&](const simaai::neat::runtime::DownstreamTarget&, simaai::neat::Sample&& sample,
          std::size_t) {
        const auto credits =
            simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(sample);
        {
          std::lock_guard<std::mutex> lock(mu);
          dispatched_frames.push_back(sample.frame_id);
          dispatched_credits.push_back(credits);
        }
        cv.notify_all();
        return true;
      },
      [] { return false; },
      [&](std::string msg) {
        std::lock_guard<std::mutex> lock(mu);
        error = std::move(msg);
        cv.notify_all();
      });

  const auto wait_for_dispatch_count = [&](std::size_t count) {
    std::unique_lock<std::mutex> lock(mu);
    require(cv.wait_for(lock, std::chrono::seconds(2),
                        [&] { return dispatched_frames.size() >= count || !error.empty(); }),
            "timed out waiting for realtime dispatch count " + std::to_string(count));
    require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
  };

  for (int frame = 0; frame < admitted_frames; ++frame) {
    const int stream = frame % stream_count;
    require(link.offer(make_raw_holder_sample("stream" + std::to_string(stream), frame),
                       simaai::neat::runtime::invalid_edge_index()),
            "RealtimeLatestLink should accept raw frame " + std::to_string(frame));
    wait_for_dispatch_count(static_cast<std::size_t>(frame + 1));
  }

  require(link.offer(make_raw_holder_sample("stream0", admitted_frames),
                     simaai::neat::runtime::invalid_edge_index()),
          "RealtimeLatestLink should accept a pending frame after " + label + " is full");
  {
    std::unique_lock<std::mutex> lock(mu);
    (void)cv.wait_for(lock, std::chrono::milliseconds(200), [&] {
      return dispatched_frames.size() > static_cast<std::size_t>(admitted_frames) || !error.empty();
    });
    require(error.empty(), "RealtimeLatestLink dispatch failed: " + error);
    require(dispatched_frames.size() == static_cast<std::size_t>(admitted_frames),
            label + " should hold the next frame pending");
  }
  require(link.stats().no_credit_skips > 0U,
          label + " should reject admission while raw frames are held");

  require(!dispatched_credits.empty(), "expected held realtime frame credits");
  simaai::neat::pipeline_internal::release_realtime_frame_credits(
      dispatched_credits.front(), "unit-realtime-link-options-release-one");
  wait_for_dispatch_count(static_cast<std::size_t>(admitted_frames + 1));

  link.close();
  link.join();
}

void verify_per_stream_cap_blocks_after_two_raw_frames() {
  simaai::neat::GraphLinkOptions options;
  options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.max_inflight_per_stream = 2;
  options.max_inflight_total = 10;
  verify_raw_frame_cap_blocks_after(options, 1, 2, "explicit per-stream raw-frame cap");
}

void verify_multiplexed_stream_ids_scale_default_total_cap() {
  simaai::neat::GraphLinkOptions options;
  options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.max_inflight_per_stream = 4;
  verify_raw_frame_cap_blocks_after(options, 4, 8,
                                    "multiplexed stream-id default global raw-frame cap",
                                    /*use_explicit_link_stream_ids=*/false);
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    ::unsetenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL");

    simaai::neat::GraphLinkOptions default_options;
    default_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    default_options.queue_depth = 1;
    verify_raw_frame_cap_blocks_after(default_options, 4, 8, "default global raw-frame cap",
                                      /*use_explicit_link_stream_ids=*/false);

    simaai::neat::GraphLinkOptions explicit_total_options;
    explicit_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    explicit_total_options.max_inflight_per_stream = 4;
    explicit_total_options.max_inflight_total = 10;
    verify_raw_frame_cap_blocks_after(explicit_total_options, 4, 10,
                                      "explicit global raw-frame cap above default");
    verify_per_stream_cap_blocks_after_two_raw_frames();
    verify_multiplexed_stream_ids_scale_default_total_cap();

    simaai::neat::GraphLinkOptions zero_options;
    zero_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    zero_options.max_inflight_per_stream = 0;
    bool rejected_zero = false;
    try {
      (void)run_one_raw_sample(zero_options);
    } catch (const std::runtime_error&) {
      rejected_zero = true;
    }
    require(rejected_zero, "max_inflight_per_stream=0 should be rejected");

    simaai::neat::GraphLinkOptions zero_total_options;
    zero_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    zero_total_options.max_inflight_total = 0;
    bool rejected_zero_total = false;
    try {
      (void)run_one_raw_sample(zero_total_options);
    } catch (const std::runtime_error&) {
      rejected_zero_total = true;
    }
    require(rejected_zero_total, "max_inflight_total=0 should be rejected");

    simaai::neat::GraphLinkOptions negative_options;
    negative_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    negative_options.max_inflight_per_stream = -2;
    bool rejected_negative = false;
    try {
      (void)run_one_raw_sample(negative_options);
    } catch (const std::runtime_error&) {
      rejected_negative = true;
    }
    require(rejected_negative, "max_inflight_per_stream<-1 should be rejected");

    simaai::neat::GraphLinkOptions negative_total_options;
    negative_total_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    negative_total_options.max_inflight_total = -2;
    bool rejected_negative_total = false;
    try {
      (void)run_one_raw_sample(negative_total_options);
    } catch (const std::runtime_error&) {
      rejected_negative_total = true;
    }
    require(rejected_negative_total, "max_inflight_total<-1 should be rejected");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return 0;
}

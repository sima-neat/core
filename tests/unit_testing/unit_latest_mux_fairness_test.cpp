#include "gst/GstInit.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSlotCount = 24U;

GstBuffer* make_input_buffer(std::int64_t frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  if (!buffer) {
    return nullptr;
  }
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!structure) {
    gst_buffer_unref(buffer);
    return nullptr;
  }
  gst_structure_set(structure, "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id), nullptr);
  GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_id) * GST_MSECOND;
  GST_BUFFER_DURATION(buffer) = GST_MSECOND;
  return buffer;
}

std::optional<std::size_t> stream_index(GstBuffer* buffer) {
  GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  const char* stream = structure ? gst_structure_get_string(structure, "orig-stream-id") : nullptr;
  if (!stream || !*stream) {
    stream = structure ? gst_structure_get_string(structure, "stream-id") : nullptr;
  }
  constexpr const char* prefix = "stream";
  if (!stream || !g_str_has_prefix(stream, prefix)) {
    return std::nullopt;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(stream + 6, &end, 10);
  if (end == stream + 6 || !end || *end != '\0' || parsed >= kSlotCount) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(parsed);
}

struct FairnessProbeState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<GstPad*> sink_pads;
  std::vector<std::size_t> picks;
  std::atomic<std::int64_t> next_frame_id{1};
  std::uint64_t observed = 0;
  bool block_next = false;
  bool blocked = false;
  bool unblock = false;
  bool collecting = false;
  bool callback_failed = false;
};

GstFlowReturn chain_one(FairnessProbeState* state, std::size_t index) {
  if (!state || index >= state->sink_pads.size() || !state->sink_pads[index]) {
    return GST_FLOW_ERROR;
  }
  GstBuffer* buffer =
      make_input_buffer(state->next_frame_id.fetch_add(1, std::memory_order_relaxed));
  if (!buffer) {
    return GST_FLOW_ERROR;
  }
  return gst_pad_chain(state->sink_pads[index], buffer);
}

GstPadProbeReturn fairness_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<FairnessProbeState*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const auto index = stream_index(buffer);
  if (!state || !index.has_value()) {
    if (state) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->callback_failed = true;
      state->cv.notify_all();
    }
    return GST_PAD_PROBE_OK;
  }

  bool trigger = false;
  bool replenish = false;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    ++state->observed;
    state->cv.notify_all();
    if (state->block_next) {
      state->block_next = false;
      state->blocked = true;
      trigger = true;
      state->cv.notify_all();
      state->cv.wait(lock, [&] { return state->unblock; });
    }
    if (!trigger && state->collecting && state->picks.size() < kSlotCount) {
      state->picks.push_back(*index);
      replenish = state->picks.size() < kSlotCount;
      state->cv.notify_all();
    }
  }

  // Requeue the selected stream before the mux worker returns from its push.
  // Thus all 24 streams remain continuously pending throughout the measured
  // scheduling window, rather than relying on appsrc thread timing.
  if (replenish && chain_one(state, *index) != GST_FLOW_OK) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->callback_failed = true;
    state->cv.notify_all();
  }
  return GST_PAD_PROBE_OK;
}

std::string stream_ids_csv() {
  std::string csv;
  for (std::size_t i = 0; i < kSlotCount; ++i) {
    if (!csv.empty()) {
      csv.push_back(',');
    }
    csv += "stream" + std::to_string(i);
  }
  return csv;
}

std::string stream_inflight_limits_csv(int limit) {
  std::string csv;
  for (std::size_t i = 0; i < kSlotCount; ++i) {
    if (!csv.empty()) {
      csv.push_back(',');
    }
    csv += std::to_string(limit);
  }
  return csv;
}

void wait_for_observation(FairnessProbeState* state, std::uint64_t before) {
  std::unique_lock<std::mutex> lock(state->mutex);
  require(state->cv.wait_for(lock, std::chrono::seconds(3),
                             [&] { return state->observed > before || state->callback_failed; }),
          "timed out waiting for latest-mux output");
  require(!state->callback_failed, "latest-mux fairness probe callback failed");
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    GstElement* pipeline = gst_pipeline_new("latest-mux-fairness-pipeline");
    GstElement* mux = gst_element_factory_make("neatlatestbystreammux", "latest-mux-fairness");
    GstElement* sink = gst_element_factory_make("fakesink", "latest-mux-fairness-sink");
    require(pipeline && mux && sink, "failed to construct latest-mux fairness pipeline");
    const std::string ids = stream_ids_csv();
    const std::string limits = stream_inflight_limits_csv(0);
    // This test isolates scheduling from completion credit through the mux's
    // construction-time property instead of a process-global private switch.
    g_object_set(mux, "stream-ids", ids.c_str(), "stream-inflight-limits", limits.c_str(), nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), mux, sink, nullptr);
    require(gst_element_link(mux, sink) == TRUE, "failed to link latest-mux to fakesink");

    FairnessProbeState state;
    state.sink_pads.reserve(kSlotCount);
    for (std::size_t i = 0; i < kSlotCount; ++i) {
      const std::string name = "sink_" + std::to_string(i);
      GstPad* pad = gst_element_request_pad_simple(mux, name.c_str());
      require(pad != nullptr, "failed to request latest-mux sink pad");
      state.sink_pads.push_back(pad);
    }
    GstPad* mux_src = gst_element_get_static_pad(mux, "src");
    require(mux_src != nullptr, "latest-mux src pad missing");
    gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER, fairness_probe, &state, nullptr);
    gst_object_unref(mux_src);

    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "failed to start latest-mux fairness pipeline");

    // Give streams 0-3 a lifetime head start. Only one buffer is pending at a
    // time during warmup, so both the old lifetime-min policy and strict RR
    // deterministically emit it.
    for (int round = 0; round < 3; ++round) {
      for (std::size_t index = 0; index < 4U; ++index) {
        std::uint64_t before = 0;
        {
          std::lock_guard<std::mutex> lock(state.mutex);
          before = state.observed;
        }
        require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux warmup chain failed");
        wait_for_observation(&state, before);
      }
    }

    // Emit one trigger from slot 0 and stop the worker inside its src push. Its
    // next cursor is now slot 1, while slot 0 has emitted four times, slots 1-3
    // three times, and slots 4-23 zero times.
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.block_next = true;
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux trigger chain failed");
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(lock, std::chrono::seconds(3), [&] { return state.blocked; }),
              "latest-mux worker did not reach the blocking probe");
    }

    // With the sole worker blocked, stage one pending buffer in every slot.
    // The probe replenishes each winner synchronously, keeping every slot ready
    // for all 24 measured selections.
    for (std::size_t index = 0; index < kSlotCount; ++index) {
      require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux staging chain failed");
    }
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.collecting = true;
      state.unblock = true;
      state.cv.notify_all();
    }
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(
                  lock, std::chrono::seconds(5),
                  [&] { return state.picks.size() == kSlotCount || state.callback_failed; }),
              "timed out collecting a complete latest-mux fairness rotation");
      require(!state.callback_failed, "latest-mux fairness probe callback failed");
      require(state.picks.size() == kSlotCount,
              "latest-mux fairness rotation produced too few selections");
      for (std::size_t i = 0; i < kSlotCount; ++i) {
        const std::size_t expected = (i + 1U) % kSlotCount;
        require(state.picks[i] == expected,
                "latest-mux must select the first eligible stream from its RR cursor");
      }
    }

    require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
            "failed to stop latest-mux fairness pipeline");
    for (GstPad* pad : state.sink_pads) {
      gst_element_release_request_pad(mux, pad);
      gst_object_unref(pad);
    }
    gst_object_unref(pipeline);

    std::cout << "[OK] unit_latest_mux_fairness_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}

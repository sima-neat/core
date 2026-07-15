#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"

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
#include <utility>
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

std::optional<std::int64_t> frame_id(GstBuffer* buffer) {
  GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  gint64 value = -1;
  if (!structure || gst_structure_get_int64(structure, "frame-id", &value) != TRUE || value < 0) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value);
}

struct FairnessProbeState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<GstPad*> sink_pads;
  std::vector<std::size_t> picks;
  std::vector<std::int64_t> pick_frame_ids;
  std::atomic<std::int64_t> next_frame_id{1};
  std::uint64_t observed = 0;
  bool block_next = false;
  bool blocked = false;
  bool unblock = false;
  bool collecting = false;
  bool block_first_collected = false;
  bool collection_blocked = false;
  bool unblock_collection = false;
  bool callback_failed = false;
  std::uint64_t mux_namespace = 0;
  bool release_loans = false;
  bool hold_next_stream0_loan = false;
  GstBuffer* held_loan = nullptr;
};

GstFlowReturn chain_one(FairnessProbeState* state, std::size_t index,
                        std::int64_t* chained_frame_id = nullptr) {
  if (!state || index >= state->sink_pads.size() || !state->sink_pads[index]) {
    return GST_FLOW_ERROR;
  }
  const std::int64_t id = state->next_frame_id.fetch_add(1, std::memory_order_relaxed);
  GstBuffer* buffer = make_input_buffer(id);
  if (!buffer) {
    return GST_FLOW_ERROR;
  }
  if (chained_frame_id) {
    *chained_frame_id = id;
  }
  return gst_pad_chain(state->sink_pads[index], buffer);
}

GstPadProbeReturn fairness_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<FairnessProbeState*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const auto index = stream_index(buffer);
  const auto id = frame_id(buffer);
  if (!state || !index.has_value() || !id.has_value()) {
    if (state) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->callback_failed = true;
      state->cv.notify_all();
    }
    return GST_PAD_PROBE_OK;
  }

  bool trigger = false;
  bool replenish = false;
  bool release_loan = false;
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
      state->pick_frame_ids.push_back(*id);
      replenish = state->picks.size() < kSlotCount;
      state->cv.notify_all();
      if (state->block_first_collected) {
        state->block_first_collected = false;
        state->collection_blocked = true;
        state->cv.notify_all();
        state->cv.wait(lock, [&] { return state->unblock_collection; });
      }
    }
    if (state->release_loans) {
      if (state->hold_next_stream0_loan && *index == 0U) {
        state->hold_next_stream0_loan = false;
        state->held_loan = gst_buffer_ref(buffer);
        state->cv.notify_all();
      } else {
        release_loan = true;
      }
    }
  }

  if (release_loan &&
      !simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
          buffer, state->mux_namespace)) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->callback_failed = true;
    state->cv.notify_all();
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

void start_mux_epoch(const FairnessProbeState& state, std::uint64_t epoch) {
  for (std::size_t index = 0; index < state.sink_pads.size(); ++index) {
    const std::string stream =
        "fairness-input-" + std::to_string(epoch) + "-" + std::to_string(index);
    require(gst_pad_send_event(state.sink_pads[index],
                               gst_event_new_stream_start(stream.c_str())) == TRUE,
            "failed to send latest-mux fairness stream-start event");
    GstSegment segment;
    gst_segment_init(&segment, GST_FORMAT_TIME);
    require(gst_pad_send_event(state.sink_pads[index], gst_event_new_segment(&segment)) == TRUE,
            "failed to send latest-mux fairness segment event");
  }
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
    const std::string limits = stream_inflight_limits_csv(1);
    // Exercise the same per-stream terminal gate used by the fused graph. The
    // probe normally completes each loan synchronously; the final scenario
    // deliberately holds one to test credit-aware ready ordering.
    g_object_set(mux, "stream-ids", ids.c_str(), "stream-inflight-limits", limits.c_str(), nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), mux, sink, nullptr);
    require(gst_element_link(mux, sink) == TRUE, "failed to link latest-mux to fakesink");

    FairnessProbeState state;
    state.mux_namespace = simaai::neat::latest_by_stream_mux_namespace(mux);
    require(state.mux_namespace != 0U, "latest-mux fairness namespace is missing");
    state.release_loans = true;
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
    start_mux_epoch(state, 1U);

    // Give streams 0-3 a lifetime head start. Streams 4-23 have not joined the
    // scheduling epoch yet.
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
    // three times, and slots 4-23 have never been active.
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

    // With the sole worker blocked, make late-joining slot 23 oldest-ready,
    // then replace it several times. Replacement must retain its original
    // ticket while the pending payload remains latest-only. A new participant
    // enters at the current service frontier rather than inheriting a lifetime
    // deficit, while established streams retain their dispatch recency.
    std::int64_t newest_slot23_frame = -1;
    for (int replacement = 0; replacement < 4; ++replacement) {
      require(chain_one(&state, 23U, &newest_slot23_frame) == GST_FLOW_OK,
              "latest-mux replacement staging chain failed");
    }
    for (std::size_t index = 1; index + 1U < kSlotCount; ++index) {
      require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux staging chain failed");
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux final staging chain failed");
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
      require(state.pick_frame_ids.size() == kSlotCount,
              "latest-mux fairness rotation lost frame identities");
      require(state.picks[0] == 1U && state.picks[1] == 2U && state.picks[2] == 3U,
              "late join must not overtake established streams with older service recency");
      require(state.picks[3] == 23U,
              "late join must enter promptly at the current service frontier");
      require(state.pick_frame_ids[3] == newest_slot23_frame,
              "ready-ticket retention must still emit the newest replacement frame");
      for (std::size_t i = 4; i + 1U < kSlotCount; ++i) {
        require(state.picks[i] == i,
                "new streams at the same service frontier must preserve ready order");
      }
      require(state.picks.back() == 0U,
              "the most recently serviced established stream must remain at the frontier tail");
    }

    // Reset the mux epoch, then make every stream active before introducing a
    // real service deficit. This distinguishes a temporarily late stream from
    // a stream which simply joined late above.
    require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
            "failed to reset latest-mux fairness pipeline");
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.picks.clear();
      state.pick_frame_ids.clear();
      state.block_next = false;
      state.blocked = false;
      state.unblock = false;
      state.collecting = false;
      state.callback_failed = false;
    }
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "failed to restart latest-mux fairness pipeline");
    start_mux_epoch(state, 2U);

    for (std::size_t index = 0; index < kSlotCount; ++index) {
      std::uint64_t before = 0;
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        before = state.observed;
      }
      require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux activation chain failed");
      wait_for_observation(&state, before);
    }
    for (int round = 0; round < 3; ++round) {
      for (std::size_t index = 0; index + 1U < kSlotCount; ++index) {
        std::uint64_t before = 0;
        {
          std::lock_guard<std::mutex> lock(state.mutex);
          before = state.observed;
        }
        require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux deficit warmup chain failed");
        wait_for_observation(&state, before);
      }
    }

    // Slot 23 is now three lifetime dispatches behind every peer. Block on a
    // slot-0 trigger, make slot 23 ready last, and then keep every stream
    // pending. The old lifetime-min policy would dispatch slot 23 repeatedly
    // to repay historical deficit, while oldest-ready would put it behind all
    // peers. Bounded service-recency fairness gives it one prompt turn, then
    // places its synchronous replenishment behind all peers.
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.block_next = true;
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux catch-up trigger chain failed");
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(lock, std::chrono::seconds(3), [&] { return state.blocked; }),
              "latest-mux catch-up worker did not reach the blocking probe");
    }
    require(chain_one(&state, 23U) == GST_FLOW_OK, "latest-mux catch-up staging chain failed");
    for (std::size_t index = 1; index + 1U < kSlotCount; ++index) {
      require(chain_one(&state, index) == GST_FLOW_OK, "latest-mux catch-up staging chain failed");
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux catch-up final staging chain failed");
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
              "timed out collecting latest-mux catch-up selections");
      require(!state.callback_failed, "latest-mux catch-up probe callback failed");
      require(state.picks.size() == kSlotCount,
              "latest-mux catch-up window produced too few selections");
      require(state.picks[0] == 23U,
              "a late-but-ready stream must receive one prompt fairness turn");
      for (std::size_t i = 1; i + 1U < kSlotCount; ++i) {
        require(state.picks[i] == i,
                "a late stream's new ticket must follow every already-ready peer");
      }
      require(state.picks.back() == 0U,
              "bounded catch-up must contain every stream exactly once, without a burst");
    }

    // Credit-blocked oldest-ready stream: hold stream 0's first terminal loan,
    // then queue it ahead of streams 2 and 1 while the worker is stopped in a
    // stream-1 trigger. Stream 0 must not head-of-line block eligible work.
    // Once its credit is returned, it gets exactly one prompt old-ticket turn;
    // synchronous replenishment then moves it behind both ready peers.
    require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
            "failed to reset latest-mux credit fairness pipeline");
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.picks.clear();
      state.pick_frame_ids.clear();
      state.block_next = false;
      state.blocked = false;
      state.unblock = false;
      state.collecting = false;
      state.block_first_collected = false;
      state.collection_blocked = false;
      state.unblock_collection = false;
      state.callback_failed = false;
      state.hold_next_stream0_loan = true;
      require(state.held_loan == nullptr, "latest-mux fairness test leaked a held loan");
    }
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "failed to restart latest-mux credit fairness pipeline");
    start_mux_epoch(state, 3U);

    std::uint64_t before = 0;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      before = state.observed;
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux held-credit activation failed");
    wait_for_observation(&state, before);
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      require(state.held_loan != nullptr, "latest-mux probe did not retain stream-0 loan");
      state.block_next = true;
    }
    require(chain_one(&state, 1U) == GST_FLOW_OK, "latest-mux credit-fairness trigger failed");
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(lock, std::chrono::seconds(3), [&] { return state.blocked; }),
              "latest-mux credit-fairness worker did not reach the blocking probe");
    }
    require(chain_one(&state, 0U) == GST_FLOW_OK, "latest-mux blocked-credit staging failed");
    require(chain_one(&state, 2U) == GST_FLOW_OK, "latest-mux eligible staging failed");
    require(chain_one(&state, 1U) == GST_FLOW_OK, "latest-mux trigger-stream staging failed");
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.collecting = true;
      state.block_first_collected = true;
      state.unblock = true;
      state.cv.notify_all();
    }
    GstBuffer* held_loan = nullptr;
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(lock, std::chrono::seconds(3),
                                [&] { return state.collection_blocked; }),
              "eligible work did not bypass the credit-blocked oldest stream");
      require(!state.picks.empty() && state.picks.front() == 2U,
              "credit-blocked oldest ticket must be skipped without blocking eligible work");
      held_loan = std::exchange(state.held_loan, nullptr);
    }
    require(held_loan != nullptr, "latest-mux held loan disappeared before release");
    require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                held_loan, state.mux_namespace),
            "failed to return held latest-mux stream credit");
    gst_buffer_unref(held_loan);
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.unblock_collection = true;
      state.cv.notify_all();
    }
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(
                  lock, std::chrono::seconds(5),
                  [&] { return state.picks.size() == kSlotCount || state.callback_failed; }),
              "timed out collecting credit-fairness selections");
      require(!state.callback_failed, "latest-mux credit-fairness probe failed");
      require(state.picks[1] == 0U,
              "restored credit must promptly service the preserved oldest ticket");
      require(state.picks[2] == 1U && state.picks[3] == 2U && state.picks[4] == 0U,
              "restored stream must receive one priority turn, not a catch-up burst");
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

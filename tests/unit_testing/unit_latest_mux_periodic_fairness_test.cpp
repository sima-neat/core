#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSlotCount = 24U;
constexpr std::size_t kTotalInflight = 16U;
constexpr int kPerStreamInflight = 4;
constexpr std::size_t kServiceEvents = 1260U;
constexpr int kServiceEventsPerInputPeriod = 21;

// Fixed, deliberately uneven phases model 24 20-fps producers against a
// 420-fps consumer. With oldest-pending scheduling, the same source phases win
// each overloaded period and the dispatch spread grows without bound.
constexpr std::array<int, kSlotCount> kInputPhases = {
    4, 18, 4, 14, 0, 18, 15, 7, 4, 8, 12, 6, 15, 0, 7, 9, 19, 7, 20, 9, 16, 2, 16, 1,
};

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
  if (!stream || !g_str_has_prefix(stream, "stream")) {
    return std::nullopt;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(stream + 6, &end, 10);
  if (end == stream + 6 || !end || *end != '\0' || parsed >= kSlotCount) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(parsed);
}

struct ProbeState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<GstPad*> sink_pads;
  std::deque<GstBuffer*> held_loans;
  std::array<std::uint64_t, kSlotCount> inputs{};
  std::array<std::uint64_t, kSlotCount> outputs{};
  std::int64_t next_frame_id = 1;
  std::uint64_t observed = 0;
  std::uint64_t mux_namespace = 0;
  bool block_first = true;
  bool first_blocked = false;
  bool unblock_first = false;
  bool callback_failed = false;
};

GstFlowReturn chain_one(ProbeState* state, std::size_t index) {
  if (!state || index >= state->sink_pads.size() || !state->sink_pads[index]) {
    return GST_FLOW_ERROR;
  }
  GstBuffer* buffer = make_input_buffer(state->next_frame_id++);
  if (!buffer) {
    return GST_FLOW_ERROR;
  }
  ++state->inputs[index];
  return gst_pad_chain(state->sink_pads[index], buffer);
}

GstPadProbeReturn output_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<ProbeState*>(user_data);
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

  std::unique_lock<std::mutex> lock(state->mutex);
  ++state->outputs[*index];
  ++state->observed;
  state->held_loans.push_back(gst_buffer_ref(buffer));
  state->cv.notify_all();
  if (state->block_first) {
    state->block_first = false;
    state->first_blocked = true;
    state->cv.notify_all();
    state->cv.wait(lock, [&] { return state->unblock_first; });
  }
  return GST_PAD_PROBE_OK;
}

std::string csv_values(const std::string& prefix, int numeric_value) {
  std::string csv;
  for (std::size_t i = 0; i < kSlotCount; ++i) {
    if (!csv.empty()) {
      csv.push_back(',');
    }
    csv += prefix.empty() ? std::to_string(numeric_value) : prefix + std::to_string(i);
  }
  return csv;
}

void start_mux_epoch(const ProbeState& state) {
  for (std::size_t index = 0; index < state.sink_pads.size(); ++index) {
    const std::string stream = "periodic-input-" + std::to_string(index);
    require(gst_pad_send_event(state.sink_pads[index],
                               gst_event_new_stream_start(stream.c_str())) == TRUE,
            "failed to send periodic fairness stream-start event");
    GstSegment segment;
    gst_segment_init(&segment, GST_FORMAT_TIME);
    require(gst_pad_send_event(state.sink_pads[index], gst_event_new_segment(&segment)) == TRUE,
            "failed to send periodic fairness segment event");
  }
}

void wait_for_observed(ProbeState* state, std::uint64_t target) {
  std::unique_lock<std::mutex> lock(state->mutex);
  require(state->cv.wait_for(lock, std::chrono::seconds(3),
                             [&] { return state->observed >= target || state->callback_failed; }),
          "timed out waiting for periodic latest-mux output");
  require(!state->callback_failed, "periodic latest-mux output probe failed");
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    GstElement* pipeline = gst_pipeline_new("latest-mux-periodic-fairness-pipeline");
    GstElement* mux =
        gst_element_factory_make("neatlatestbystreammux", "latest-mux-periodic-fairness");
    GstElement* sink = gst_element_factory_make("fakesink", "latest-mux-periodic-fairness-sink");
    require(pipeline && mux && sink, "failed to construct periodic latest-mux pipeline");

    const std::string ids = csv_values("stream", 0);
    const std::string limits = csv_values("", kPerStreamInflight);
    g_object_set(mux, "stream-ids", ids.c_str(), "stream-inflight-limits", limits.c_str(),
                 "max-inflight-total", static_cast<int>(kTotalInflight), nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), mux, sink, nullptr);
    require(gst_element_link(mux, sink) == TRUE, "failed to link periodic latest-mux to fakesink");

    ProbeState state;
    state.mux_namespace = simaai::neat::latest_by_stream_mux_namespace(mux);
    require(state.mux_namespace != 0U, "periodic latest-mux namespace is missing");
    state.sink_pads.reserve(kSlotCount);
    for (std::size_t i = 0; i < kSlotCount; ++i) {
      const std::string name = "sink_" + std::to_string(i);
      GstPad* pad = gst_element_request_pad_simple(mux, name.c_str());
      require(pad != nullptr, "failed to request periodic latest-mux sink pad");
      state.sink_pads.push_back(pad);
    }
    GstPad* mux_src = gst_element_get_static_pad(mux, "src");
    require(mux_src != nullptr, "periodic latest-mux src pad missing");
    gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER, output_probe, &state, nullptr);
    gst_object_unref(mux_src);

    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "failed to start periodic latest-mux pipeline");
    start_mux_epoch(state);

    // Stop the only mux worker in its first downstream push. This lets every
    // producer join the same fairness frontier before terminal credits fill.
    require(chain_one(&state, 0U) == GST_FLOW_OK, "periodic fairness trigger chain failed");
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      require(state.cv.wait_for(lock, std::chrono::seconds(3), [&] { return state.first_blocked; }),
              "periodic latest-mux worker did not reach the blocking probe");
    }
    for (std::size_t index = 0; index < kSlotCount; ++index) {
      require(chain_one(&state, index) == GST_FLOW_OK,
              "periodic latest-mux initial staging failed");
    }
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.unblock_first = true;
      state.cv.notify_all();
    }
    wait_for_observed(&state, kTotalInflight);

    // Keep the total terminal-credit window saturated. Each virtual service
    // completion releases the oldest held loan after staging every input whose
    // fixed phase precedes that completion. The test runs much faster than
    // wall clock but preserves the overloaded 24*20 versus 420 cadence.
    for (std::size_t service = 0; service < kServiceEvents; ++service) {
      const int phase = static_cast<int>(service % kServiceEventsPerInputPeriod);
      for (std::size_t index = 0; index < kSlotCount; ++index) {
        if (kInputPhases[index] == phase) {
          require(chain_one(&state, index) == GST_FLOW_OK,
                  "periodic latest-mux phase input failed");
        }
      }

      GstBuffer* completed = nullptr;
      std::uint64_t before = 0;
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        require(state.held_loans.size() == kTotalInflight,
                "periodic latest-mux terminal window lost saturation");
        completed = state.held_loans.front();
        state.held_loans.pop_front();
        before = state.observed;
      }
      require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
                  completed, state.mux_namespace),
              "failed to release periodic latest-mux terminal credit");
      gst_buffer_unref(completed);
      wait_for_observed(&state, before + 1U);
    }

    const auto [minimum, maximum] = std::minmax_element(state.outputs.begin(), state.outputs.end());
    require(maximum != state.outputs.end() && minimum != state.outputs.end(),
            "periodic latest-mux produced no stream counters");
    const std::uint64_t spread = *maximum - *minimum;
    require(spread <= 2U, "periodic latest-mux dispatch skew exceeded bounded service recency: " +
                              std::to_string(spread));
    for (std::size_t index = 0; index < kSlotCount; ++index) {
      require(state.inputs[index] > state.outputs[index],
              "periodic overload did not exercise latest-frame replacement");
    }

    require(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
            "failed to stop periodic latest-mux pipeline");
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      while (!state.held_loans.empty()) {
        gst_buffer_unref(state.held_loans.front());
        state.held_loans.pop_front();
      }
    }
    for (GstPad* pad : state.sink_pads) {
      gst_element_release_request_pad(mux, pad);
      gst_object_unref(pad);
    }
    gst_object_unref(pipeline);

    std::cout << "[OK] unit_latest_mux_periodic_fairness_test passed: spread=" << spread << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}

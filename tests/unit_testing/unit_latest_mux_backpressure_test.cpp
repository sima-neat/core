#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct OutputFrame {
  std::string stream_id;
  std::int64_t frame_id = -1;
};

GstBuffer* make_buffer(std::int64_t frame_id) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate latest-mux backpressure buffer");
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  require(structure != nullptr, "failed to attach latest-mux backpressure metadata");
  gst_structure_set(structure, "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id), nullptr);
  GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(frame_id) * GST_MSECOND;
  GST_BUFFER_DURATION(buffer) = GST_MSECOND;
  return buffer;
}

OutputFrame read_output(GstBuffer* buffer) {
  GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  require(structure != nullptr, "latest-mux backpressure output metadata is missing");
  const char* stream = gst_structure_get_string(structure, "orig-stream-id");
  if (!stream || !*stream) {
    stream = gst_structure_get_string(structure, "stream-id");
  }
  gint64 frame = -1;
  require(stream && *stream && gst_structure_get_int64(structure, "frame-id", &frame) == TRUE &&
              frame >= 0,
          "latest-mux backpressure output identity is invalid");
  return OutputFrame{stream, static_cast<std::int64_t>(frame)};
}

struct ProbeState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<OutputFrame> outputs;
  bool block_next = false;
  bool blocked = false;
  bool unblock = false;
  std::uint64_t mux_namespace = 0;
  bool auto_release_loans = false;
  bool hold_next_loan = false;
  bool hold_all_loans = false;
  GstBuffer* held_loan = nullptr;
  std::vector<GstBuffer*> held_loans;
  std::chrono::milliseconds output_delay{0};
  bool callback_failed = false;
};

GstPadProbeReturn output_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<ProbeState*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!state || !buffer) {
    return GST_PAD_PROBE_OK;
  }
  OutputFrame output;
  try {
    output = read_output(buffer);
  } catch (...) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->callback_failed = true;
    state->cv.notify_all();
    return GST_PAD_PROBE_OK;
  }
  bool release_loan = false;
  std::chrono::milliseconds delay{0};
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->block_next) {
      state->block_next = false;
      state->blocked = true;
      state->cv.notify_all();
      state->cv.wait(lock, [&] { return state->unblock; });
      return GST_PAD_PROBE_OK;
    }
    state->outputs.push_back(output);
    if (state->hold_all_loans) {
      state->held_loans.push_back(gst_buffer_ref(buffer));
    }
    if (state->auto_release_loans) {
      if (state->hold_next_loan) {
        state->hold_next_loan = false;
        state->held_loan = gst_buffer_ref(buffer);
      } else {
        release_loan = true;
      }
    }
    delay = state->output_delay;
    state->cv.notify_all();
  }
  if (release_loan) {
    if (!simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
            buffer, state->mux_namespace)) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->callback_failed = true;
      state->cv.notify_all();
    }
  }
  if (delay.count() > 0) {
    std::this_thread::sleep_for(delay);
  }
  return GST_PAD_PROBE_OK;
}

struct Fixture {
  GstElement* pipeline = nullptr;
  GstElement* mux = nullptr;
  GstElement* sink = nullptr;
  std::vector<GstPad*> pads;
  ProbeState probe;
};

void init_fixture(Fixture* fixture, const char* name, const char* inflight_limits = "0,0",
                  int max_inflight_total = 0) {
  require(fixture != nullptr, "latest-mux backpressure fixture pointer is null");
  Fixture& f = *fixture;
  f.pipeline = gst_pipeline_new(name);
  f.mux = gst_element_factory_make("neatlatestbystreammux", nullptr);
  f.sink = gst_element_factory_make("fakesink", nullptr);
  require(f.pipeline && f.mux && f.sink, "failed to create latest-mux backpressure fixture");
  g_object_set(f.mux, "stream-ids", "stream0,stream1", "max-inflight-total", max_inflight_total,
               nullptr);
  if (inflight_limits) {
    g_object_set(f.mux, "stream-inflight-limits", inflight_limits, nullptr);
  }
  g_object_set(f.sink, "sync", FALSE, "async", FALSE, nullptr);
  gst_bin_add_many(GST_BIN(f.pipeline), f.mux, f.sink, nullptr);
  require(gst_element_link(f.mux, f.sink) == TRUE,
          "failed to link latest-mux backpressure fixture");
  f.probe.mux_namespace = simaai::neat::latest_by_stream_mux_namespace(f.mux);
  for (std::size_t index = 0; index < 2U; ++index) {
    const std::string pad_name = "sink_" + std::to_string(index);
    GstPad* pad = gst_element_request_pad_simple(f.mux, pad_name.c_str());
    require(pad != nullptr, "failed to request latest-mux backpressure sink pad");
    f.pads.push_back(pad);
  }
  GstPad* src = gst_element_get_static_pad(f.mux, "src");
  require(src != nullptr, "latest-mux backpressure src pad is missing");
  gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER, output_probe, &f.probe, nullptr);
  gst_object_unref(src);
  require(gst_element_set_state(f.pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start latest-mux backpressure fixture");
  for (std::size_t index = 0; index < f.pads.size(); ++index) {
    const std::string stream = std::string(name) + "-input-" + std::to_string(index);
    require(gst_pad_send_event(f.pads[index], gst_event_new_stream_start(stream.c_str())) == TRUE,
            "failed to send latest-mux backpressure stream-start");
    GstSegment segment;
    gst_segment_init(&segment, GST_FORMAT_TIME);
    require(gst_pad_send_event(f.pads[index], gst_event_new_segment(&segment)) == TRUE,
            "failed to send latest-mux backpressure segment");
  }
}

void stop_fixture(Fixture* f) {
  if (!f || !f->pipeline) {
    return;
  }
  require(gst_element_set_state(f->pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
          "failed to stop latest-mux backpressure fixture");
  std::vector<GstBuffer*> held_loans;
  {
    std::lock_guard<std::mutex> lock(f->probe.mutex);
    held_loans.swap(f->probe.held_loans);
  }
  for (GstBuffer* buffer : held_loans) {
    gst_buffer_unref(buffer);
  }
  for (GstPad*& pad : f->pads) {
    if (!pad) {
      continue;
    }
    gst_element_release_request_pad(f->mux, pad);
    gst_object_unref(pad);
    pad = nullptr;
  }
  gst_object_unref(f->pipeline);
  f->pipeline = nullptr;
}

void arm_blocking_probe(Fixture* f) {
  {
    std::lock_guard<std::mutex> lock(f->probe.mutex);
    f->probe.block_next = true;
  }
  require(gst_pad_chain(f->pads[1], make_buffer(1000)) == GST_FLOW_OK,
          "failed to chain latest-mux probe trigger");
  std::unique_lock<std::mutex> lock(f->probe.mutex);
  require(f->probe.cv.wait_for(lock, std::chrono::seconds(3),
                               [&] { return f->probe.blocked || f->probe.callback_failed; }),
          "latest-mux worker did not enter blocking probe");
  require(!f->probe.callback_failed, "latest-mux blocking probe callback failed");
}

void unblock_probe(Fixture* f) {
  std::lock_guard<std::mutex> lock(f->probe.mutex);
  f->probe.unblock = true;
  f->probe.cv.notify_all();
}

void wait_for_outputs(Fixture* f, std::size_t count) {
  std::unique_lock<std::mutex> lock(f->probe.mutex);
  require(f->probe.cv.wait_for(
              lock, std::chrono::seconds(3),
              [&] { return f->probe.outputs.size() >= count || f->probe.callback_failed; }),
          "timed out waiting for latest-mux backpressure outputs");
  require(!f->probe.callback_failed, "latest-mux backpressure probe callback failed");
}

void test_default_mode_keeps_latest() {
  Fixture f;
  init_fixture(&f, "latest-mux-default-latest");
  arm_blocking_probe(&f);
  require(gst_pad_chain(f.pads[0], make_buffer(1)) == GST_FLOW_OK,
          "default latest mode rejected first pending frame");
  require(gst_pad_chain(f.pads[0], make_buffer(2)) == GST_FLOW_OK,
          "default latest mode rejected replacement frame");
  unblock_probe(&f);
  wait_for_outputs(&f, 1U);
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    require(f.probe.outputs.size() == 1U && f.probe.outputs[0].stream_id == "stream0" &&
                f.probe.outputs[0].frame_id == 2,
            "default latest mode must replace the pending frame without blocking");
  }
  stop_fixture(&f);
}

void test_unconfigured_direct_mux_does_not_take_terminal_loans() {
  Fixture f;
  init_fixture(&f, "latest-mux-direct-no-loans", nullptr);
  constexpr std::size_t kFrames = 12U;
  for (std::size_t i = 0; i < kFrames; ++i) {
    const std::size_t stream = i % f.pads.size();
    require(gst_pad_chain(f.pads[stream], make_buffer(static_cast<std::int64_t>(i + 1U))) ==
                GST_FLOW_OK,
            "unconfigured direct mux rejected input");
    wait_for_outputs(&f, i + 1U);
  }
  stop_fixture(&f);
}

void test_credit_release_emits_latest_pending_frame() {
  Fixture f;
  init_fixture(&f, "latest-mux-latest-credit", "1,1");
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    f.probe.auto_release_loans = true;
    f.probe.hold_next_loan = true;
  }
  require(gst_pad_chain(f.pads[0], make_buffer(41)) == GST_FLOW_OK,
          "credit fixture rejected first frame");
  wait_for_outputs(&f, 1U);
  require(gst_pad_chain(f.pads[0], make_buffer(42)) == GST_FLOW_OK,
          "credit fixture rejected pending frame");
  require(gst_pad_chain(f.pads[0], make_buffer(43)) == GST_FLOW_OK,
          "credit fixture rejected latest replacement frame");
  {
    std::unique_lock<std::mutex> lock(f.probe.mutex);
    require(!f.probe.cv.wait_for(lock, std::chrono::milliseconds(100),
                                 [&] { return f.probe.outputs.size() >= 2U; }),
            "terminal-credit saturation must keep the latest frame pending");
  }
  GstBuffer* held = nullptr;
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    held = std::exchange(f.probe.held_loan, nullptr);
  }
  require(held != nullptr, "credit fixture did not retain its first loan");
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              held, f.probe.mux_namespace),
          "terminal release did not return mux credit");
  gst_buffer_unref(held);
  wait_for_outputs(&f, 2U);
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    require(f.probe.outputs[1].frame_id == 43,
            "credit release must emit the latest replacement instead of a stale pending frame");
  }
  stop_fixture(&f);
}

void test_total_credit_caps_all_streams() {
  Fixture f;
  init_fixture(&f, "latest-mux-total-credit", "2,2", 2);
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    f.probe.hold_all_loans = true;
  }

  require(gst_pad_chain(f.pads[0], make_buffer(61)) == GST_FLOW_OK,
          "total-credit fixture rejected stream0 frame");
  wait_for_outputs(&f, 1U);
  require(gst_pad_chain(f.pads[1], make_buffer(71)) == GST_FLOW_OK,
          "total-credit fixture rejected stream1 frame");
  wait_for_outputs(&f, 2U);
  require(gst_pad_chain(f.pads[0], make_buffer(62)) == GST_FLOW_OK,
          "total-credit fixture rejected pending frame");
  {
    std::unique_lock<std::mutex> lock(f.probe.mutex);
    require(!f.probe.cv.wait_for(lock, std::chrono::milliseconds(150),
                                 [&] { return f.probe.outputs.size() >= 3U; }),
            "mux-wide max-inflight-total=2 must hold a third loan even when its stream has "
            "credit");
  }

  GstBuffer* terminal = nullptr;
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    require(!f.probe.held_loans.empty(), "total-credit fixture did not retain a loan");
    terminal = f.probe.held_loans.front();
    f.probe.held_loans.erase(f.probe.held_loans.begin());
  }
  require(simaai::neat::pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(
              terminal, f.probe.mux_namespace),
          "terminal completion did not return mux-wide credit");
  gst_buffer_unref(terminal);
  wait_for_outputs(&f, 3U);
  stop_fixture(&f);
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    test_default_mode_keeps_latest();
    test_unconfigured_direct_mux_does_not_take_terminal_loans();
    test_credit_release_emits_latest_pending_frame();
    test_total_credit_caps_all_streams();
    std::cout << "[OK] unit_latest_mux_backpressure_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}

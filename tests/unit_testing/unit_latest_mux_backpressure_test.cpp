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

void init_fixture(Fixture* fixture, const char* name, bool block_when_pending,
                  const char* inflight_limits = "0,0", int max_inflight_total = 0) {
  require(fixture != nullptr, "latest-mux backpressure fixture pointer is null");
  Fixture& f = *fixture;
  f.pipeline = gst_pipeline_new(name);
  f.mux = gst_element_factory_make("neatlatestbystreammux", nullptr);
  f.sink = gst_element_factory_make("fakesink", nullptr);
  require(f.pipeline && f.mux && f.sink, "failed to create latest-mux backpressure fixture");
  g_object_set(f.mux, "stream-ids", "stream0,stream1", "max-inflight-total", max_inflight_total,
               "block-when-pending", block_when_pending ? TRUE : FALSE, nullptr);
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
  init_fixture(&f, "latest-mux-default-latest", false);
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
  init_fixture(&f, "latest-mux-direct-no-loans", false, nullptr);
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

void test_blocking_mode_preserves_burst() {
  Fixture f;
  init_fixture(&f, "latest-mux-every-frame", true);
  arm_blocking_probe(&f);
  require(gst_pad_chain(f.pads[0], make_buffer(11)) == GST_FLOW_OK,
          "blocking mode rejected first pending frame");
  GstFlowReturn second_flow = GST_FLOW_ERROR;
  bool second_done = false;
  std::mutex done_mutex;
  std::condition_variable done_cv;
  std::thread second([&] {
    second_flow = gst_pad_chain(f.pads[0], make_buffer(12));
    {
      std::lock_guard<std::mutex> lock(done_mutex);
      second_done = true;
    }
    done_cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(!done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return second_done; }),
            "blocking mode must backpressure a producer whose pending slot is occupied");
  }
  unblock_probe(&f);
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return second_done; }),
            "blocked producer did not resume after the worker consumed its pending frame");
  }
  second.join();
  require(second_flow == GST_FLOW_OK, "resumed producer should enqueue its frame successfully");
  wait_for_outputs(&f, 2U);
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    require(f.probe.outputs[0].frame_id == 11 && f.probe.outputs[1].frame_id == 12,
            "blocking mode must emit every burst frame in order");
  }
  stop_fixture(&f);
}

void test_credit_release_unblocks_pending_producer() {
  Fixture f;
  init_fixture(&f, "latest-mux-blocked-credit", true, "1,1");
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
  GstFlowReturn third_flow = GST_FLOW_ERROR;
  bool third_done = false;
  std::mutex done_mutex;
  std::condition_variable done_cv;
  std::thread third([&] {
    third_flow = gst_pad_chain(f.pads[0], make_buffer(43));
    {
      std::lock_guard<std::mutex> lock(done_mutex);
      third_done = true;
    }
    done_cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(!done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return third_done; }),
            "terminal-credit saturation should keep the producer blocked");
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
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return third_done; }),
            "credit release did not unblock the pending producer");
  }
  third.join();
  require(third_flow == GST_FLOW_OK, "credit-unblocked producer should return GST_FLOW_OK");
  wait_for_outputs(&f, 3U);
  stop_fixture(&f);
}

void test_total_credit_caps_all_streams() {
  Fixture f;
  init_fixture(&f, "latest-mux-total-credit", true, "2,2", 2);
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

void test_flush_start_wakes_blocked_producer() {
  Fixture f;
  init_fixture(&f, "latest-mux-blocked-flush", true);
  arm_blocking_probe(&f);
  require(gst_pad_chain(f.pads[0], make_buffer(51)) == GST_FLOW_OK,
          "flush fixture rejected pending frame");
  GstFlowReturn blocked_flow = GST_FLOW_OK;
  bool blocked_done = false;
  std::mutex done_mutex;
  std::condition_variable done_cv;
  std::thread producer([&] {
    blocked_flow = gst_pad_chain(f.pads[0], make_buffer(52));
    {
      std::lock_guard<std::mutex> lock(done_mutex);
      blocked_done = true;
    }
    done_cv.notify_all();
  });
  require(gst_pad_send_event(f.pads[0], gst_event_new_flush_start()) == TRUE,
          "failed to send blocking mux FLUSH_START");
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return blocked_done; }),
            "FLUSH_START did not wake the blocked mux producer");
  }
  require(blocked_flow == GST_FLOW_FLUSHING, "flushed producer must return GST_FLOW_FLUSHING");
  producer.join();
  unblock_probe(&f);
  require(gst_pad_send_event(f.pads[0], gst_event_new_flush_stop(FALSE)) == TRUE,
          "failed to send blocking mux FLUSH_STOP");
  stop_fixture(&f);
}

void test_multistream_every_frame_order_and_eos() {
  Fixture f;
  init_fixture(&f, "latest-mux-multistream-every-frame", true);
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    f.probe.output_delay = std::chrono::milliseconds(1);
  }
  constexpr int kFrames = 40;
  GstFlowReturn first_flow = GST_FLOW_OK;
  GstFlowReturn second_flow = GST_FLOW_OK;
  std::thread first([&] {
    for (int i = 0; i < kFrames; ++i) {
      first_flow = gst_pad_chain(f.pads[0], make_buffer(100 + i));
      if (first_flow != GST_FLOW_OK) {
        return;
      }
    }
  });
  std::thread second([&] {
    for (int i = 0; i < kFrames; ++i) {
      second_flow = gst_pad_chain(f.pads[1], make_buffer(200 + i));
      if (second_flow != GST_FLOW_OK) {
        return;
      }
    }
  });
  first.join();
  second.join();
  require(first_flow == GST_FLOW_OK && second_flow == GST_FLOW_OK,
          "multistream every-frame producer failed");
  require(gst_pad_send_event(f.pads[0], gst_event_new_eos()) == TRUE, "failed to send stream0 EOS");
  require(gst_pad_send_event(f.pads[1], gst_event_new_eos()) == TRUE, "failed to send stream1 EOS");
  wait_for_outputs(&f, 2U * kFrames);
  std::vector<std::int64_t> stream0;
  std::vector<std::int64_t> stream1;
  {
    std::lock_guard<std::mutex> lock(f.probe.mutex);
    for (const auto& output : f.probe.outputs) {
      (output.stream_id == "stream0" ? stream0 : stream1).push_back(output.frame_id);
    }
  }
  require(stream0.size() == kFrames && stream1.size() == kFrames,
          "every-frame multistream mode must emit every input before EOS");
  for (int i = 0; i < kFrames; ++i) {
    require(stream0[i] == 100 + i && stream1[i] == 200 + i,
            "every-frame multistream mode must preserve per-stream order");
  }
  stop_fixture(&f);
}

void test_state_stop_wakes_blocked_producer() {
  Fixture f;
  init_fixture(&f, "latest-mux-blocked-stop", true);
  arm_blocking_probe(&f);
  require(gst_pad_chain(f.pads[0], make_buffer(21)) == GST_FLOW_OK,
          "blocked-stop fixture rejected pending frame");
  GstFlowReturn blocked_flow = GST_FLOW_OK;
  bool blocked_done = false;
  std::mutex done_mutex;
  std::condition_variable done_cv;
  std::thread producer([&] {
    blocked_flow = gst_pad_chain(f.pads[0], make_buffer(22));
    {
      std::lock_guard<std::mutex> lock(done_mutex);
      blocked_done = true;
    }
    done_cv.notify_all();
  });
  GstStateChangeReturn stop_result = GST_STATE_CHANGE_FAILURE;
  std::thread stop([&] { stop_result = gst_element_set_state(f.pipeline, GST_STATE_NULL); });
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return blocked_done; }),
            "state stop did not wake the blocked mux producer");
  }
  require(blocked_flow == GST_FLOW_FLUSHING,
          "state-stopped producer must return GST_FLOW_FLUSHING");
  unblock_probe(&f);
  producer.join();
  stop.join();
  require(stop_result != GST_STATE_CHANGE_FAILURE, "blocking mux state stop failed");
  for (GstPad*& pad : f.pads) {
    gst_element_release_request_pad(f.mux, pad);
    gst_object_unref(pad);
    pad = nullptr;
  }
  gst_object_unref(f.pipeline);
  f.pipeline = nullptr;
}

void test_release_pad_wakes_blocked_producer() {
  Fixture f;
  init_fixture(&f, "latest-mux-blocked-release-pad", true);
  arm_blocking_probe(&f);
  require(gst_pad_chain(f.pads[0], make_buffer(31)) == GST_FLOW_OK,
          "release-pad fixture rejected pending frame");
  GstFlowReturn blocked_flow = GST_FLOW_OK;
  bool blocked_done = false;
  std::mutex done_mutex;
  std::condition_variable done_cv;
  std::thread producer([&] {
    blocked_flow = gst_pad_chain(f.pads[0], make_buffer(32));
    {
      std::lock_guard<std::mutex> lock(done_mutex);
      blocked_done = true;
    }
    done_cv.notify_all();
  });
  GstPad* released_pad = f.pads[0];
  std::thread release([&] { gst_element_release_request_pad(f.mux, released_pad); });
  {
    std::unique_lock<std::mutex> lock(done_mutex);
    require(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return blocked_done; }),
            "request-pad release did not wake the blocked mux producer");
  }
  require(blocked_flow == GST_FLOW_FLUSHING, "released-pad producer must return GST_FLOW_FLUSHING");
  producer.join();
  release.join();
  gst_object_unref(released_pad);
  f.pads[0] = nullptr;
  unblock_probe(&f);
  stop_fixture(&f);
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    test_default_mode_keeps_latest();
    test_unconfigured_direct_mux_does_not_take_terminal_loans();
    test_blocking_mode_preserves_burst();
    test_credit_release_unblocks_pending_producer();
    test_total_credit_caps_all_streams();
    test_flush_start_wakes_blocked_producer();
    test_multistream_every_frame_order_and_eos();
    test_state_stop_wakes_blocked_producer();
    test_release_pad_wakes_blocked_producer();
    std::cout << "[OK] unit_latest_mux_backpressure_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}

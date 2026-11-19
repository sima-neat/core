#include "pipeline/internal/RtspReconnect.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/GstErrorNormalizer.h"

#include <gst/gst.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace simaai::neat::pipeline_internal {
namespace {
using Clock = std::chrono::steady_clock;

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}

GQuark controller_key() {
  static const GQuark key = g_quark_from_static_string("neat-rtsp-reconnect");
  return key;
}

class RtspReconnectController {
  struct Source {
    RtspReconnectSource settings;
    GstElement* bin = nullptr;
    GstElement* source = nullptr;
    GstBus* message_bus = nullptr;
    GstPad* boundary = nullptr;
    GstPad* rtp_sink = nullptr;
    GstPad* output = nullptr;
    gulong activity_probe = 0;
    gulong boundary_probe = 0;
    gulong pad_added = 0;
    std::atomic<std::int64_t> last_data{0};
    std::atomic<bool> awaiting_media{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> resetting{false};
    std::atomic<bool> exhausted{false};
    std::atomic<bool> discontinuity{false};
    Clock::time_point next_attempt{};
    bool outage = false;
    std::optional<PullError> close_error;
    void notify(SourceStatus status) const {
      if (settings.events)
        settings.events->push({settings.label, status});
    }
    void interrupt_encoded_push(bool interrupted) const {
      if (settings.interrupt_encoded_push)
        settings.interrupt_encoded_push->store(interrupted);
    }
    int retries = 0;
    std::string reason;
    RtspReconnectController* owner = nullptr;

    ~Source() {
      if (message_bus) {
        gst_bus_set_sync_handler(message_bus, forward_bin_message, bin, nullptr);
        gst_object_unref(message_bus);
      }
      if (boundary) {
        if (boundary_probe)
          gst_pad_remove_probe(boundary, boundary_probe);
        gst_object_unref(boundary);
      }
      if (rtp_sink) {
        if (activity_probe)
          gst_pad_remove_probe(rtp_sink, activity_probe);
        gst_object_unref(rtp_sink);
      }
      if (output)
        gst_object_unref(output);
      if (source) {
        if (pad_added)
          g_signal_handler_disconnect(source, pad_added);
        gst_object_unref(source);
      }
      if (bin)
        gst_object_unref(bin);
    }
  };

public:
  RtspReconnectController(GstElement* pipeline, const std::vector<RtspReconnectSource>& settings) {
    for (const auto& spec : settings) {
      auto slot = std::make_unique<Source>();
      slot->settings = spec;
      slot->owner = this;
      slot->bin = gst_bin_get_by_name(GST_BIN(pipeline), spec.bin_name.c_str());
      slot->source = gst_bin_get_by_name(GST_BIN(pipeline), spec.source_name.c_str());
      GstElement* boundary = gst_bin_get_by_name(GST_BIN(pipeline), spec.boundary_name.c_str());
      if (boundary) {
        slot->boundary = gst_element_get_static_pad(boundary, "src");
        gst_object_unref(boundary);
      }
      if (!slot->bin || !slot->source || !slot->boundary)
        throw std::runtime_error("RTSP reconnect: source boundary is missing for " +
                                 spec.source_name);
      GstIterator* outputs = gst_element_iterate_src_pads(slot->bin);
      GValue pad = G_VALUE_INIT;
      if (gst_iterator_next(outputs, &pad) == GST_ITERATOR_OK) {
        slot->output = GST_PAD(g_value_dup_object(&pad));
        g_value_unset(&pad);
      }
      gst_iterator_free(outputs);
      if (!slot->output)
        throw std::runtime_error("RTSP reconnect: source bin has no output");
      // Async scheduling can insert a queue immediately after rtspsrc.
      // Its first dynamic RTP pad will link to this exact entry on each restart.
      slot->rtp_sink = gst_bin_find_unlinked_pad(GST_BIN(slot->bin), GST_PAD_SINK);
      if (!slot->rtp_sink)
        throw std::runtime_error("RTSP reconnect: RTP input boundary is missing");
      if (spec.interval_sec > 0) {
        slot->activity_probe = gst_pad_add_probe(
            slot->rtp_sink,
            static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER |
                                         GST_PAD_PROBE_TYPE_BUFFER_LIST),
            +[](GstPad*, GstPadProbeInfo*, gpointer data) {
              static_cast<Source*>(data)->last_data.store(now_ns(), std::memory_order_relaxed);
              return GST_PAD_PROBE_OK;
            },
            slot.get(), nullptr);
      }
      auto probe_types = GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM | GST_PAD_PROBE_TYPE_EVENT_FLUSH;
      if (spec.interval_sec > 0 && spec.attempts != 0)
        probe_types |= GST_PAD_PROBE_TYPE_BUFFER;
      slot->boundary_probe =
          gst_pad_add_probe(slot->boundary, static_cast<GstPadProbeType>(probe_types),
                            boundary_probe, slot.get(), nullptr);
      slot->pad_added =
          g_signal_connect(slot->source, "pad-added", G_CALLBACK(source_pad_added), slot.get());
      // Intercept connection errors before GstBin aborts its state transition.
      // Other messages still go through the ordinary bin's message handler.
      slot->message_bus = gst_element_get_bus(slot->source);
      gst_bus_set_sync_handler(slot->message_bus, source_message, slot.get(), nullptr);
      sources_.push_back(std::move(slot));
    }
  }

  ~RtspReconnectController() {
    stop();
  }

  void start() {
    std::lock_guard lock(mutex_);
    if (started_ || stopping_.load())
      return;
    started_ = true;
    const auto now = now_ns();
    for (const auto& slot : sources_) {
      slot->last_data.store(now);
      if (slot->settings.interval_sec || slot->failed.load())
        ensure_worker();
    }
  }

  void stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    {
      std::lock_guard state_lock(mutex_);
      stopping_.store(true);
      for (const auto& slot : sources_)
        slot->interrupt_encoded_push(true);
    }
    cv_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }

  std::optional<PullError> close_detail() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Fused branches close their own outputs. A pipeline-level EOS belongs
    // to all producers; never attribute it to one lost peer.
    if (sources_.size() == 1)
      return sources_.front()->close_error;
    return std::nullopt;
  }

private:
  // Caller holds mutex_. Healthy sources with retries disabled need no timer thread.
  void ensure_worker() {
    if (!worker_.joinable())
      worker_ = std::thread([this] { run(); });
  }

  static void source_pad_added(GstElement*, GstPad* pad, gpointer data) {
    auto& slot = *static_cast<Source*>(data);
    // gst_parse's delayed link may be gone after a failed first connection.
    if (!gst_pad_is_linked(pad) && !gst_pad_is_linked(slot.rtp_sink))
      gst_pad_link(pad, slot.rtp_sink);
  }

  static GstPadProbeReturn boundary_probe(GstPad*, GstPadProbeInfo* info, gpointer data) {
    auto& slot = *static_cast<Source*>(data);
    if (slot.owner->stopping_.load() || slot.exhausted.load())
      return GST_PAD_PROBE_OK;
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
      if (slot.resetting.load())
        return GST_PAD_PROBE_DROP;
      if (slot.discontinuity.exchange(false)) {
        GstBuffer* buffer = gst_buffer_make_writable(GST_PAD_PROBE_INFO_BUFFER(info));
        GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DISCONT);
        GST_PAD_PROBE_INFO_DATA(info) = buffer;
      }
      if (slot.awaiting_media.exchange(false)) {
        std::lock_guard lock(slot.owner->mutex_);
        if (!slot.failed.load() && !slot.exhausted.load()) {
          slot.outage = false;
          slot.retries = 0;
          slot.next_attempt = {};
          slot.notify(SourceStatus::Recovered);
        }
      }
      return GST_PAD_PROBE_OK;
    }
    GstEvent* event = GST_PAD_PROBE_INFO_EVENT(info);
    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
      if (!slot.resetting.load())
        slot.owner->failure(slot, "source reached end of stream");
      return GST_PAD_PROBE_DROP;
    case GST_EVENT_FLUSH_START:
    case GST_EVENT_FLUSH_STOP:
      return slot.resetting.load() ? GST_PAD_PROBE_DROP : GST_PAD_PROBE_OK;
    default:
      return GST_PAD_PROBE_OK;
    }
  }

  static GstBusSyncReply forward_bin_message(GstBus*, GstMessage* message, gpointer data) {
    auto* bin = GST_BIN(data);
    GST_BIN_GET_CLASS(bin)->handle_message(bin, message);
    return GST_BUS_DROP;
  }

  static bool is_connection_error(int code) {
    return code == GST_RESOURCE_ERROR_READ || code == GST_RESOURCE_ERROR_WRITE ||
           code == GST_RESOURCE_ERROR_OPEN_READ || code == GST_RESOURCE_ERROR_OPEN_READ_WRITE ||
           code == GST_RESOURCE_ERROR_NOT_FOUND;
  }

  static GstBusSyncReply source_message(GstBus* bus, GstMessage* message, gpointer data) {
    auto& slot = *static_cast<Source*>(data);
    auto& owner = *slot.owner;
    const auto type = GST_MESSAGE_TYPE(message);
    if (!owner.stopping_.load() && (type == GST_MESSAGE_ERROR || type == GST_MESSAGE_WARNING) &&
        GST_MESSAGE_SRC(message) == GST_OBJECT(slot.source)) {
      GError* error = nullptr;
      gchar* debug = nullptr;
      if (type == GST_MESSAGE_ERROR)
        gst_message_parse_error(message, &error, nullptr);
      else
        gst_message_parse_warning(message, &error, &debug);
      const bool server_closed = debug && g_strstr_len(debug, -1, "server closed the connection");
      g_free(debug);
      // rtspsrc reports a server EOF as RESOURCE/READ warning. Other
      // warnings (including UDP transport fallback) remain diagnostic.
      if (error && error->domain == GST_RESOURCE_ERROR &&
          ((server_closed && error->code == GST_RESOURCE_ERROR_READ) ||
           (type == GST_MESSAGE_ERROR && is_connection_error(error->code)))) {
        const auto reason = sanitize_gst_diagnostic_text(error->message);
        g_error_free(error);
        owner.failure(slot, reason);
        gst_message_unref(message);
        return GST_BUS_DROP;
      }
      if (error)
        g_error_free(error);
    }
    return forward_bin_message(bus, message, slot.bin);
  }

  void failure(Source& source, std::string reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_.load() || source.exhausted.load() || source.resetting.load() ||
        source.failed.exchange(true))
      return;
    source.reason = std::move(reason);
    if (started_)
      ensure_worker();
    ++generation_;
    cv_.notify_one();
  }

  void run() {
    std::unique_lock lock(mutex_);
    while (!stopping_.load()) {
      const auto generation = generation_;
      auto next = Clock::time_point::max();
      for (const auto& entry : sources_) {
        auto& slot = *entry;
        if (slot.exhausted.load())
          continue;
        const auto interval = std::chrono::seconds(slot.settings.interval_sec);
        const auto now = Clock::now();
        auto due =
            interval.count()
                ? Clock::time_point(std::chrono::nanoseconds(slot.last_data.load())) + interval
                : Clock::time_point::max();
        if (slot.failed.load())
          due = slot.next_attempt;
        if (now < due) {
          next = std::min(next, due);
          continue;
        }
        if (!slot.failed.load())
          slot.reason = "no media received";
        const bool terminal = interval.count() == 0 || slot.retries == slot.settings.attempts;
        if (terminal) {
          slot.close_error = PullError{};
          slot.close_error->source = slot.settings.label;
          slot.close_error->code = error_codes::kRtspConnectionFailed;
          slot.close_error->message = "RTSP source " + slot.settings.label + ": " + slot.reason;
          slot.exhausted.store(true);
          slot.notify(SourceStatus::Disconnected);
        } else {
          if (!slot.outage)
            slot.notify(SourceStatus::Reconnecting);
          slot.outage = true;
          slot.awaiting_media.store(false);
          slot.next_attempt = now + interval;
          ++slot.retries;
          next = std::min(next, slot.next_attempt);
        }
        // The source state change joins its streaming task. Let a blocked
        // EveryFrame tap leave the queue before waiting for that task.
        slot.interrupt_encoded_push(true);
        slot.resetting.store(true);
        slot.failed.store(false);
        lock.unlock(); // GStreamer can synchronously deliver messages during state changes.
        reset(slot, terminal);
        lock.lock();
        if (stopping_.load())
          return;
      }
      if (!stopping_.load())
        cv_.wait_until(lock, next, [&] { return stopping_.load() || generation_ != generation; });
    }
  }

  void reset(Source& slot, bool terminal) {
    gst_element_set_locked_state(slot.bin, TRUE);
    gst_element_set_state(slot.bin, GST_STATE_NULL);
    if (stopping_.load())
      return;
    if (terminal) {
      if (slot.settings.on_exhausted)
        slot.settings.on_exhausted(*slot.close_error);
      GstPad* peer = gst_pad_get_peer(slot.output);
      if (peer) {
        // A sink that never prerolled can block EOS until teardown. Keep the
        // recovery worker free so stop() can cancel and join it first.
        GstElement* downstream = gst_pad_get_parent_element(peer);
        if (downstream) {
          gst_element_call_async(
              downstream,
              +[](GstElement*, gpointer pad) {
                gst_pad_send_event(GST_PAD(pad), gst_event_new_eos());
              },
              gst_object_ref(peer), +[](gpointer pad) { gst_object_unref(pad); });
          gst_object_unref(downstream);
        }
        gst_object_unref(peer);
      }
      return;
    }
    {
      std::lock_guard lock(mutex_);
      if (stopping_.load())
        return;
      slot.interrupt_encoded_push(false);
    }
    slot.last_data.store(now_ns());
    slot.awaiting_media.store(true);
    slot.discontinuity.store(true);
    slot.resetting.store(false);
    gst_element_set_state(slot.bin, GST_STATE_PLAYING);
    gst_element_set_locked_state(slot.bin, FALSE);
  }

  std::vector<std::unique_ptr<Source>> sources_;
  std::atomic<bool> stopping_{false};
  std::mutex mutex_;

  bool started_ = false;
  unsigned generation_ = 0;
  std::mutex stop_mutex_;
  std::condition_variable cv_;
  std::thread worker_;
};

RtspReconnectController* controller(GstElement* pipeline) {
  return pipeline ? static_cast<RtspReconnectController*>(
                        g_object_get_qdata(G_OBJECT(pipeline), controller_key()))
                  : nullptr;
}
} // namespace

void attach_rtsp_reconnect(GstElement* pipeline, const std::vector<RtspReconnectSource>& sources) {
  if (sources.empty())
    return;
  auto owner = std::make_unique<RtspReconnectController>(pipeline, sources);
  g_object_set_qdata_full(
      G_OBJECT(pipeline), controller_key(), owner.release(),
      +[](gpointer data) { delete static_cast<RtspReconnectController*>(data); });
}

bool start_rtsp_reconnect(GstElement* pipeline) {
  if (auto* owner = controller(pipeline)) {
    owner->start();
    return true;
  }
  return false;
}

void stop_rtsp_reconnect(GstElement* pipeline) {
  if (auto* owner = controller(pipeline))
    owner->stop();
}

std::optional<PullError> rtsp_reconnect_close_detail(GstElement* pipeline) {
  if (auto* owner = controller(pipeline))
    return owner->close_detail();
  return std::nullopt;
}
} // namespace simaai::neat::pipeline_internal

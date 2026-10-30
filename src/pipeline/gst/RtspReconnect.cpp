#include "pipeline/internal/RtspReconnect.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/Diagnostics.h"
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
    gulong boundary_probe = 0;
    gulong pad_added = 0;
    std::atomic<std::int64_t> last_data{0};
    std::atomic<std::uint64_t> frames{0};
    std::atomic<bool> failed{false};
    std::atomic<bool> resetting{false};
    std::atomic<bool> exhausted{false};
    std::atomic<bool> discontinuity{false};
    std::uint64_t observed_frames = 0;
    int retries = 0;
    std::string reason;
    RtspReconnectController* owner = nullptr;

    ~Source() {
      if (message_bus) {
        gst_bus_set_sync_handler(message_bus, forward_bin_message, bin, nullptr);
        gst_object_unref(message_bus);
      }
      if (boundary) {
        gst_pad_remove_probe(boundary, boundary_probe);
        gst_object_unref(boundary);
      }
      if (rtp_sink)
        gst_object_unref(rtp_sink);
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
  RtspReconnectController(GstElement* pipeline, const std::vector<RtspReconnectSource>& settings,
                          std::shared_ptr<DiagCtx> diag)
      : diag_(std::move(diag)) {
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
      GstIterator* elements = gst_bin_iterate_elements(GST_BIN(slot->bin));
      GValue element = G_VALUE_INIT;
      while (gst_iterator_next(elements, &element) == GST_ITERATOR_OK) {
        GstPad* sink =
            gst_element_get_static_pad(GST_ELEMENT(g_value_get_object(&element)), "sink");
        if (sink && !gst_pad_is_linked(sink)) {
          slot->rtp_sink = sink;
          g_value_unset(&element);
          break;
        }
        if (sink)
          gst_object_unref(sink);
        g_value_reset(&element);
      }
      gst_iterator_free(elements);
      if (!slot->rtp_sink)
        throw std::runtime_error("RTSP reconnect: RTP input boundary is missing");
      slot->boundary_probe =
          gst_pad_add_probe(slot->boundary,
                            static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER |
                                                         GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM |
                                                         GST_PAD_PROBE_TYPE_EVENT_FLUSH),
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
    if (worker_.joinable())
      return;
    const auto now = now_ns();
    for (const auto& slot : sources_)
      slot->last_data.store(now);
    worker_ = std::thread([this] { run(); });
  }

  void stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    stopping_.store(true);
    cv_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }

  std::optional<PullError> close_detail() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& slot : sources_) {
      if (slot->exhausted.load()) {
        PullError result;
        result.code = error_codes::kRtspConnectionFailed;
        result.message = "RTSP source " + slot->settings.source_name + " exhausted " +
                         std::to_string(slot->retries) + " reconnect attempts: " + slot->reason;
        return result;
      }
    }
    return std::nullopt;
  }

private:
  static void source_pad_added(GstElement*, GstPad* pad, gpointer data) {
    auto& slot = *static_cast<Source*>(data);
    // gst_parse's delayed link may be gone after a failed first connection.
    if (!gst_pad_is_linked(pad) && !gst_pad_is_linked(slot.rtp_sink))
      gst_pad_link(pad, slot.rtp_sink);
    gst_pad_add_probe(
        pad, GST_PAD_PROBE_TYPE_BUFFER,
        +[](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
          static_cast<Source*>(data)->last_data.store(now_ns(), std::memory_order_relaxed);
          return GST_PAD_PROBE_OK;
        },
        data, nullptr);
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
      slot.frames.fetch_add(1, std::memory_order_relaxed);
      return GST_PAD_PROBE_OK;
    }
    GstEvent* event = GST_PAD_PROBE_INFO_EVENT(info);
    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
      slot.owner->failure(slot, "source reached end of stream");
      return GST_PAD_PROBE_DROP;
    case GST_EVENT_FLUSH_START:
    case GST_EVENT_FLUSH_STOP:
      return GST_PAD_PROBE_DROP;
    default:
      return GST_PAD_PROBE_OK;
    }
  }

  static GstBusSyncReply forward_bin_message(GstBus*, GstMessage* message, gpointer data) {
    auto* bin = GST_BIN(data);
    GST_BIN_GET_CLASS(bin)->handle_message(bin, message);
    return GST_BUS_DROP;
  }

  static GstBusSyncReply source_message(GstBus* bus, GstMessage* message, gpointer data) {
    auto& slot = *static_cast<Source*>(data);
    auto& owner = *slot.owner;
    if (!owner.stopping_.load() && GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR &&
        GST_MESSAGE_SRC(message) == GST_OBJECT(slot.source)) {
      GError* error = nullptr;
      gst_message_parse_error(message, &error, nullptr);
      if (error && error->domain == GST_RESOURCE_ERROR) {
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
    if (!source.failed.exchange(true)) {
      source.reason = std::move(reason);
      if (diag_)
        diag_->push_bus("warning", source.settings.source_name,
                        "RTSP reconnect pending: " + source.reason);
    }
  }

  void run() {
    while (!stopping_.load()) {
      auto next = Clock::time_point::max();
      for (const auto& slot : sources_) {
        if (stopping_.load())
          return;
        if (slot->exhausted.load())
          continue;
        const auto frames = slot->frames.load();
        if (frames != slot->observed_frames) {
          slot->observed_frames = frames;
          if (slot->retries > 0 && diag_)
            diag_->push_bus("info", slot->settings.source_name, "RTSP source reconnected");
          slot->retries = 0;
        }
        const auto interval = std::chrono::milliseconds(slot->settings.timeout_ms);
        const auto deadline =
            Clock::time_point(std::chrono::nanoseconds(slot->last_data.load())) + interval;
        if (Clock::now() >= deadline) {
          retry(*slot);
          next = std::min(next, Clock::now() + interval);
        } else {
          next = std::min(next, deadline);
        }
      }
      if (next == Clock::time_point::max())
        return;
      std::unique_lock<std::mutex> lock(wait_mutex_);
      cv_.wait_until(lock, next, [this] { return stopping_.load(); });
    }
  }

  void retry(Source& slot) {
    slot.resetting.store(true);
    gst_element_set_locked_state(slot.bin, TRUE);
    gst_element_set_state(slot.bin, GST_STATE_NULL);
    if (stopping_.load())
      return;
    if (slot.settings.attempts > 0 && slot.retries >= slot.settings.attempts) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot.reason.empty())
          slot.reason = "no data received";
        slot.exhausted.store(true);
      }
      if (diag_)
        diag_->push_bus("warning", slot.settings.source_name, "RTSP reconnect attempts exhausted");
      if (slot.settings.on_exhausted) {
        PullError detail;
        detail.code = error_codes::kRtspConnectionFailed;
        detail.message = "RTSP source " + slot.settings.source_name + " exhausted " +
                         std::to_string(slot.retries) + " reconnect attempts: " + slot.reason;
        slot.settings.on_exhausted(detail);
      }
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
    ++slot.retries;
    if (diag_)
      diag_->push_bus("warning", slot.settings.source_name,
                      "RTSP reconnect attempt " + std::to_string(slot.retries));
    slot.last_data.store(now_ns());
    slot.observed_frames = slot.frames.load();
    slot.failed.store(false);
    slot.discontinuity.store(true);
    slot.resetting.store(false);
    gst_element_set_state(slot.bin, GST_STATE_PLAYING);
    gst_element_set_locked_state(slot.bin, FALSE);
  }

  std::shared_ptr<DiagCtx> diag_;
  std::vector<std::unique_ptr<Source>> sources_;
  std::atomic<bool> stopping_{false};
  std::mutex mutex_;
  std::mutex wait_mutex_;
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

void attach_rtsp_reconnect(GstElement* pipeline, const std::vector<RtspReconnectSource>& sources,
                           const std::shared_ptr<DiagCtx>& diag) {
  if (sources.empty())
    return;
  auto owner = std::make_unique<RtspReconnectController>(pipeline, sources, diag);
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

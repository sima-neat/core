#include "pipeline/internal/InputStreamWatchdog.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

using Element = std::shared_ptr<GstElement>;
using Work = std::shared_ptr<InputStreamWork>;

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void observe_pad(GstElement* element, const char* name, const Work& work, bool entering) {
  GstPad* pad = gst_element_get_static_pad(element, name);
  if (!pad)
    throw std::runtime_error("InputStream watchdog: missing observation pad");
  struct Observer {
    Work work;
    bool entering;
  };
  gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER,
      +[](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
        const auto& observer = *static_cast<Observer*>(data);
        if (observer.entering)
          observer.work->begin();
        else
          observer.work->complete();
        return GST_PAD_PROBE_OK;
      },
      new Observer{work, entering}, +[](gpointer data) { delete static_cast<Observer*>(data); });
  gst_object_unref(pad);
}

// InputStream owns one appsrc/appsink segment. Follow its linear data path;
// branching/aggregation remains the graph runtime's responsibility.
std::vector<Element> path_between(GstElement* source, GstElement* sink) {
  std::vector<Element> path;
  std::unordered_set<GstElement*> visited;
  Element current(static_cast<GstElement*>(gst_object_ref(source)), gst_object_unref);
  while (current && current.get() != sink && visited.insert(current.get()).second) {
    GstPad* output = gst_element_get_static_pad(current.get(), "src");
    if (!output)
      return {};
    GstPad* peer = gst_pad_get_peer(output);
    gst_object_unref(output);
    if (!peer)
      return {};
    Element next(gst_pad_get_parent_element(peer), gst_object_unref);
    gst_object_unref(peer);
    if (!next)
      return {};
    path.push_back(next);
    current = std::move(next);
  }
  return current.get() == sink ? path : std::vector<Element>{};
}

} // namespace

struct InputStreamWatchdog::QueueProgress {
  struct Activity {
    std::atomic<std::int64_t> last_progress_ns{now_ns()};

    void progress() {
      const auto now = now_ns();
      auto previous = last_progress_ns.load(std::memory_order_acquire);
      while (previous < now &&
             !last_progress_ns.compare_exchange_weak(previous, now, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
      }
    }
  };
  using Clock = std::shared_ptr<Activity>;
  Element queue;
  Clock activity = std::make_shared<Activity>();

  explicit QueueProgress(Element element) : queue(std::move(element)) {
    // Reset on progress or admission into an empty queue, never merely on
    // continued input into a full queue. Observe occupancy rather than trying
    // to infer a dropped-buffer count from GStreamer's overrun signal.
    for (const char* name : {"sink", "src"}) {
      GstPad* pad = gst_element_get_static_pad(queue.get(), name);
      gst_pad_add_probe(
          pad, GST_PAD_PROBE_TYPE_BUFFER,
          +[](GstPad* pad, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
            const auto& clock = *static_cast<Clock*>(data);
            if (GST_PAD_DIRECTION(pad) == GST_PAD_SRC) {
              clock->progress();
            } else if (auto* parent = gst_pad_get_parent_element(pad)) {
              guint level = 0;
              g_object_get(parent, "current-level-buffers", &level, nullptr);
              gst_object_unref(parent);
              if (level == 0)
                clock->progress();
            }
            return GST_PAD_PROBE_OK;
          },
          new Clock(activity), +[](gpointer data) { delete static_cast<Clock*>(data); });
      gst_object_unref(pad);
    }
  }

  guint pending() const {
    guint level = 0;
    g_object_get(queue.get(), "current-level-buffers", &level, nullptr);
    return level;
  }

  bool timed_out(int timeout_ms) {
    if (pending() == 0) {
      activity->progress();
      return false;
    }
    return now_ns() - activity->last_progress_ns.load(std::memory_order_acquire) >
           static_cast<std::int64_t>(timeout_ms) * 1000000;
  }
};

InputStreamWatchdog::InputStreamWatchdog() : spans_{std::make_shared<InputStreamWork>()} {}
InputStreamWatchdog::~InputStreamWatchdog() = default;

void InputStreamWatchdog::observe(GstElement* source, GstElement* sink) {
  for (const auto& element : path_between(source, sink)) {
    GstElementFactory* factory = gst_element_get_factory(element.get());
    if (!factory)
      continue;
    const char* kind = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
    const bool is_queue = std::strcmp(kind, "queue") == 0;
    bool boundary = std::strcmp(kind, "videorate") == 0 || std::strcmp(kind, "valve") == 0;
    if (is_queue) {
      gint leaky = 0;
      g_object_get(element.get(), "leaky", &leaky, nullptr);
      boundary = leaky != 0;
    }
    if (!boundary)
      continue;
    // The upstream span ends when the policy accepts the buffer; its output
    // starts a separate span. Continued upstream traffic cannot extend the
    // deadline of blocked downstream processing.
    observe_pad(element.get(), "sink", spans_.back(), false);
    spans_.push_back(std::make_shared<InputStreamWork>());
    observe_pad(element.get(), "src", spans_.back(), true);
    if (is_queue)
      queues_.push_back(std::make_unique<QueueProgress>(element));
  }
  observe_pad(sink, "sink", spans_.back(), false);
}

std::int64_t InputStreamWatchdog::pending() const {
  std::int64_t count = 0;
  for (const auto& span : spans_)
    count += span->pending();
  for (const auto& queue : queues_)
    count += queue->pending();
  return count;
}

bool InputStreamWatchdog::timed_out(int timeout_ms) {
  if (timeout_ms <= 0)
    return false;
  for (const auto& span : spans_)
    if (span->timed_out(timeout_ms))
      return true;
  for (const auto& queue : queues_)
    if (queue->timed_out(timeout_ms))
      return true;
  return false;
}

} // namespace simaai::neat::pipeline_internal

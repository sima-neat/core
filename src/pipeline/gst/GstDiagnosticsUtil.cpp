// src/pipeline/internal/GstDiagnosticsUtil.cpp
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/TensorMath.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"

#include <gst/app/gstappsink.h>
#include <gst/gstdebugutils.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

const char* state_name(GstState s) {
  switch (s) {
  case GST_STATE_VOID_PENDING:
    return "VOID_PENDING";
  case GST_STATE_NULL:
    return "NULL";
  case GST_STATE_READY:
    return "READY";
  case GST_STATE_PAUSED:
    return "PAUSED";
  case GST_STATE_PLAYING:
    return "PLAYING";
  default:
    return "UNKNOWN";
  }
}

void log_appsink_pull_state(GstElement* appsink, GstSample* sample, const char* tag) {
  if (!env_bool("SIMA_APPSINK_PULL_DEBUG", false))
    return;
  if (!appsink || !sample)
    return;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  const guint sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(sample);
  const guint buf_ref = buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(buf) : 0u;
  const gboolean buf_writable = buf ? gst_buffer_is_writable(buf) : FALSE;
  gboolean mem_readonly = FALSE;
  if (buf && gst_buffer_n_memory(buf) > 0) {
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    if (mem) {
      mem_readonly = GST_MEMORY_IS_READONLY(mem);
    }
  }
  const char* pool_type = (buf && buf->pool) ? G_OBJECT_TYPE_NAME(buf->pool) : "<none>";
  std::fprintf(stderr,
               "[APPSINK_PULL] %s sample=%p sample_ref=%u buffer=%p buf_ref=%u "
               "writable=%s mem_readonly=%s pool=%p type=%s\n",
               tag ? tag : "pull", static_cast<void*>(sample), sample_ref, static_cast<void*>(buf),
               buf_ref, buf_writable ? "true" : "false", mem_readonly ? "true" : "false",
               buf ? static_cast<void*>(buf->pool) : nullptr, pool_type ? pool_type : "<null>");
}

void log_appsink_last_sample(GstElement* appsink, const char* tag) {
  if (!env_bool("SIMA_APPSINK_LAST_SAMPLE_DEBUG", false))
    return;
  if (!appsink)
    return;
  GstSample* last = nullptr;
  GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(appsink), "last-sample");
  if (pspec) {
    g_object_get(appsink, "last-sample", &last, nullptr);
  }
  if (!last) {
    std::fprintf(stderr, "[APPSINK_LAST] %s last_sample=<null>\n", tag ? tag : "last");
    return;
  }
  GstBuffer* buf = gst_sample_get_buffer(last);
  const guint sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(last);
  const guint buf_ref = buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(buf) : 0u;
  const gboolean buf_writable = buf ? gst_buffer_is_writable(buf) : FALSE;
  std::fprintf(stderr, "[APPSINK_LAST] %s last_sample=%p ref=%u buffer=%p buf_ref=%u writable=%s\n",
               tag ? tag : "last", static_cast<void*>(last), sample_ref, static_cast<void*>(buf),
               buf_ref, buf_writable ? "true" : "false");
  gst_sample_unref(last);
}

int teardown_timeout_ms() {
  const int val = env_int("SIMA_GST_TEARDOWN_TIMEOUT_MS", 2000);
  return (val < 0) ? 0 : val;
}

int reaper_sleep_ms() {
  return std::max(10, env_int("SIMA_GST_TEARDOWN_REAPER_MS", 250));
}

struct TeardownReaperState {
  std::mutex mu;
  std::condition_variable cv;
  struct Pending {
    GstElement* pipeline = nullptr;
    bool flush = true;
  };
  std::vector<Pending> zombies;
  std::atomic<bool> started{false};
};

TeardownReaperState& teardown_reaper_state() {
  static auto* state = new TeardownReaperState();
  return *state;
}

void begin_teardown(GstElement* pipeline, bool flush) {
  if (!pipeline)
    return;
  if (flush) {
    gst_element_send_event(pipeline, gst_event_new_flush_start());
    gst_element_send_event(pipeline, gst_event_new_flush_stop(TRUE));
  }
  gst_element_set_state(pipeline, GST_STATE_NULL);
}

bool finish_teardown(GstElement* pipeline, int timeout_ms) {
  if (!pipeline)
    return true;
  GstState cur = GST_STATE_VOID_PENDING;
  GstState pend = GST_STATE_VOID_PENDING;
  const GstClockTime timeout = static_cast<GstClockTime>(timeout_ms) * GST_MSECOND;
  gst_element_get_state(pipeline, &cur, &pend, timeout);
  if (cur != GST_STATE_NULL)
    return false;
  gst_object_unref(pipeline);
  return true;
}

void reaper_main() {
  auto& state = teardown_reaper_state();
  while (true) {
    std::vector<TeardownReaperState::Pending> work;
    {
      std::unique_lock<std::mutex> lock(state.mu);
      if (state.zombies.empty()) {
        state.cv.wait_for(lock, std::chrono::milliseconds(reaper_sleep_ms()));
      }
      if (state.zombies.empty())
        continue;
      work.swap(state.zombies);
    }

    std::vector<TeardownReaperState::Pending> retry;
    const int timeout_ms = teardown_timeout_ms();
    for (const auto& item : work) {
      begin_teardown(item.pipeline, item.flush);
      if (!finish_teardown(item.pipeline, timeout_ms)) {
        retry.push_back(item);
      }
    }

    if (!retry.empty()) {
      std::lock_guard<std::mutex> lock(state.mu);
      state.zombies.insert(state.zombies.end(), retry.begin(), retry.end());
    }
  }
}

void ensure_reaper_thread() {
  auto& state = teardown_reaper_state();
  bool expected = false;
  if (!state.started.compare_exchange_strong(expected, true))
    return;
  std::thread(reaper_main).detach();
}

void enqueue_teardown(GstElement* pipeline, bool flush) {
  if (!pipeline)
    return;
  ensure_reaper_thread();
  auto& state = teardown_reaper_state();
  {
    std::lock_guard<std::mutex> lock(state.mu);
    state.zombies.push_back(TeardownReaperState::Pending{pipeline, flush});
  }
  state.cv.notify_one();
}

} // namespace

bool map_video_frame_read(SampleHolder& h, std::string& err) {
  err.clear();
  if (!h.sample) {
    err = "missing sample";
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(h.sample);
  if (!caps) {
    err = "missing caps";
    return false;
  }

  GstBuffer* buf = gst_sample_get_buffer(h.sample);
  if (!buf) {
    err = "missing buffer";
    return false;
  }

  if (!gst_video_info_from_caps(&h.vinfo, caps)) {
    err = "gst_video_info_from_caps failed";
    return false;
  }

  if (!gst_video_frame_map(&h.frame, &h.vinfo, buf, GST_MAP_READ)) {
    err = "gst_video_frame_map failed (non-mappable memory?)";
    return false;
  }

  h.mapped = true;
  return true;
}

std::string gst_caps_to_string_safe(GstCaps* caps) {
  if (!caps)
    return "<null caps>";
  gchar* s = gst_caps_to_string(caps);
  if (!s)
    return "<caps_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

std::string gst_structure_to_string_safe(const GstStructure* st) {
  if (!st)
    return "<null structure>";
  gchar* s = gst_structure_to_string(st);
  if (!s)
    return "<structure_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

std::string gst_message_to_string(GstMessage* msg) {
  if (!msg)
    return "<null message>";

  std::ostringstream ss;
  const GstMessageType t = GST_MESSAGE_TYPE(msg);
  ss << gst_message_type_get_name(t);

  if (t == GST_MESSAGE_ERROR) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_WARNING) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_warning(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_INFO) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_info(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg)
      ss << " | " << dbg;
    if (e)
      g_error_free(e);
    if (dbg)
      g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_STATE_CHANGED) {
    if (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg))) {
      ss << " src=" << GST_OBJECT_NAME(GST_MESSAGE_SRC(msg));
    }
    GstState old_s, new_s, pend_s;
    gst_message_parse_state_changed(msg, &old_s, &new_s, &pend_s);
    ss << " " << state_name(old_s) << " -> " << state_name(new_s) << " (pending "
       << state_name(pend_s) << ")";
    return ss.str();
  }
  if (t == GST_MESSAGE_EOS) {
    ss << " (EOS)";
    return ss.str();
  }
  if (t == GST_MESSAGE_ASYNC_DONE) {
    ss << " (ASYNC_DONE)";
    return ss.str();
  }
  if (t == GST_MESSAGE_STREAM_START) {
    ss << " (STREAM_START)";
    return ss.str();
  }

  const GstStructure* st = gst_message_get_structure(msg);
  if (st)
    ss << " " << gst_structure_to_string_safe(st);
  return ss.str();
}

std::string caps_features_string(GstCaps* caps) {
  if (!caps)
    return "<none>";
  GstCapsFeatures* f = gst_caps_get_features(caps, 0);
  if (!f)
    return "<none>";

#if GST_CHECK_VERSION(1, 16, 0)
  gchar* s = gst_caps_features_to_string(f);
  if (!s)
    return "<none>";
  std::string out = s;
  g_free(s);
  return out;
#else
  if (gst_caps_features_is_any(f))
    return "ANY";
  if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY))
    return "memory:SystemMemory";
  return "<features>";
#endif
}

void maybe_dump_dot(GstElement* pipeline, const std::string& tag) {
  if (!pipeline)
    return;

  const std::string dir = env_str("SIMA_GST_DOT_DIR", "");
  if (dir.empty())
    return;

  // Tell GStreamer where to dump dot graphs.
  g_setenv("GST_DEBUG_DUMP_DOT_DIR", dir.c_str(), TRUE);

  const std::string t = "sima_" + sanitize_name(tag);
  gst_debug_bin_to_dot_file_with_ts(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, t.c_str());
}

std::string boundary_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->boundaries.empty())
    return "";

  const int64_t now = (int64_t)g_get_monotonic_time();

  int best_idx = -1;
  int64_t best_t = 0;
  bool best_out = false;

  // Use snapshots to keep this readable and consistent.
  std::vector<BoundaryFlowStats> snaps;
  snaps.reserve(diag->boundaries.size());
  for (const auto& bp : diag->boundaries) {
    if (!bp)
      continue;
    snaps.push_back(bp->snapshot());
  }
  if (snaps.empty())
    return "";

  for (size_t i = 0; i < snaps.size(); ++i) {
    const auto& b = snaps[i];
    if (b.last_out_wall_us > best_t) {
      best_t = b.last_out_wall_us;
      best_idx = (int)i;
      best_out = true;
    }
    if (b.last_in_wall_us > best_t) {
      best_t = b.last_in_wall_us;
      best_idx = (int)i;
      best_out = false;
    }
  }

  std::ostringstream ss;
  ss << "BoundaryFlow:\n";
  for (const auto& b : snaps) {
    ss << "  - " << b.boundary_name << " after=" << b.after_node_index
       << " before=" << b.before_node_index << " in=" << b.in_buffers << " out=" << b.out_buffers
       << " last_in_age_ms=" << (b.last_in_wall_us ? ((now - b.last_in_wall_us) / 1000) : 0)
       << " last_out_age_ms=" << (b.last_out_wall_us ? ((now - b.last_out_wall_us) / 1000) : 0)
       << "\n";
  }

  if (best_idx >= 0 && best_t > 0) {
    const auto& b = snaps[(size_t)best_idx];
    ss << "LikelyStall: last activity " << (best_out ? "leaving " : "entering ") << b.boundary_name
       << " age_ms=" << ((now - best_t) / 1000) << " (after node " << b.after_node_index
       << ", before node " << b.before_node_index << ")\n";
  }

  return ss.str();
}

std::string stage_timing_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->stage_timings.empty())
    return "";

  std::vector<StageTimingStats> snaps;
  snaps.reserve(diag->stage_timings.size());
  for (const auto& s : diag->stage_timings) {
    if (!s)
      continue;
    snaps.push_back(s->snapshot());
  }
  if (snaps.empty())
    return "";

  std::sort(snaps.begin(), snaps.end(), [](const StageTimingStats& a, const StageTimingStats& b) {
    const double avg_a = (a.samples == 0) ? 0.0 : (double)a.total_us / (double)a.samples;
    const double avg_b = (b.samples == 0) ? 0.0 : (double)b.total_us / (double)b.samples;
    return avg_a > avg_b;
  });

  std::ostringstream ss;
  ss << "StageTiming:\n";
  for (const auto& s : snaps) {
    const double avg_ms = (s.samples == 0) ? 0.0 : (double)s.total_us / (double)s.samples / 1000.0;
    const double max_ms = (double)s.max_us / 1000.0;
    ss << "  - " << s.stage_name << " samples=" << s.samples << " avg_ms=" << avg_ms
       << " max_ms=" << max_ms << "\n";
  }
  return ss.str();
}

std::string element_timing_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->element_timings.empty())
    return "";

  std::vector<ElementTimingStats> snaps;
  snaps.reserve(diag->element_timings.size());
  for (const auto& s : diag->element_timings) {
    if (!s)
      continue;
    snaps.push_back(s->snapshot());
  }
  if (snaps.empty())
    return "";

  std::sort(snaps.begin(), snaps.end(),
            [](const ElementTimingStats& a, const ElementTimingStats& b) {
              const double avg_a = (a.samples == 0) ? 0.0 : (double)a.total_us / (double)a.samples;
              const double avg_b = (b.samples == 0) ? 0.0 : (double)b.total_us / (double)b.samples;
              return avg_a > avg_b;
            });

  std::ostringstream ss;
  ss << "ElementTiming (residency = sink-arrival -> src-emit, INCLUDES backpressure wait):\n";
  for (const auto& s : snaps) {
    const double avg_ms = (s.samples == 0) ? 0.0 : (double)s.total_us / (double)s.samples / 1000.0;
    const double max_ms = (double)s.max_us / 1000.0;
    const double min_ms = (double)s.min_us / 1000.0;
    ss << "  - " << s.element_name << " samples=" << s.samples << " avg_ms=" << avg_ms
       << " min_ms=" << min_ms << " max_ms=" << max_ms << " missed_in=" << s.missed_in
       << " missed_out=" << s.missed_out;
    ss << "\n";
  }
  return ss.str();
}

std::string element_flow_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->element_flows.empty())
    return "";

  std::vector<ElementFlowStats> snaps;
  snaps.reserve(diag->element_flows.size());
  for (const auto& s : diag->element_flows) {
    if (!s)
      continue;
    snaps.push_back(s->snapshot());
  }
  if (snaps.empty())
    return "";

  std::sort(snaps.begin(), snaps.end(), [](const ElementFlowStats& a, const ElementFlowStats& b) {
    return a.in_bytes > b.in_bytes;
  });

  std::ostringstream ss;
  ss << "ElementFlow:\n";
  for (const auto& s : snaps) {
    ss << "  - " << s.element_name << " in_buffers=" << s.in_buffers
       << " out_buffers=" << s.out_buffers << " in_bytes=" << s.in_bytes
       << " out_bytes=" << s.out_bytes << " caps_changes=" << s.caps_changes << "\n";
  }
  return ss.str();
}

void drain_bus(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag, const char* /*where*/,
               bool* eos_seen) {
  if (!pipeline)
    return;

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return;

  const GstMessageType mask = static_cast<GstMessageType>(GST_MESSAGE_ANY & ~GST_MESSAGE_ERROR);
  while (GstMessage* msg = gst_bus_pop_filtered(bus, mask)) {
    const GstMessageType t = GST_MESSAGE_TYPE(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                          ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                          : "<unknown>";

    std::string line = gst_message_to_string(msg);
    if (diag)
      diag->push_bus(gst_message_type_get_name(t), src ? src : "<unknown>", line);
    if (eos_seen && t == GST_MESSAGE_EOS) {
      *eos_seen = true;
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
}

void throw_if_bus_error(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                        const char* where) {
  if (!pipeline)
    return;

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return;

  GstMessage* msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
  if (!msg) {
    gst_object_unref(bus);
    return;
  }

  const GstMessageType t = GST_MESSAGE_TYPE(msg);
  const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                        ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                        : "<unknown>";

  std::string line = gst_message_to_string(msg);
  if (diag)
    diag->push_bus(gst_message_type_get_name(t), src ? src : "<unknown>", line);

  gst_message_unref(msg);
  gst_object_unref(bus);

  maybe_dump_dot(pipeline, std::string(where) + "_error");

  SessionReport rep = diag ? diag->snapshot_basic() : SessionReport{};
  rep.error_code = error_codes::kCaps;
  std::ostringstream note;
  note << "where=" << (where ? where : "GstDiagnosticsUtil::throw_if_bus_error")
       << " code=" << rep.error_code << " summary=GST ERROR"
       << " details=element='" << (src ? src : "<unknown>") << "' message='" << line << "'";
  rep.repro_note = note.str();
  if (diag)
    rep.repro_note += "\n" + boundary_summary(diag);
  rep.repro_note += "\nHint: inspect offending caps and upstream/downstream element contract.";
  throw SessionError(pipeline_internal::error_util::decorate_error(rep.error_code, rep.repro_note),
                     std::move(rep));
}

std::optional<GstSample*> try_pull_sample_sliced(GstElement* pipeline, GstElement* appsink,
                                                 int timeout_ms,
                                                 const std::shared_ptr<DiagCtx>& diag,
                                                 const char* where, bool* eos_seen) {
  if (!pipeline || !appsink)
    return std::nullopt;

  const int slice_ms = std::max(10, std::atoi(env_str("SIMA_GST_POLL_SLICE_MS", "200").c_str()));

  const bool infinite = (timeout_ms < 0);
  int remaining = timeout_ms;

  auto maybe_log_caps = [&](const char* reason) {
    if (!env_bool("SIMA_APPSINK_CAPS_DEBUG", false))
      return;
    if (!appsink)
      return;
    GstPad* pad = gst_element_get_static_pad(appsink, "sink");
    if (!pad) {
      std::fprintf(stderr, "[DBG] %s: appsink has no sink pad (%s)\n",
                   where ? where : "try_pull_sample_sliced", reason ? reason : "no_reason");
      return;
    }
    GstCaps* cur = gst_pad_get_current_caps(pad);
    GstCaps* allowed = gst_pad_get_allowed_caps(pad);
    std::string cur_str = gst_caps_to_string_safe(cur);
    std::string allowed_str = gst_caps_to_string_safe(allowed);
    if (cur)
      gst_caps_unref(cur);
    if (allowed)
      gst_caps_unref(allowed);
    gst_object_unref(pad);

    std::fprintf(stderr, "[DBG] %s: appsink caps (%s) current='%s' allowed='%s'\n",
                 where ? where : "try_pull_sample_sliced", reason ? reason : "unknown",
                 cur_str.c_str(), allowed_str.c_str());
    if (diag && !diag->pipeline_string.empty()) {
      std::fprintf(stderr, "[DBG] %s: pipeline=%s\n", where ? where : "try_pull_sample_sliced",
                   diag->pipeline_string.c_str());
    }
  };

  while (true) {
    int this_ms = 0;
    if (timeout_ms == 0)
      this_ms = 0;
    else if (infinite)
      this_ms = slice_ms;
    else
      this_ms = std::min(slice_ms, remaining);

    GstSample* s =
        gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), (guint64)this_ms * GST_MSECOND);
    if (s) {
      log_appsink_pull_state(appsink, s, where);
      log_appsink_last_sample(appsink, where);
      return s;
    }

    throw_if_bus_error(pipeline, diag, where);
    drain_bus(pipeline, diag, where, eos_seen);

    if (timeout_ms == 0) {
      maybe_log_caps("timeout=0");
      return std::nullopt;
    }

    if (!infinite) {
      remaining -= this_ms;
      if (remaining <= 0) {
        maybe_log_caps("timeout");
        return std::nullopt;
      }
    }
  }
}

void stop_and_unref(GstElement*& e) {
  if (!e)
    return;

  GstElement* local = e;
  e = nullptr;

  begin_teardown(local, /*flush=*/true);

  const bool async_only = env_bool("SIMA_GST_TEARDOWN_ASYNC", false);
  if (!async_only) {
    const int timeout_ms = teardown_timeout_ms();
    if (finish_teardown(local, timeout_ms))
      return;
    std::cerr << "[WARN] stop_and_unref(): teardown timed out; deferring to reaper.\n";
  }

  enqueue_teardown(local, /*flush=*/true);
}

void stop_and_unref_no_flush(GstElement*& e) {
  if (!e)
    return;

  GstElement* local = e;
  e = nullptr;

  // Defer teardown to the reaper to avoid blocking in gst_element_set_state.
  if (env_bool("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", true)) {
    enqueue_teardown(local, /*flush=*/false);
    return;
  }

  begin_teardown(local, /*flush=*/false);

  const bool async_only = env_bool("SIMA_GST_TEARDOWN_ASYNC", false);
  if (!async_only) {
    const int timeout_ms = teardown_timeout_ms();
    if (finish_teardown(local, timeout_ms))
      return;
    std::cerr << "[WARN] stop_and_unref(): teardown timed out; deferring to reaper.\n";
  }

  enqueue_teardown(local, /*flush=*/false);
}

} // namespace simaai::neat::pipeline_internal

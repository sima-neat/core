/**
 * @file SessionBuild.cpp
 * @brief Graph build/run implementation and shared pipeline construction helpers.
 *
 * This is a mechanical split from the original monolithic Graph.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Graph.h"
#include "GraphDetail.h"
#include "internal/GraphBuildInternal.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "pipeline/ErrorCodes.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/SimaaiMemory.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/internal/contract/ContractApply.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelRouteRetarget.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/Tess.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "pipeline/Tensor.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "nodes/common/Output.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/base/gstbasetransform.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <csignal>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

namespace {

std::string decorate_with_error_code(const std::string& code, const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

std::string append_hint_if_any(const std::string& message, const std::string& hint) {
  return pipeline_internal::error_util::append_hint(message, hint);
}

GraphReport make_error_report(const std::shared_ptr<DiagCtx>& diag, const std::string& code,
                              const std::string& message) {
  GraphReport rep = diag ? diag->snapshot_basic() : GraphReport{};
  rep.error_code = code;
  rep.repro_note = message;
  return rep;
}

[[noreturn]] void throw_session_error_simple(const std::string& code, const std::string& message,
                                             const std::string& hint = {},
                                             const std::string& pipeline = {}) {
  GraphReport rep = pipeline_internal::error_util::make_report(code, message, pipeline, hint);
  throw NeatError(decorate_with_error_code(rep.error_code, rep.repro_note), std::move(rep));
}

const internal::ModelLineageBinding* node_model_lineage_binding(const std::shared_ptr<Node>& node) {
  if (!node) {
    return nullptr;
  }
  if (auto* pre = dynamic_cast<Preproc*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return pre->options().model_lineage.get();
#else
    return nullptr;
#endif
  }
  if (auto* quant = dynamic_cast<Quant*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return quant->options().model_lineage.get();
#else
    return nullptr;
#endif
  }
  if (auto* tess = dynamic_cast<Tess*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return tess->options().model_lineage.get();
#else
    return nullptr;
#endif
  }
  if (auto* quanttess = dynamic_cast<QuantTess*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return quanttess->options().model_lineage.get();
#else
    return nullptr;
#endif
  }
  if (auto* cast = dynamic_cast<Cast*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return cast->options().model_lineage.get();
#else
    return nullptr;
#endif
  }
  if (auto* box = dynamic_cast<SimaBoxDecode*>(node.get())) {
#ifdef SIMA_NEAT_INTERNAL
    return box->model_lineage_binding_internal().get();
#else
    return nullptr;
#endif
  }
  if (auto* provider = dynamic_cast<const internal::ModelLineageProvider*>(node.get())) {
    return provider->model_lineage_binding();
  }
  return nullptr;
}

struct MaterializedLineageState {
  std::shared_ptr<Model> effective_model;
  bool changed = false;
};

} // namespace

void trace_step(const char* label) {
  if (!env_bool("SIMA_DISPATCHER_TRACE", false))
    return;
  std::fprintf(stderr, "[TRACE] %s\n", label);
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

void dump_prefixed_multiline(const std::string& prefix, const std::string& text) {
  if (prefix.empty() || text.empty()) {
    return;
  }
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find('\n', start);
    const std::string_view line = (end == std::string::npos)
                                      ? std::string_view(text).substr(start)
                                      : std::string_view(text).substr(start, end - start);
    std::fprintf(stderr, "%s%s\n", prefix.c_str(), std::string(line).c_str());
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
    if (start == text.size()) {
      break;
    }
  }
}

void maybe_dump_pipeline_string(const std::string& pipeline, const char* label) {
  if (!env_bool("SIMA_PIPELINE_STRING_DEBUG", false))
    return;
  if (!label)
    label = "pipeline";
  dump_prefixed_multiline(std::string("[PIPELINE:") + label + "] ", pipeline);
}

void stop_and_unref(GstElement*& e) {
  pipeline_internal::stop_and_unref(e);
}

// =====================================================================================
// Diag context + boundary stats (compatible with GraphReport)
// =====================================================================================

std::string boundary_summary_local(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->boundaries.empty())
    return "";

  const int64_t now = (int64_t)g_get_monotonic_time();
  int best_idx = -1;
  int64_t best_t = 0;
  bool best_out = false;

  // Take consistent snapshots (atomics -> plain stats) to avoid races and to keep
  // this function simple.
  std::vector<BoundaryFlowStats> snaps;
  snaps.reserve(diag->boundaries.size());

  for (size_t i = 0; i < diag->boundaries.size(); ++i) {
    const auto* b = diag->boundaries[i].get();
    if (!b)
      continue;

    BoundaryFlowStats s = b->snapshot();
    snaps.push_back(s);

    if (s.last_out_wall_us > best_t) {
      best_t = s.last_out_wall_us;
      best_idx = (int)i;
      best_out = true;
    }
    if (s.last_in_wall_us > best_t) {
      best_t = s.last_in_wall_us;
      best_idx = (int)i;
      best_out = false;
    }
  }

  std::ostringstream ss;
  ss << "BoundaryFlow:\n";
  for (const auto& s : snaps) {
    ss << "  - " << s.boundary_name << " after=" << s.after_node_index
       << " before=" << s.before_node_index << " in=" << s.in_buffers << " out=" << s.out_buffers
       << " last_in_age_ms=" << (s.last_in_wall_us ? ((now - s.last_in_wall_us) / 1000) : 0)
       << " last_out_age_ms=" << (s.last_out_wall_us ? ((now - s.last_out_wall_us) / 1000) : 0)
       << "\n";
  }

  if (best_idx >= 0 && best_t > 0) {
    const auto* b = diag->boundaries[(size_t)best_idx].get();
    if (b) {
      BoundaryFlowStats s = b->snapshot();
      ss << "LikelyStall: last activity " << (best_out ? "leaving " : "entering ")
         << s.boundary_name << " age_ms=" << ((now - best_t) / 1000) << " (after node "
         << s.after_node_index << ", before node " << s.before_node_index << ")\n";
    }
  }

  return ss.str();
}

// =====================================================================================
// Boundary probes
// =====================================================================================

struct BoundaryProbeCtx {
  simaai::neat::pipeline_internal::BoundaryFlowCounters* counters = nullptr; // atomics
  bool is_in = false;
};

struct StageProbeCtx {
  simaai::neat::pipeline_internal::StageTimingCounters* counters = nullptr;
};

struct ElementTimingProbeCtx {
  simaai::neat::pipeline_internal::ElementTimingCounters* counters = nullptr;
  bool is_sink = false;
  GQuark quark = 0;
};

struct ElementFlowProbeCtx {
  simaai::neat::pipeline_internal::ElementFlowCounters* counters = nullptr;
  bool is_sink = false;
  bool track_caps = false;
};

struct ElementProbeAttachState {
  simaai::neat::pipeline_internal::ElementTimingCounters* timing = nullptr;
  simaai::neat::pipeline_internal::ElementFlowCounters* flow = nullptr;
  GQuark timing_quark = 0;
  bool track_timing = false;
  bool track_flow = false;
  bool pad_added_connected = false;

  // Phase A: per-pad timing.  These point to DiagCtx-owned pad-counter rows;
  // we look them up lazily on first probe fire because some pads are
  // dynamically added (pad-added signal).
  simaai::neat::pipeline_internal::DiagCtx* diag_ctx = nullptr;
  std::string element_name;
  bool track_pad_timing = false;
  GQuark pad_timing_quark = 0;
  GQuark src_depart_quark = 0;
  GQuark src_depart_element_quark = 0;
};

struct ElementPadTimingProbeCtx {
  simaai::neat::pipeline_internal::DiagCtx* diag_ctx = nullptr;
  std::string element_name;
  std::string pad_name;
  bool is_sink = false;
  GQuark src_depart_quark = 0;
  GQuark src_depart_element_quark = 0;
  // Lazily resolved on first probe fire to avoid mutex contention on the
  // hot path; once non-null, all probe fires increment it directly.
  std::atomic<simaai::neat::pipeline_internal::ElementPadTimingCounters*> counters{nullptr};
};

struct ElementPadDirectionCounts {
  int sink = 0;
  int src = 0;
};

static bool should_skip_stage_element(GstElement* elem) {
  if (!elem)
    return true;
  const char* name = GST_ELEMENT_NAME(elem);
  if (name) {
    if (std::strcmp(name, "mysrc") == 0 || std::strcmp(name, "mysink") == 0)
      return true;
    if (g_str_has_prefix(name, "q_") || g_str_has_prefix(name, "boundary_"))
      return true;
  }
  GstElementFactory* factory = gst_element_get_factory(elem);
  if (!factory)
    return true;
  const char* fname = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
  if (!fname)
    return false;
  if (std::strcmp(fname, "queue") == 0)
    return true;
  if (std::strcmp(fname, "identity") == 0)
    return true;
  if (std::strcmp(fname, "appsrc") == 0)
    return true;
  if (std::strcmp(fname, "appsink") == 0)
    return true;
  return false;
}

static bool should_attach_transport_pad_timing(GstElement* elem) {
  if (!elem)
    return false;
  if (!should_skip_stage_element(elem))
    return true;
  GstElementFactory* factory = gst_element_get_factory(elem);
  if (!factory)
    return false;
  const char* fname = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
  if (!fname)
    return false;
  // Endpoints should stamp/observe transport edges, but they should not become
  // processing-node residency rows.  Keep queues/identities/boundaries skipped so they do not
  // overwrite the upstream processing element's src-departure stamp.
  return std::strcmp(fname, "appsrc") == 0 || std::strcmp(fname, "appsink") == 0;
}

static bool extract_sima_meta_key(GstBuffer* buf, pipeline_internal::ElementTimingKey& out) {
  if (!buf)
    return false;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buf, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return false;
  gint64 frame_id = -1;
  if (!gst_structure_get_int64(s, "frame-id", &frame_id))
    return false;
  out.frame_id = frame_id;
  const gchar* stream_id = gst_structure_get_string(s, "stream-id");
  out.stream_hash = stream_id ? std::hash<std::string>{}(stream_id) : 0U;
  return true;
}

static void update_element_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters,
                                  uint64_t delta) {
  if (!counters || delta == 0)
    return;
  using std::memory_order_relaxed;
  counters->samples.fetch_add(1, memory_order_relaxed);
  counters->total_us.fetch_add(delta, memory_order_relaxed);

  uint64_t cur = counters->max_us.load(memory_order_relaxed);
  while (delta > cur && !counters->max_us.compare_exchange_weak(cur, delta)) {
  }
  uint64_t min_cur = counters->min_us.load(memory_order_relaxed);
  while ((min_cur == 0 || delta < min_cur) &&
         !counters->min_us.compare_exchange_weak(min_cur, delta)) {
  }
}

static void add_pending_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters,
                               const pipeline_internal::ElementTimingKey& key, int64_t ts_us) {
  if (!counters)
    return;
  std::lock_guard<std::mutex> lock(counters->pending_mu);
  if (counters->pending.size() >= counters->max_pending) {
    counters->pending.clear();
  }
  counters->pending[key] = ts_us;
}

static void
add_pending_fifo_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters,
                        int64_t ts_us) {
  if (!counters || !counters->fifo_match_enabled)
    return;
  std::lock_guard<std::mutex> lock(counters->pending_mu);
  if (counters->pending_fifo.size() >= counters->max_pending) {
    counters->pending_fifo.clear();
  }
  counters->pending_fifo.push_back(ts_us);
}

static bool pop_pending_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters,
                               const pipeline_internal::ElementTimingKey& key, int64_t& ts_us) {
  if (!counters)
    return false;
  std::lock_guard<std::mutex> lock(counters->pending_mu);
  auto it = counters->pending.find(key);
  if (it == counters->pending.end())
    return false;
  ts_us = it->second;
  counters->pending.erase(it);
  return true;
}

static bool
pop_pending_fifo_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters,
                        int64_t& ts_us) {
  if (!counters || !counters->fifo_match_enabled)
    return false;
  std::lock_guard<std::mutex> lock(counters->pending_mu);
  if (counters->pending_fifo.empty())
    return false;
  ts_us = counters->pending_fifo.front();
  counters->pending_fifo.pop_front();
  return true;
}

static void
discard_pending_fifo_timing(simaai::neat::pipeline_internal::ElementTimingCounters* counters) {
  if (!counters || !counters->fifo_match_enabled)
    return;
  std::lock_guard<std::mutex> lock(counters->pending_mu);
  if (!counters->pending_fifo.empty()) {
    counters->pending_fifo.pop_front();
  }
}

static GstPadProbeReturn stage_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = reinterpret_cast<StageProbeCtx*>(user_data);
  if (!ctx || !ctx->counters)
    return GST_PAD_PROBE_OK;
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  static GQuark stage_quark = g_quark_from_static_string("sima.stage.ts.us");
  gint64* last =
      reinterpret_cast<gint64*>(gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), stage_quark));
  const gint64 now = g_get_monotonic_time();

  if (!last) {
    last = reinterpret_cast<gint64*>(g_malloc(sizeof(gint64)));
    *last = now;
    gst_mini_object_set_qdata(GST_MINI_OBJECT(buf), stage_quark, last,
                              reinterpret_cast<GDestroyNotify>(g_free));
    return GST_PAD_PROBE_OK;
  }

  const gint64 prev = *last;
  *last = now;
  if (prev <= 0 || now < prev)
    return GST_PAD_PROBE_OK;

  const uint64_t delta = static_cast<uint64_t>(now - prev);
  using std::memory_order_relaxed;
  ctx->counters->samples.fetch_add(1, memory_order_relaxed);
  ctx->counters->total_us.fetch_add(delta, memory_order_relaxed);

  uint64_t cur = ctx->counters->max_us.load(memory_order_relaxed);
  while (delta > cur && !ctx->counters->max_us.compare_exchange_weak(cur, delta)) {
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn element_timing_probe_cb(GstPad*, GstPadProbeInfo* info,
                                                 gpointer user_data) {
  auto* ctx = reinterpret_cast<ElementTimingProbeCtx*>(user_data);
  if (!ctx || !ctx->counters)
    return GST_PAD_PROBE_OK;

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  const gint64 now = g_get_monotonic_time();
  if (ctx->is_sink) {
    gint64* ts =
        reinterpret_cast<gint64*>(gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), ctx->quark));
    if (!ts) {
      ts = reinterpret_cast<gint64*>(g_malloc(sizeof(gint64)));
      gst_mini_object_set_qdata(GST_MINI_OBJECT(buf), ctx->quark, ts,
                                reinterpret_cast<GDestroyNotify>(g_free));
    }
    *ts = now;

    pipeline_internal::ElementTimingKey key;
    if (extract_sima_meta_key(buf, key)) {
      add_pending_timing(ctx->counters, key, now);
      // Also keep a bounded FIFO timestamp for simple 1:1 transforms that replace buffers or
      // strip metadata before the src pad.  If the normal qdata/key path succeeds on src, the
      // FIFO entry is discarded to stay aligned.
      add_pending_fifo_timing(ctx->counters, now);
    } else if (ctx->counters->fifo_match_enabled) {
      add_pending_fifo_timing(ctx->counters, now);
    } else {
      ctx->counters->missed_in.fetch_add(1, std::memory_order_relaxed);
    }
    return GST_PAD_PROBE_OK;
  }

  bool used = false;
  gint64 start = -1;
  gint64* ts =
      reinterpret_cast<gint64*>(gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), ctx->quark));
  if (ts && *ts > 0 && now >= *ts) {
    start = *ts;
    used = true;
    discard_pending_fifo_timing(ctx->counters);
  } else {
    pipeline_internal::ElementTimingKey key;
    if (extract_sima_meta_key(buf, key)) {
      if (pop_pending_timing(ctx->counters, key, start)) {
        used = true;
        discard_pending_fifo_timing(ctx->counters);
      } else if (pop_pending_fifo_timing(ctx->counters, start)) {
        used = true;
      } else {
        ctx->counters->missed_out.fetch_add(1, std::memory_order_relaxed);
      }
    } else if (pop_pending_fifo_timing(ctx->counters, start)) {
      used = true;
    } else {
      ctx->counters->missed_out.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (used && start > 0 && now >= start) {
    const uint64_t delta = static_cast<uint64_t>(now - start);
    update_element_timing(ctx->counters, delta);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn element_flow_probe_cb(GstPad* pad, GstPadProbeInfo* info,
                                               gpointer user_data) {
  auto* ctx = reinterpret_cast<ElementFlowProbeCtx*>(user_data);
  if (!ctx || !ctx->counters)
    return GST_PAD_PROBE_OK;

  const GstPadProbeType type = GST_PAD_PROBE_INFO_TYPE(info);
  if ((type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) != 0) {
    if (ctx->track_caps) {
      GstEvent* ev = GST_PAD_PROBE_INFO_EVENT(info);
      if (ev && GST_EVENT_TYPE(ev) == GST_EVENT_CAPS) {
        ctx->counters->caps_changes.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  if ((type & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;
  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  const uint64_t bytes = static_cast<uint64_t>(gst_buffer_get_size(buf));
  if (ctx->is_sink) {
    ctx->counters->in_buffers.fetch_add(1, std::memory_order_relaxed);
    ctx->counters->in_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    ctx->counters->out_buffers.fetch_add(1, std::memory_order_relaxed);
    ctx->counters->out_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  (void)pad;
  return GST_PAD_PROBE_OK;
}

static ElementProbeAttachState* get_element_attach_state(GstElement* elem) {
  if (!elem)
    return nullptr;
  static GQuark state_quark = g_quark_from_static_string("sima.elem.attach.state");
  auto* state =
      reinterpret_cast<ElementProbeAttachState*>(g_object_get_qdata(G_OBJECT(elem), state_quark));
  if (!state) {
    state = new ElementProbeAttachState();
    g_object_set_qdata_full(
        G_OBJECT(elem), state_quark, state,
        +[](gpointer p) { delete reinterpret_cast<ElementProbeAttachState*>(p); });
  }
  return state;
}

// Phase A: per-pad probe.  Updates inter-arrival cadence (samples + total +
// max gap between consecutive buffers on the same pad) and, for sink pads,
// queue-wait time read from the upstream src pad's qdata stamp.  Counters
// are looked up once per pad and cached in the ctx.
static GstPadProbeReturn element_pad_timing_probe_cb(GstPad*, GstPadProbeInfo* info,
                                                     gpointer user_data) {
  auto* ctx = reinterpret_cast<ElementPadTimingProbeCtx*>(user_data);
  if (!ctx || !ctx->diag_ctx)
    return GST_PAD_PROBE_OK;
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;
  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  // Lazy resolve the per-pad counter.  Hot-path: just one acquire-load.
  using simaai::neat::pipeline_internal::ElementPadTimingCounters;
  ElementPadTimingCounters* counters = ctx->counters.load(std::memory_order_acquire);
  if (!counters) {
    std::lock_guard<std::mutex> lk(ctx->diag_ctx->element_pad_timings_mu);
    counters = ctx->counters.load(std::memory_order_acquire);
    if (!counters) {
      auto fresh = std::make_unique<ElementPadTimingCounters>();
      fresh->element_name = ctx->element_name;
      fresh->pad_name = ctx->pad_name;
      fresh->is_sink = ctx->is_sink;
      if (ctx->is_sink) {
        fresh->transport_to_element_name = ctx->element_name;
      } else {
        fresh->transport_from_element_name = ctx->element_name;
      }
      counters = fresh.get();
      ctx->diag_ctx->element_pad_timings.push_back(std::move(fresh));
      ctx->counters.store(counters, std::memory_order_release);
    }
  }

  const gint64 now = g_get_monotonic_time();

  // Inter-arrival cadence.  First sample seeds last_arrival_us; subsequent
  // samples accumulate.
  const int64_t prev = counters->last_arrival_us.exchange(now, std::memory_order_acq_rel);
  if (prev > 0) {
    const uint64_t delta = static_cast<uint64_t>(now - prev);
    counters->inter_arrival_total_us.fetch_add(delta, std::memory_order_relaxed);
    uint64_t cur_max = counters->inter_arrival_max_us.load(std::memory_order_relaxed);
    while (delta > cur_max && !counters->inter_arrival_max_us.compare_exchange_weak(
                                  cur_max, delta, std::memory_order_relaxed)) {
    }
  }
  counters->samples.fetch_add(1, std::memory_order_relaxed);
  const gsize bsz = gst_buffer_get_size(buf);
  counters->bytes.fetch_add(static_cast<uint64_t>(bsz), std::memory_order_relaxed);

  if (ctx->is_sink && ctx->src_depart_quark) {
    // Queue wait = now - last upstream src departure timestamp.
    gint64* dep_ts = reinterpret_cast<gint64*>(
        gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), ctx->src_depart_quark));
    if (dep_ts && *dep_ts > 0 && now >= *dep_ts) {
      if (ctx->src_depart_element_quark) {
        const gchar* dep_element = reinterpret_cast<const gchar*>(
            gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), ctx->src_depart_element_quark));
        if (dep_element && *dep_element) {
          std::lock_guard<std::mutex> lk(ctx->diag_ctx->element_pad_timings_mu);
          if (counters->transport_from_element_name.empty()) {
            counters->transport_from_element_name = dep_element;
          }
        }
      }
      const uint64_t wait = static_cast<uint64_t>(now - *dep_ts);
      counters->queue_wait_total_us.fetch_add(wait, std::memory_order_relaxed);
      counters->queue_wait_samples.fetch_add(1, std::memory_order_relaxed);
      uint64_t cur_max = counters->queue_wait_max_us.load(std::memory_order_relaxed);
      while (wait > cur_max && !counters->queue_wait_max_us.compare_exchange_weak(
                                   cur_max, wait, std::memory_order_relaxed)) {
      }
    }
  } else if (!ctx->is_sink && ctx->src_depart_quark) {
    // Src pad: stamp departure timestamp on the buffer for the next sink
    // pad's queue-wait calculation.
    gint64* dep_ts = reinterpret_cast<gint64*>(
        gst_mini_object_get_qdata(GST_MINI_OBJECT(buf), ctx->src_depart_quark));
    if (!dep_ts) {
      dep_ts = reinterpret_cast<gint64*>(g_malloc(sizeof(gint64)));
      gst_mini_object_set_qdata(GST_MINI_OBJECT(buf), ctx->src_depart_quark, dep_ts,
                                reinterpret_cast<GDestroyNotify>(g_free));
    }
    *dep_ts = now;
    if (ctx->src_depart_element_quark) {
      gst_mini_object_set_qdata(GST_MINI_OBJECT(buf), ctx->src_depart_element_quark,
                                g_strdup(ctx->element_name.c_str()),
                                reinterpret_cast<GDestroyNotify>(g_free));
    }
  }

  return GST_PAD_PROBE_OK;
}

static void attach_element_probes_for_pad(GstPad* pad, ElementProbeAttachState* state) {
  if (!pad || !state)
    return;
  const GstPadDirection dir = gst_pad_get_direction(pad);
  const bool is_sink = (dir == GST_PAD_SINK);
  const bool is_src = (dir == GST_PAD_SRC);
  if (!is_sink && !is_src)
    return;

  static GQuark timing_pad_quark = g_quark_from_static_string("sima.elem.timing.attached");
  static GQuark flow_pad_quark = g_quark_from_static_string("sima.elem.flow.attached");
  static GQuark pad_timing_pad_quark = g_quark_from_static_string("sima.elem.pad_timing.attached");

  if (state->track_pad_timing && state->diag_ctx) {
    if (!g_object_get_qdata(G_OBJECT(pad), pad_timing_pad_quark)) {
      auto* ctx = new ElementPadTimingProbeCtx();
      ctx->diag_ctx = state->diag_ctx;
      ctx->element_name = state->element_name;
      const gchar* pad_name = GST_PAD_NAME(pad);
      ctx->pad_name = pad_name ? std::string(pad_name) : std::string("?");
      ctx->is_sink = is_sink;
      ctx->src_depart_quark = state->src_depart_quark;
      ctx->src_depart_element_quark = state->src_depart_element_quark;
      gst_pad_add_probe(
          pad, GST_PAD_PROBE_TYPE_BUFFER, element_pad_timing_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<ElementPadTimingProbeCtx*>(p); });
      g_object_set_qdata(G_OBJECT(pad), pad_timing_pad_quark, GINT_TO_POINTER(1));
    }
  }

  if (state->track_timing && state->timing && state->timing_quark) {
    if (!g_object_get_qdata(G_OBJECT(pad), timing_pad_quark)) {
      auto* ctx = new ElementTimingProbeCtx();
      ctx->counters = state->timing;
      ctx->is_sink = is_sink;
      ctx->quark = state->timing_quark;
      gst_pad_add_probe(
          pad, GST_PAD_PROBE_TYPE_BUFFER, element_timing_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<ElementTimingProbeCtx*>(p); });
      g_object_set_qdata(G_OBJECT(pad), timing_pad_quark, GINT_TO_POINTER(1));
    }
  }

  if (state->track_flow && state->flow) {
    if (!g_object_get_qdata(G_OBJECT(pad), flow_pad_quark)) {
      auto* ctx = new ElementFlowProbeCtx();
      ctx->counters = state->flow;
      ctx->is_sink = is_sink;
      ctx->track_caps = is_sink;
      gst_pad_add_probe(
          pad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM),
          element_flow_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<ElementFlowProbeCtx*>(p); });
      g_object_set_qdata(G_OBJECT(pad), flow_pad_quark, GINT_TO_POINTER(1));
    }
  }
}

static void attach_element_probes_for_existing_pads(GstElement* elem,
                                                    ElementProbeAttachState* state) {
  if (!elem || !state)
    return;
  GstIterator* it = gst_element_iterate_pads(elem);
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstPad* pad = GST_PAD(g_value_get_object(&item));
    if (pad) {
      attach_element_probes_for_pad(pad, state);
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

static void element_pad_added_cb(GstElement*, GstPad* pad, gpointer user_data) {
  auto* state = reinterpret_cast<ElementProbeAttachState*>(user_data);
  if (!state)
    return;
  attach_element_probes_for_pad(pad, state);
}

static ElementPadDirectionCounts count_element_pad_directions(GstElement* elem) {
  ElementPadDirectionCounts counts;
  if (!elem)
    return counts;
  GstIterator* it = gst_element_iterate_pads(elem);
  if (!it)
    return counts;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstPad* pad = GST_PAD(g_value_get_object(&item));
    if (pad) {
      const GstPadDirection dir = gst_pad_get_direction(pad);
      if (dir == GST_PAD_SINK) {
        ++counts.sink;
      } else if (dir == GST_PAD_SRC) {
        ++counts.src;
      }
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  return counts;
}

static GstPadProbeReturn boundary_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = reinterpret_cast<BoundaryProbeCtx*>(user_data);
  if (!ctx || !ctx->counters)
    return GST_PAD_PROBE_OK;

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  const int64_t now = (int64_t)g_get_monotonic_time();
  const GstClockTime pts = GST_BUFFER_PTS(buf);
  const int64_t pts_ns = (pts == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)pts;

  using std::memory_order_relaxed;

  if (ctx->is_in) {
    ctx->counters->in_buffers.fetch_add(1, memory_order_relaxed);
    ctx->counters->last_in_wall_us.store(now, memory_order_relaxed);
    if (pts_ns >= 0)
      ctx->counters->last_in_pts_ns.store(pts_ns, memory_order_relaxed);
  } else {
    ctx->counters->out_buffers.fetch_add(1, memory_order_relaxed);
    ctx->counters->last_out_wall_us.store(now, memory_order_relaxed);
    if (pts_ns >= 0)
      ctx->counters->last_out_pts_ns.store(pts_ns, memory_order_relaxed);
  }

  return GST_PAD_PROBE_OK;
}

void attach_stage_timing_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                                bool enable_from_options) {
  if (!pipeline || !diag)
    return;
  if (!enable_from_options && !env_bool("SIMA_GST_STAGE_TIMINGS", false))
    return;

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    if (should_skip_stage_element(elem)) {
      g_value_reset(&item);
      continue;
    }
    GstPad* src = gst_element_get_static_pad(elem, "src");
    if (!src) {
      g_value_reset(&item);
      continue;
    }
    auto counters = std::make_unique<pipeline_internal::StageTimingCounters>();
    const char* name = GST_ELEMENT_NAME(elem);
    counters->stage_name = name ? name : "unknown";
    pipeline_internal::StageTimingCounters* ptr = counters.get();
    diag->stage_timings.push_back(std::move(counters));

    auto* ctx = new StageProbeCtx();
    ctx->counters = ptr;
    gst_pad_add_probe(
        src, GST_PAD_PROBE_TYPE_BUFFER, stage_probe_cb, ctx,
        +[](gpointer p) { delete reinterpret_cast<StageProbeCtx*>(p); });
    gst_object_unref(src);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

static void attach_element_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                                  bool enable_timing, bool enable_flow) {
  if (!pipeline || !diag)
    return;
  if (!enable_timing && !enable_flow)
    return;

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const bool skip_stage = should_skip_stage_element(elem);
    const bool attach_pad_transport = enable_timing && should_attach_transport_pad_timing(elem);
    if (skip_stage && !attach_pad_transport) {
      g_value_reset(&item);
      continue;
    }

    ElementProbeAttachState* state = get_element_attach_state(elem);
    if (!state) {
      g_value_reset(&item);
      continue;
    }

    const char* name = GST_ELEMENT_NAME(elem);
    const std::string elem_name = name ? name : "unknown";
    const ElementPadDirectionCounts pad_counts = count_element_pad_directions(elem);

    if (enable_timing && !skip_stage && !state->timing) {
      auto counters = std::make_unique<pipeline_internal::ElementTimingCounters>();
      counters->element_name = elem_name;
      counters->fifo_match_enabled = (pad_counts.sink == 1 && pad_counts.src == 1);
      const std::string quark_name = "sima.elem.ts." + elem_name;
      state->timing_quark = g_quark_from_string(quark_name.c_str());
      state->timing = counters.get();
      diag->element_timings.push_back(std::move(counters));
    }

    if (enable_flow && !skip_stage && !state->flow) {
      auto counters = std::make_unique<pipeline_internal::ElementFlowCounters>();
      counters->element_name = elem_name;
      state->flow = counters.get();
      diag->element_flows.push_back(std::move(counters));
    }

    if (enable_timing && !skip_stage)
      state->track_timing = true;
    if (enable_flow && !skip_stage)
      state->track_flow = true;
    // Phase A: per-pad timing rides on the same enable-timing knob.  Counters
    // are added lazily on first probe fire (so dynamically-added pads work).
    if (attach_pad_transport) {
      state->track_pad_timing = true;
      state->diag_ctx = diag.get();
      state->element_name = elem_name;
      static GQuark pad_timing_quark = g_quark_from_static_string("sima.elem.pad_timing.ts");
      state->pad_timing_quark = pad_timing_quark;
      // src_depart_quark is process-global so any src-pad probe can stamp a
      // buffer and any sink-pad probe further downstream can read it.
      static GQuark src_depart_quark = g_quark_from_static_string("sima.elem.src_depart.us");
      state->src_depart_quark = src_depart_quark;
      static GQuark src_depart_element_quark =
          g_quark_from_static_string("sima.elem.src_depart.element");
      state->src_depart_element_quark = src_depart_element_quark;
    }
    if (!state->track_timing && !state->track_flow && !state->track_pad_timing) {
      g_value_reset(&item);
      continue;
    }

    attach_element_probes_for_existing_pads(elem, state);

    if (!state->pad_added_connected) {
      g_signal_connect(elem, "pad-added", G_CALLBACK(element_pad_added_cb), state);
      state->pad_added_connected = true;
    }

    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

void attach_element_timing_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
                                  bool enable_from_options) {
  if (!enable_from_options && !env_bool("SIMA_GST_ELEMENT_TIMINGS", false))
    return;
  attach_element_probes(pipeline, diag, true, false);
}

void attach_element_flow_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag) {
  if (!env_bool("SIMA_GST_FLOW_DEBUG", false))
    return;
  attach_element_probes(pipeline, diag, false, true);
}

struct DebugBufferProbeCtx {
  std::string element_name;
  std::string label;
  std::string pad_name;
  int remaining = 5;
  int seen = 0;
  guint64 last_size = 0;
  bool logged_transition = false;
  bool log_pool = false;
  bool logged_pool_before = false;
  bool logged_pool_after = false;
};

static void update_simaai_mem_flags(GstMemory* mem, std::uint64_t* target_flags,
                                    std::uint64_t* mem_flags) {
  if (!mem || !target_flags || !mem_flags)
    return;
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_EV74)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_EV74;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS0)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS0;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS1)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS1;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS2)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS2;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS3)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS3;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_OCM)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_OCM;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_GENERIC)) {
    *target_flags |= GST_SIMAAI_MEMORY_TARGET_GENERIC;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_CACHED)) {
    *mem_flags |= GST_SIMAAI_MEMORY_FLAG_CACHED;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_RDONLY)) {
    *mem_flags |= GST_SIMAAI_MEMORY_FLAG_RDONLY;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_DEFAULT)) {
    *mem_flags |= GST_SIMAAI_MEMORY_FLAG_DEFAULT;
  }
}

static void log_detess_output_memory_flags(const DebugBufferProbeCtx& ctx, GstBuffer* buf) {
  if (!buf)
    return;
  const guint n_mems = gst_buffer_n_memory(buf);
  std::fprintf(stderr, "[DBG] detess-output-memflags element=%s mems=%u\n",
               ctx.element_name.c_str(), n_mems);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buf, i);
    if (!mem)
      continue;
    std::uint64_t target_flags = 0;
    std::uint64_t mem_flags = 0;
    update_simaai_mem_flags(mem, &target_flags, &mem_flags);
    gsize offset = 0;
    gsize maxsize = 0;
    gst_memory_get_sizes(mem, &offset, &maxsize);
    std::fprintf(stderr,
                 "[DBG] detess-output-mem[%u] target_flags=0x%llx mem_flags=0x%llx "
                 "offset=%zu maxsize=%zu\n",
                 i, static_cast<unsigned long long>(target_flags),
                 static_cast<unsigned long long>(mem_flags), static_cast<size_t>(offset),
                 static_cast<size_t>(maxsize));
  }
}

static void log_buffer_memory_flags(const DebugBufferProbeCtx& ctx, GstBuffer* buf) {
  if (!buf)
    return;
  const guint n_mems = gst_buffer_n_memory(buf);
  std::uint64_t target_flags = 0;
  std::uint64_t mem_flags = 0;
  std::fprintf(stderr, "[DBG] buffer-memflags element=%s label=%s mems=%u\n",
               ctx.element_name.c_str(), ctx.label.empty() ? "<none>" : ctx.label.c_str(), n_mems);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buf, i);
    if (!mem) {
      std::fprintf(stderr, "  - mem[%u]=null\n", i);
      continue;
    }
    update_simaai_mem_flags(mem, &target_flags, &mem_flags);
    gsize offset = 0;
    gsize maxsize = 0;
    const gsize size = gst_memory_get_sizes(mem, &offset, &maxsize);
    const char* alloc_name = (mem->allocator && GST_IS_OBJECT(mem->allocator))
                                 ? GST_OBJECT_NAME(mem->allocator)
                                 : "<none>";
    std::fprintf(stderr, "  - mem[%u] size=%zu offset=%zu max=%zu flags=0x%x alloc=%s\n", i,
                 static_cast<size_t>(size), static_cast<size_t>(offset),
                 static_cast<size_t>(maxsize), mem->mini_object.flags, alloc_name);
  }
  std::fprintf(stderr, "[DBG] buffer-memflags-summary target=0x%llx flags=0x%llx\n",
               static_cast<unsigned long long>(target_flags),
               static_cast<unsigned long long>(mem_flags));
}

static void log_detess_output_pool_params(GstPad* pad, DebugBufferProbeCtx* ctx, const char* when) {
  if (!pad || !ctx)
    return;
  GstElement* elem = gst_pad_get_parent_element(pad);
  if (!elem)
    return;
  bool logged = false;
  if (GST_IS_BASE_TRANSFORM(elem)) {
    GstBufferPool* pool = gst_base_transform_get_buffer_pool(GST_BASE_TRANSFORM(elem));
    if (pool) {
      GstStructure* config = gst_buffer_pool_get_config(pool);
      if (config) {
        GstCaps* caps = nullptr;
        guint size = 0;
        guint min_buffers = 0;
        guint max_buffers = 0;
        gst_buffer_pool_config_get_params(config, &caps, &size, &min_buffers, &max_buffers);
        gchar* caps_str = caps ? gst_caps_to_string(caps) : nullptr;
        std::fprintf(stderr,
                     "[DBG] detess-output-pool element=%s when=%s size=%u min=%u max=%u caps=%s\n",
                     ctx->element_name.c_str(), when ? when : "unknown", size, min_buffers,
                     max_buffers, caps_str ? caps_str : "(null)");
        if (caps_str)
          g_free(caps_str);
        gst_structure_free(config);
        logged = true;
      }
      gst_object_unref(pool);
    }
  }

  if (!logged) {
    GstCaps* caps = gst_pad_get_current_caps(pad);
    GstQuery* query = gst_query_new_allocation(caps, TRUE);
    const GstPadDirection dir = gst_pad_get_direction(pad);
    gboolean ok = false;
    if (dir == GST_PAD_SRC) {
      ok = gst_pad_peer_query(pad, query);
    } else {
      ok = gst_pad_query(pad, query);
      if (!ok)
        ok = gst_pad_peer_query(pad, query);
    }
    if (ok) {
      const guint pools = gst_query_get_n_allocation_pools(query);
      std::fprintf(stderr, "[DBG] detess-output-pool element=%s when=%s pools=%u\n",
                   ctx->element_name.c_str(), when ? when : "unknown", pools);
      for (guint i = 0; i < pools; ++i) {
        GstBufferPool* pool = nullptr;
        guint size = 0;
        guint min_buffers = 0;
        guint max_buffers = 0;
        gst_query_parse_nth_allocation_pool(query, i, &pool, &size, &min_buffers, &max_buffers);
        std::fprintf(stderr, "[DBG] detess-output-pool[%u] size=%u min=%u max=%u\n", i, size,
                     min_buffers, max_buffers);
        if (pool)
          gst_object_unref(pool);
      }
      logged = true;
    }
    if (query)
      gst_query_unref(query);
    if (caps)
      gst_caps_unref(caps);
  }
  if (!logged) {
    std::fprintf(stderr, "[DBG] detess-output-pool element=%s when=%s pool=<unavailable>\n",
                 ctx->element_name.c_str(), when ? when : "unknown");
  }
  gst_object_unref(elem);
}

static void dump_node_debug_options(const std::vector<std::shared_ptr<Node>>& nodes,
                                    const NameTransform& name_transform) {
  if (!env_bool("SIMA_GST_OPTIONS_DEBUG", false))
    return;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node)
      continue;
    NodeFragment frag = make_node_fragment(node, static_cast<int>(i), name_transform);
    std::ostringstream ss;
    ss << "[OPTS] node=" << i << " kind=" << node->kind() << " elements=[";
    for (size_t k = 0; k < frag.element_names.size(); ++k) {
      if (k)
        ss << ",";
      ss << frag.element_names[k];
    }
    ss << "]";
    if (!frag.fragment.empty())
      ss << " fragment=\"" << frag.fragment << "\"";
    std::fprintf(stderr, "%s\n", ss.str().c_str());

    if (auto* src = dynamic_cast<Input*>(node.get())) {
      const auto& opt = src->options();
      std::ostringstream os;
      os << "[OPTS]   Input caps=\"" << src->caps_string() << "\""
         << " payload_type=" << static_cast<int>(opt.payload_type)
         << " media_type=" << resolve_input_media_type(opt) << " format=" << opt.format
         << " w=" << opt.width << " h=" << opt.height << " d=" << opt.depth << " fps=" << opt.fps_n
         << "/" << opt.fps_d << " buffer_name=" << opt.buffer_name
         << " use_simaai_pool=" << (opt.use_simaai_pool ? "true" : "false")
         << " pool_min=" << opt.pool_min_buffers << " pool_max=" << opt.pool_max_buffers;
      std::fprintf(stderr, "%s\n", os.str().c_str());
    }

    if (auto* sink = dynamic_cast<Output*>(node.get())) {
      const auto& opt = sink->options();
      std::ostringstream os;
      os << "[OPTS]   Output max_buffers=" << opt.max_buffers
         << " drop=" << (opt.drop ? "true" : "false") << " sync=" << (opt.sync ? "true" : "false");
      std::fprintf(stderr, "%s\n", os.str().c_str());
    }

    if (auto* pre = dynamic_cast<Preproc*>(node.get())) {
      const auto& opt = pre->options();
      std::ostringstream os;
      os << "[OPTS]   Preproc config=\"" << pre->config_path() << "\""
         << " next_cpu=" << opt.next_cpu << " input=" << opt.input_width() << "x"
         << opt.input_height() << " fmt=" << opt.input_img_type << " output=" << opt.output_width()
         << "x" << opt.output_height() << " out_fmt=" << opt.output_img_type
         << " normalize=" << (opt.normalize ? "true" : "false")
         << " tessellate=" << (opt.tessellate ? "true" : "false")
         << " num_buffers=" << opt.num_buffers;
      std::fprintf(stderr, "%s\n", os.str().c_str());
    }

    if (auto* det = dynamic_cast<DetessDequant*>(node.get())) {
      const auto& opt = det->options();
      std::ostringstream os;
      os << "[OPTS]   DetessDequant config=\"" << det->config_path() << "\""
         << " upstream_name=" << opt.upstream_name << " element_name=" << opt.element_name
         << " num_buffers=" << opt.num_buffers;
      std::fprintf(stderr, "%s\n", os.str().c_str());
    }

    if (auto* box = dynamic_cast<SimaBoxDecode*>(node.get())) {
      std::ostringstream os;
      os << "[OPTS]   SimaBoxDecode typed-manifest=true";
      std::fprintf(stderr, "%s\n", os.str().c_str());
    }
  }
}

static void dump_element_properties(GstElement* elem) {
  if (!elem)
    return;
  GObject* obj = G_OBJECT(elem);
  GObjectClass* klass = G_OBJECT_GET_CLASS(obj);
  const char* name = GST_ELEMENT_NAME(elem);
  const char* type = G_OBJECT_TYPE_NAME(obj);
  std::fprintf(stderr, "[OPTS] element=%s type=%s\n", name ? name : "<none>",
               type ? type : "<unknown>");
  guint n_props = 0;
  GParamSpec** props = g_object_class_list_properties(klass, &n_props);
  for (guint i = 0; i < n_props; ++i) {
    GParamSpec* ps = props[i];
    if (!ps || !(ps->flags & G_PARAM_READABLE))
      continue;
    GValue val = G_VALUE_INIT;
    g_value_init(&val, ps->value_type);
    g_object_get_property(obj, ps->name, &val);
    gchar* value_str = g_strdup_value_contents(&val);
    std::fprintf(stderr, "  - %s=%s\n", ps->name, value_str ? value_str : "<null>");
    if (value_str)
      g_free(value_str);
    g_value_unset(&val);
  }
  if (props)
    g_free(props);
}

void dump_pipeline_element_properties(GstElement* pipeline) {
  if (!pipeline || !env_bool("SIMA_GST_OPTIONS_DEBUG", false))
    return;
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    dump_element_properties(elem);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

static GstPadProbeReturn debug_buffer_probe_cb(GstPad* pad, GstPadProbeInfo* info,
                                               gpointer user_data) {
  auto* ctx = reinterpret_cast<DebugBufferProbeCtx*>(user_data);
  if (!ctx || ctx->remaining <= 0)
    return GST_PAD_PROBE_OK;
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;
  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;
  const guint64 size = gst_buffer_get_size(buf);
  GstCaps* caps = gst_pad_get_current_caps(pad);
  gchar* caps_str = caps ? gst_caps_to_string(caps) : nullptr;
  const std::string label = ctx->label.empty() ? "buffer" : ctx->label;
  const guint64 prev = ctx->last_size;
  ctx->last_size = size;
  const int count = ++ctx->seen;
  std::cerr << "[DBG] buffer label=" << label << " element=" << ctx->element_name
            << " count=" << count << " size=" << size;
  if (count > 1) {
    std::cerr << " prev=" << prev;
  }
  if (GST_BUFFER_PTS_IS_VALID(buf)) {
    std::cerr << " pts=" << GST_BUFFER_PTS(buf);
  }
  if (caps_str) {
    std::cerr << " caps=" << caps_str;
  }
  std::cerr << "\n";
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
    dump_sima_meta_full(buf, label.c_str());
  }
  if (env_bool("SIMA_GST_BUFFER_MEMFLAGS_DEBUG", false)) {
    log_buffer_memory_flags(*ctx, buf);
  }
  if (ctx->log_pool && label == "detess-output" && size > 0 && !ctx->logged_pool_before) {
    log_detess_output_pool_params(pad, ctx, "before_flip");
    ctx->logged_pool_before = true;
  }
  if (prev > 0 && size == 0) {
    std::cerr << "[DBG] buffer-transition label=" << label << " element=" << ctx->element_name
              << " count=" << count << " prev_size=" << prev << " now_size=0\n";
    if (label == "detess-output" && ctx->log_pool && !ctx->logged_pool_after) {
      log_detess_output_pool_params(pad, ctx, "after_flip");
      ctx->logged_pool_after = true;
    }
    if (label == "detess-output" && !ctx->logged_transition) {
      log_detess_output_memory_flags(*ctx, buf);
      ctx->logged_transition = true;
    }
    if (label == "detess-output" && env_bool("SIMA_DETESS_ASSERT_ON_ZERO", false)) {
      std::fprintf(stderr, "[ERR] detess output size==0, aborting for backtrace.\n");
      std::raise(SIGABRT);
    }
  }
  if (caps_str)
    g_free(caps_str);
  if (caps)
    gst_caps_unref(caps);
  --ctx->remaining;
  return GST_PAD_PROBE_OK;
}

void attach_debug_detess_input_probes(GstElement* pipeline) {
  if (!pipeline)
    return;
  const bool want_in = env_bool("SIMA_GST_DETESS_INPUT_DEBUG", false);
  const bool want_out = env_bool("SIMA_GST_DETESS_OUTPUT_DEBUG", false);
  if (!want_in && !want_out)
    return;
  const int limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 5);
  const bool want_pool = env_bool("SIMA_GST_DETESS_POOL_DEBUG", false);

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    const char* name = elem ? GST_ELEMENT_NAME(elem) : nullptr;
    const std::string elem_name = name ? name : "";
    if (elem_name.find("detess") == std::string::npos) {
      g_value_reset(&item);
      continue;
    }
    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (pad_it) {
      GValue pad_val = G_VALUE_INIT;
      while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
        GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
        if (pad && gst_pad_get_direction(pad) == GST_PAD_SINK && want_in) {
          auto* ctx = new DebugBufferProbeCtx();
          ctx->element_name = elem_name;
          ctx->label = "detess-input";
          ctx->remaining = limit;
          gst_pad_add_probe(
              pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
              +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
        }
        if (pad && gst_pad_get_direction(pad) == GST_PAD_SRC && want_out) {
          auto* ctx = new DebugBufferProbeCtx();
          ctx->element_name = elem_name;
          ctx->label = "detess-output";
          ctx->remaining = limit;
          ctx->log_pool = want_pool;
          gst_pad_add_probe(
              pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
              +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
        }
        g_value_reset(&pad_val);
      }
      g_value_unset(&pad_val);
      gst_iterator_free(pad_it);
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

void attach_debug_appsink_probes(GstElement* pipeline) {
  if (!pipeline || !env_bool("SIMA_GST_APPSINK_BUFFER_DEBUG", false))
    return;
  const int limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 5);
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem || !GST_IS_APP_SINK(elem)) {
      g_value_reset(&item);
      continue;
    }
    const char* name = GST_ELEMENT_NAME(elem);
    const std::string elem_name = name ? name : "appsink";
    GstPad* pad = gst_element_get_static_pad(elem, "sink");
    if (pad) {
      auto* ctx = new DebugBufferProbeCtx();
      ctx->element_name = elem_name;
      ctx->label = "appsink";
      ctx->remaining = limit;
      gst_pad_add_probe(
          pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
      gst_object_unref(pad);
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

void attach_debug_all_buffer_probes(GstElement* pipeline) {
  if (!pipeline || !env_bool("SIMA_GST_ALL_BUFFER_DEBUG", false))
    return;
  const int limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 3);
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const char* name = GST_ELEMENT_NAME(elem);
    const std::string elem_name = name ? name : "";
    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (!pad_it) {
      g_value_reset(&item);
      continue;
    }
    GValue pad_val = G_VALUE_INIT;
    while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
      GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
      if (pad) {
        const char* pad_name = GST_PAD_NAME(pad);
        const GstPadDirection dir = gst_pad_get_direction(pad);
        const char* dir_name = (dir == GST_PAD_SRC)    ? "src"
                               : (dir == GST_PAD_SINK) ? "sink"
                                                       : "pad";
        auto* ctx = new DebugBufferProbeCtx();
        ctx->element_name = elem_name;
        ctx->pad_name = pad_name ? pad_name : "";
        ctx->label = "all:" + std::string(dir_name) + ":" + ctx->pad_name;
        ctx->remaining = limit;
        gst_pad_add_probe(
            pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
            +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
      }
      g_value_reset(&pad_val);
    }
    g_value_unset(&pad_val);
    gst_iterator_free(pad_it);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

static std::vector<std::string> split_debug_patterns(const std::string& raw) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    const std::string trimmed = trim_copy(cur);
    if (!trimmed.empty())
      out.push_back(lower_copy(trimmed));
    cur.clear();
  };
  for (char c : raw) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      flush();
      continue;
    }
    cur.push_back(c);
  }
  flush();
  return out;
}

static bool match_any_pattern(const std::string& name, const std::vector<std::string>& patterns) {
  if (name.empty() || patterns.empty())
    return false;
  const std::string lower = lower_copy(name);
  for (const auto& pat : patterns) {
    if (!pat.empty() && lower.find(pat) != std::string::npos)
      return true;
  }
  return false;
}

void attach_debug_element_buffer_probes(GstElement* pipeline) {
  if (!pipeline)
    return;
  const std::string raw = env_str("SIMA_GST_ELEMENT_BUFFER_DEBUG", "");
  if (raw.empty())
    return;
  const auto patterns = split_debug_patterns(raw);
  if (patterns.empty())
    return;

  const std::string dir = lower_copy(env_str("SIMA_GST_ELEMENT_BUFFER_DEBUG_DIR", "both"));
  const bool want_src = (dir == "both" || dir == "src");
  const bool want_sink = (dir == "both" || dir == "sink");
  const int limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 5);

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const char* elem_name_c = GST_ELEMENT_NAME(elem);
    const std::string elem_name = elem_name_c ? elem_name_c : "";
    GstElementFactory* factory = gst_element_get_factory(elem);
    const char* factory_name =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
    const std::string factory_str = factory_name ? factory_name : "";

    if (!match_any_pattern(elem_name, patterns) && !match_any_pattern(factory_str, patterns)) {
      g_value_reset(&item);
      continue;
    }

    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (!pad_it) {
      g_value_reset(&item);
      continue;
    }
    GValue pad_val = G_VALUE_INIT;
    while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
      GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
      if (pad) {
        const GstPadDirection dirn = gst_pad_get_direction(pad);
        const bool is_src = (dirn == GST_PAD_SRC);
        const bool is_sink = (dirn == GST_PAD_SINK);
        if ((is_src && !want_src) || (is_sink && !want_sink)) {
          g_value_reset(&pad_val);
          continue;
        }
        const char* pad_name = GST_PAD_NAME(pad);
        const char* dir_name = is_src ? "src" : "sink";
        auto* ctx = new DebugBufferProbeCtx();
        ctx->element_name = elem_name.empty() ? factory_str : elem_name;
        ctx->pad_name = pad_name ? pad_name : "";
        ctx->label =
            std::string("elem:") + ctx->element_name + ":" + dir_name + ":" + ctx->pad_name;
        ctx->remaining = limit;
        gst_pad_add_probe(
            pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
            +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
      }
      g_value_reset(&pad_val);
    }
    g_value_unset(&pad_val);
    gst_iterator_free(pad_it);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

struct BoxDecodeProbeState {
  int limit = 0;
  bool log_links = false;
  std::string element_name;
};

static bool is_boxdecode_element(GstElement* elem) {
  if (!elem)
    return false;
  GstElementFactory* factory = gst_element_get_factory(elem);
  const gchar* factory_name =
      factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
  if (factory_name && std::string(factory_name).find("boxdecode") != std::string::npos) {
    return true;
  }
  const char* elem_name = GST_ELEMENT_NAME(elem);
  return elem_name && std::string(elem_name).find("boxdecode") != std::string::npos;
}

static void log_pad_link_status(GstPad* pad, const BoxDecodeProbeState* state) {
  if (!pad || !state || !state->log_links)
    return;
  GstPad* peer = gst_pad_get_peer(pad);
  GstElement* peer_parent = peer ? gst_pad_get_parent_element(peer) : nullptr;
  const GstPadDirection dir = gst_pad_get_direction(pad);
  const char* dir_name = (dir == GST_PAD_SINK) ? "sink" : (dir == GST_PAD_SRC) ? "src" : "pad";
  const char* pad_name = GST_PAD_NAME(pad);
  const char* peer_name = peer ? GST_PAD_NAME(peer) : "<none>";
  const char* peer_parent_name = peer_parent ? GST_ELEMENT_NAME(peer_parent) : "<none>";
  std::fprintf(
      stderr,
      "[DBG] boxdecode pad link element=%s pad=%s dir=%s linked=%d peer=%s peer-parent=%s\n",
      state->element_name.c_str(), pad_name ? pad_name : "<unknown>", dir_name,
      gst_pad_is_linked(pad), peer_name ? peer_name : "<none>",
      peer_parent_name ? peer_parent_name : "<none>");
  if (peer_parent)
    gst_object_unref(peer_parent);
  if (peer)
    gst_object_unref(peer);
}

static void attach_boxdecode_buffer_probe(GstPad* pad, const BoxDecodeProbeState* state) {
  if (!pad || !state)
    return;
  const GstPadDirection dir = gst_pad_get_direction(pad);
  if (dir != GST_PAD_SINK && dir != GST_PAD_SRC)
    return;
  const char* dir_name = (dir == GST_PAD_SINK) ? "sink" : "src";
  const char* pad_name = GST_PAD_NAME(pad);
  auto* ctx = new DebugBufferProbeCtx();
  ctx->element_name = state->element_name;
  ctx->pad_name = pad_name ? pad_name : "";
  ctx->label = std::string("boxdecode:") + dir_name + ":" + ctx->pad_name;
  ctx->remaining = state->limit;
  gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, ctx,
      +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
  log_pad_link_status(pad, state);
}

static void boxdecode_pad_added_cb(GstElement* /*elem*/, GstPad* pad, gpointer user_data) {
  auto* state = reinterpret_cast<BoxDecodeProbeState*>(user_data);
  attach_boxdecode_buffer_probe(pad, state);
}

void attach_boxdecode_debug_probes(GstElement* pipeline) {
  if (!pipeline || !env_bool("SIMA_GST_BOXDECODE_BUFFER_DEBUG", false))
    return;
  const int limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 3);
  const bool log_links = env_bool("SIMA_GST_PAD_LINK_DEBUG", false);
  static GQuark state_quark = g_quark_from_static_string("sima_gst_boxdecode_probe_state");

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem || !is_boxdecode_element(elem)) {
      g_value_reset(&item);
      continue;
    }
    if (g_object_get_qdata(G_OBJECT(elem), state_quark)) {
      g_value_reset(&item);
      continue;
    }
    const char* elem_name = GST_ELEMENT_NAME(elem);
    auto* state = new BoxDecodeProbeState();
    state->limit = limit;
    state->log_links = log_links;
    state->element_name = elem_name ? elem_name : "boxdecode";
    g_object_set_qdata_full(
        G_OBJECT(elem), state_quark, state,
        +[](gpointer p) { delete reinterpret_cast<BoxDecodeProbeState*>(p); });

    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (pad_it) {
      GValue pad_val = G_VALUE_INIT;
      while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
        GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
        if (pad) {
          attach_boxdecode_buffer_probe(pad, state);
        }
        g_value_reset(&pad_val);
      }
      g_value_unset(&pad_val);
      gst_iterator_free(pad_it);
    }

    g_signal_connect(elem, "pad-added", G_CALLBACK(boxdecode_pad_added_cb), state);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
}

void attach_boundary_probes(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline || !diag)
    return;
  if (!env_bool("SIMA_GST_BOUNDARY_PROBES", false))
    return;
  const bool want_debug = env_bool("SIMA_GST_BOUNDARY_BUFFER_DEBUG", false);
  const int debug_limit = env_int("SIMA_GST_BUFFER_DEBUG_LIMIT", 5);

  for (auto& bptr : diag->boundaries) {
    auto* b = bptr.get();
    if (!b)
      continue;

    GstElement* ident = gst_bin_get_by_name(GST_BIN(pipeline), b->boundary_name.c_str());
    if (!ident)
      continue;

    GstPad* sink = gst_element_get_static_pad(ident, "sink");
    GstPad* src = gst_element_get_static_pad(ident, "src");

    if (sink) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->counters = b;
      ctx->is_in = true;
      gst_pad_add_probe(
          sink, GST_PAD_PROBE_TYPE_BUFFER, boundary_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      if (want_debug) {
        auto* dbg = new DebugBufferProbeCtx();
        dbg->element_name = b->boundary_name;
        dbg->label = "boundary:in";
        dbg->remaining = debug_limit;
        gst_pad_add_probe(
            sink, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, dbg,
            +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
      }
      gst_object_unref(sink);
    }

    if (src) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->counters = b;
      ctx->is_in = false;
      gst_pad_add_probe(
          src, GST_PAD_PROBE_TYPE_BUFFER, boundary_probe_cb, ctx,
          +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      if (want_debug) {
        auto* dbg = new DebugBufferProbeCtx();
        dbg->element_name = b->boundary_name;
        dbg->label = "boundary:out";
        dbg->remaining = debug_limit;
        gst_pad_add_probe(
            src, GST_PAD_PROBE_TYPE_BUFFER, debug_buffer_probe_cb, dbg,
            +[](gpointer p) { delete reinterpret_cast<DebugBufferProbeCtx*>(p); });
      }
      gst_object_unref(src);
    }

    gst_object_unref(ident);
  }
}

void drain_bus_into_diag(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline || !diag)
    return;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus)
    return;

  const GstMessageType mask = static_cast<GstMessageType>(GST_MESSAGE_ANY & ~GST_MESSAGE_ERROR);
  while (GstMessage* msg = gst_bus_pop_filtered(bus, mask)) {
    std::string line = gst_message_to_string(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                          ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                          : "<unknown>";
    diag->push_bus(gst_message_type_get_name(GST_MESSAGE_TYPE(msg)), src ? src : "<unknown>", line);
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

void throw_if_bus_error_local(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag,
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

  // Always record something helpful in the bus log.
  std::string line = gst_message_to_string(msg);
  const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                        ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                        : "<unknown>";

  if (diag) {
    diag->push_bus(gst_message_type_get_name(GST_MESSAGE_TYPE(msg)), src ? src : "<unknown>", line);
  }

  // Prefer structured error parsing (better than generic stringify).
  GError* e = nullptr;
  gchar* dbg = nullptr;
  gst_message_parse_error(msg, &e, &dbg);

  const std::string err_msg = (e && e->message) ? e->message : "unknown";
  const std::string dbg_msg = (dbg) ? dbg : "";

  if (e)
    g_error_free(e);
  if (dbg)
    g_free(dbg);

  gst_message_unref(msg);
  gst_object_unref(bus);

  maybe_dump_dot(pipeline, std::string(where) + "_error");

  GraphReport rep = diag ? diag->snapshot_basic() : GraphReport{};
  rep.error_code = error_codes::kCaps;
  std::ostringstream note;
  note << "where=" << (where ? where : "Graph::build") << " code=" << rep.error_code
       << " summary=GST ERROR" << " details=element='" << (src ? src : "<unknown>") << "' error='"
       << err_msg << "'";
  if (!dbg_msg.empty()) {
    note << " gst_debug='" << dbg_msg << "'";
  }
  rep.repro_note = note.str();
  const std::string boundary = boundary_summary_local(diag);
  if (!boundary.empty()) {
    rep.repro_note += "\n" + boundary;
  }
  rep.repro_note += "\nHint: inspect caps negotiation and offending element diagnostics.";
  throw NeatError(decorate_with_error_code(rep.error_code, rep.repro_note), std::move(rep));
}

bool should_insert_boundaries_for_mode(const char* mode_key, bool def_val) {
  return env_bool(mode_key, def_val);
}

static int async_queue2_depth(int requested_depth = 0) {
  if (requested_depth > 0) {
    return std::min(requested_depth, 1024);
  }
  static int cached_env_or_default = []() {
    const char* env = std::getenv("SIMA_ASYNC_QUEUE2_DEPTH");
    if (env && *env) {
      char* end = nullptr;
      const long parsed = std::strtol(env, &end, 10);
      if (end != env && parsed > 0 && parsed <= 1024) {
        return static_cast<int>(parsed);
      }
    }
    return 5;
  }();
  return cached_env_or_default;
}
static std::string async_queue2_fragment(int requested_depth = 0) {
  const int depth = async_queue2_depth(requested_depth);
  return "queue max-size-buffers=" + std::to_string(depth) + " max-size-bytes=0 max-size-time=0";
}

static std::string append_boolean_property_if_absent(std::string fragment,
                                                     const std::string& factory,
                                                     const std::string& property, bool value) {
  if (!value || fragment.find(factory) == std::string::npos ||
      fragment.find(property + "=") != std::string::npos) {
    return fragment;
  }
  const std::size_t factory_pos = fragment.find(factory);
  std::size_t insert_pos = fragment.find('!', factory_pos);
  if (insert_pos == std::string::npos) {
    insert_pos = fragment.size();
  }
  fragment.insert(insert_pos, " " + property + "=true ");
  return fragment;
}

static std::string set_int_property_if_factory_present(std::string fragment,
                                                       const std::string& factory,
                                                       const std::string& property, int value) {
  if (value <= 0 || fragment.find(factory) == std::string::npos) {
    return fragment;
  }
  const std::string key = property + "=";
  const size_t pos = fragment.find(key);
  if (pos != std::string::npos) {
    size_t vstart = pos + key.size();
    size_t vend = vstart;
    while (vend < fragment.size() && std::isdigit(static_cast<unsigned char>(fragment[vend]))) {
      ++vend;
    }
    fragment.replace(vstart, vend - vstart, std::to_string(value));
    return fragment;
  }
  const std::size_t factory_pos = fragment.find(factory);
  std::size_t insert_pos = fragment.find('!', factory_pos);
  if (insert_pos == std::string::npos) {
    insert_pos = fragment.size();
  }
  fragment.insert(insert_pos, " " + key + std::to_string(value) + " ");
  return fragment;
}

static std::string read_gst_fragment_property(const std::string& fragment,
                                              const std::string& property) {
  const std::string key = property + "=";
  const size_t pos = fragment.find(key);
  if (pos == std::string::npos) {
    return {};
  }
  size_t vstart = pos + key.size();
  if (vstart >= fragment.size()) {
    return {};
  }
  char quote = 0;
  if (fragment[vstart] == '\'' || fragment[vstart] == '"') {
    quote = fragment[vstart++];
  }
  size_t vend = vstart;
  while (vend < fragment.size()) {
    const char c = fragment[vend];
    if (quote) {
      if (c == quote)
        break;
    } else if (std::isspace(static_cast<unsigned char>(c)) || c == '!') {
      break;
    }
    ++vend;
  }
  return fragment.substr(vstart, vend - vstart);
}

static bool fragment_is_graph223_dequant_processcvu(const std::string& fragment) {
  const std::string lower = lower_copy(fragment);
  if (lower.find("neatprocesscvu") == std::string::npos) {
    return false;
  }
  return lower.find("dequant") != std::string::npos ||
         lower.find("graph-id=223") != std::string::npos ||
         lower.find("graph_id=223") != std::string::npos;
}

static std::string maybe_replace_post_dequant_with_prepared_runner(const std::string& fragment,
                                                                   const GraphOptions* sess_opt) {
  if (!sess_opt) {
    return fragment;
  }
  const std::string mode = lower_copy(sess_opt->prepared_runner.mode);
  if (!(mode == "dequant" || mode == "post" || mode == "dequant-only")) {
    return fragment;
  }
  if (!fragment_is_graph223_dequant_processcvu(fragment)) {
    return fragment;
  }

  const std::string name = read_gst_fragment_property(fragment, "name");
  std::string stage_id = read_gst_fragment_property(fragment, "stage-id");
  if (stage_id.empty()) {
    stage_id = read_gst_fragment_property(fragment, "stage_id");
  }
  if (stage_id.empty()) {
    stage_id = name;
  }

  std::ostringstream out;
  out << "neatpreparedrunner";
  if (!name.empty()) {
    out << " name=" << name;
  }
  out << " mode=dequant";
  if (!stage_id.empty()) {
    out << " post-stage-id=" << stage_id;
  }
  if (sess_opt->prepared_runner.ring_depth > 0) {
    out << " ring-depth=" << std::min(sess_opt->prepared_runner.ring_depth, 64);
  }
  if (sess_opt->prepared_runner.profile) {
    out << " profile=true";
  }
  if (!sess_opt->prepared_runner.dequant_flags.empty()) {
    out << " dequant-flags=" << sess_opt->prepared_runner.dequant_flags;
  }
  return out.str();
}

static bool env_explicit_false_session_build_local(const char* name) {
  const char* raw = std::getenv(name);
  return raw && *raw && std::strcmp(raw, "0") == 0;
}

static std::string apply_session_fast_path_options_to_fragment(std::string fragment,
                                                               const GraphOptions* sess_opt) {
  if (!sess_opt) {
    return fragment;
  }
  fragment = maybe_replace_post_dequant_with_prepared_runner(fragment, sess_opt);
  const bool processcvu_async = sess_opt->processcvu.async &&
                                !env_explicit_false_session_build_local("SIMA_PROCESSCVU_ASYNC");
  const bool processmla_async =
      sess_opt->processmla.async &&
      !env_explicit_false_session_build_local("SIMA_PROCESSMLA_SAFE_ASYNC");
  fragment = append_boolean_property_if_absent(std::move(fragment), "neatprocesscvu", "async",
                                               processcvu_async);
  fragment = append_boolean_property_if_absent(std::move(fragment), "neatprocessmla", "async",
                                               processmla_async);
  fragment =
      set_int_property_if_factory_present(std::move(fragment), "neatprocessmla", "num-buffers",
                                          sess_opt->processmla.output_pool_buffers);
  fragment = append_boolean_property_if_absent(std::move(fragment), "neatprocessmla",
                                               "defer-output-invalidate",
                                               sess_opt->processmla.defer_output_invalidate);
  return fragment;
}
static std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b);

TensorCompatDims sample_shape_limits_dims(const SampleSpec& spec) {
  if (spec.media_type == "application/vnd.simaai.tensor" && !spec.shape.empty()) {
    return tensor_compat_dims_from_shape(spec.shape, spec.layout);
  }
  TensorCompatDims out;
  out.width = spec.width;
  out.height = spec.height;
  out.depth = spec.depth;
  return out;
}

std::string sample_shape_debug_string(const SampleSpec& spec) {
  if (spec.shape.empty()) {
    return "<none>";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < spec.shape.size(); ++i) {
    if (i) {
      oss << ",";
    }
    oss << spec.shape[i];
  }
  oss << "]";
  return oss.str();
}

std::uint64_t checked_tensor_shape_elements(const std::vector<int64_t>& shape) {
  std::uint64_t total = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    total = checked_mul_u64(total, static_cast<std::uint64_t>(dim));
    if (total == 0U) {
      return 0U;
    }
  }
  return total;
}

SampleSpec infer_input_spec(const InputOptions& opt, const cv::Mat& input, const char* where) {
  const char* tag = where ? where : "Graph::infer_input_spec";
  try {
    if (input.empty()) {
      throw std::invalid_argument(std::string(tag) + ": input frame is empty");
    }
    cv::Mat contiguous = input;
    if (!contiguous.isContinuous()) {
      contiguous = input.clone();
    }
    InputOptions relaxed = pipeline_internal::normalize_shape_bounds(opt);
    relaxed.width = -1;
    relaxed.height = -1;
    relaxed.depth = -1;
    simaai::neat::Tensor neat = tensor_from_cv_mat(contiguous, relaxed, tag);
    return derive_tensor_spec_or_throw(neat, relaxed, tag);
  } catch (const NeatError&) {
    throw;
  } catch (const std::exception& e) {
    const std::string msg = std::string(tag) + ": input shape/spec inference failed: " + e.what();
    throw_session_error_simple(error_codes::kInputShape, msg,
                               "Verify input media type, format, dimensions, and tensor layout.");
  }
}

SampleSpec infer_input_spec(const InputOptions& opt, const simaai::neat::Tensor& input,
                            const char* where) {
  const char* tag = where ? where : "Graph::infer_input_spec";
  try {
    InputOptions relaxed = pipeline_internal::normalize_shape_bounds(opt);
    relaxed.width = -1;
    relaxed.height = -1;
    relaxed.depth = -1;
    return derive_tensor_spec_or_throw(input, relaxed, tag);
  } catch (const NeatError&) {
    throw;
  } catch (const std::exception& e) {
    const std::string msg =
        std::string(tag) + ": input tensor shape/spec inference failed: " + e.what();
    throw_session_error_simple(error_codes::kInputShape, msg,
                               "Verify tensor shape/layout/dtype and media mapping.");
  }
}

SampleSpec infer_input_spec(const InputOptions& opt, const Sample& sample, const char* where) {
  const char* tag = where ? where : "Graph::infer_input_spec";
  try {
    SampleSpec spec = derive_sample_spec_or_throw(sample);
    const TensorCompatDims actual_dims = sample_shape_limits_dims(spec);

    const InputOptions normalized = pipeline_internal::normalize_shape_bounds(opt);
    const auto limits = pipeline_internal::resolve_shape_limits(normalized, spec);
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_DEBUG", false)) {
      std::fprintf(stderr,
                   "[DBG] infer_input_spec(sample) tag=%s opt{media=%s format=%s w=%d h=%d d=%d "
                   "max=%dx%dx%d} "
                   "spec{media=%s format=%s shape=%s w=%d h=%d d=%d tensor_envelope=%d} "
                   "limits{seed=%dx%dx%d max=%dx%dx%d}\n",
                   tag, resolve_input_media_type(normalized).c_str(),
                   normalized.format.str().c_str(), normalized.width, normalized.height,
                   normalized.depth, normalized.max_width, normalized.max_height,
                   normalized.max_depth, spec.media_type.c_str(), spec.format.c_str(),
                   sample_shape_debug_string(spec).c_str(), actual_dims.width, actual_dims.height,
                   actual_dims.depth, spec.tensor_envelope_transport ? 1 : 0, limits.seed_width,
                   limits.seed_height, limits.seed_depth, limits.max_width, limits.max_height,
                   limits.max_depth);
    }

    if (!spec.tensor_envelope_transport && limits.max_width > 0 && actual_dims.width > 0 &&
        actual_dims.width > limits.max_width) {
      throw std::invalid_argument(std::string(tag) + ": width exceeds effective max");
    }
    if (!spec.tensor_envelope_transport && limits.max_height > 0 && actual_dims.height > 0 &&
        actual_dims.height > limits.max_height) {
      throw std::invalid_argument(std::string(tag) + ": height exceeds effective max");
    }
    if (!spec.tensor_envelope_transport && limits.max_depth > 0 && actual_dims.depth > 0 &&
        actual_dims.depth > limits.max_depth) {
      throw std::invalid_argument(std::string(tag) + ": depth exceeds effective max");
    }

    if (spec.kind == SampleMediaKind::RawVideo) {
      spec.fps_n = opt.fps_n;
      spec.fps_d = opt.fps_d;
      spec.caps_key = capkey_from_spec(spec);
      spec.caps_string = caps_string_from_spec(spec);
    }
    return spec;
  } catch (const NeatError&) {
    throw;
  } catch (const std::exception& e) {
    const std::string msg =
        std::string(tag) + ": input sample shape/spec inference failed: " + e.what();
    throw_session_error_simple(error_codes::kInputShape, msg,
                               "Verify Sample kind/payload metadata and tensor storage.");
  }
}

void enforce_mla_pipeline_guard(const char* where, const std::string& pipeline, const void* owner) {
  pipeline_internal::enforce_single_mla_pipeline(where ? where : "Graph", pipeline, owner, "Graph");
}

void dump_flow_snapshot(const std::shared_ptr<DiagCtx>& diag, const char* label) {
  if (!diag)
    return;
  const char* tag = label ? label : "flow";
  std::fprintf(stderr, "[FLOW:%s] boundaries=%zu elements=%zu\n", tag, diag->boundaries.size(),
               diag->element_flows.size());
  for (const auto& b : diag->boundaries) {
    if (!b)
      continue;
    const auto s = b->snapshot();
    std::fprintf(stderr,
                 "[FLOW:%s] boundary=%s after=%d before=%d in=%llu out=%llu last_in_us=%lld "
                 "last_out_us=%lld\n",
                 tag, s.boundary_name.c_str(), s.after_node_index, s.before_node_index,
                 static_cast<unsigned long long>(s.in_buffers),
                 static_cast<unsigned long long>(s.out_buffers),
                 static_cast<long long>(s.last_in_wall_us),
                 static_cast<long long>(s.last_out_wall_us));
  }
  for (const auto& e : diag->element_flows) {
    if (!e)
      continue;
    const auto s = e->snapshot();
    if (s.in_buffers == 0 && s.out_buffers == 0 && s.caps_changes == 0)
      continue;
    std::fprintf(
        stderr, "[FLOW:%s] element=%s in=%llu out=%llu in_bytes=%llu out_bytes=%llu caps=%llu\n",
        tag, s.element_name.c_str(), static_cast<unsigned long long>(s.in_buffers),
        static_cast<unsigned long long>(s.out_buffers), static_cast<unsigned long long>(s.in_bytes),
        static_cast<unsigned long long>(s.out_bytes),
        static_cast<unsigned long long>(s.caps_changes));
  }
}

void dump_pipeline_string_force(const std::shared_ptr<DiagCtx>& diag, const char* label) {
  if (!diag)
    return;
  if (diag->pipeline_string.empty())
    return;
  if (!env_bool("SIMA_PIPELINE_STRING_DEBUG", false) &&
      !pipeline_internal::ux::should_emit_topic_for_current_context(
          pipeline_internal::ux::VerboseTopic::Pipeline)) {
    return;
  }
  const char* tag = label ? label : "pipeline";
  dump_prefixed_multiline(std::string("[PIPELINE:") + tag + "] ", diag->pipeline_string);
}

static bool starts_with_token(const std::string& s, const char* token) {
  const size_t len = std::strlen(token);
  if (s.size() < len)
    return false;
  if (s.compare(0, len, token) != 0)
    return false;
  if (s.size() == len)
    return true;
  return std::isspace(static_cast<unsigned char>(s[len])) != 0;
}

// =====================================================================================
// gst_props — typed primitives for reading/writing "key=value" properties in
// gst-launch segments. Single source of truth for the value-token grammar.
//
// Replaces the ad-hoc per-call-site scanners that produced bugs #4 (force_bool_prop
// corrupting numeric values), #5 (clamp_int_prop conflating absent with empty), and
// #6 (enforce_num_buffers_for_plugin conflating absent with required-match).
//
// Value-token grammar: from the byte after '=' up to (but not including) the next
// whitespace, '!', or end-of-segment. The same grammar applies to numbers, booleans,
// and any other unquoted property value gst-launch accepts in our generated strings.
// =====================================================================================
namespace gst_props {

enum class State {
  Absent,        // "key=" not found
  PresentEmpty,  // "key=" found but value is empty (e.g. "key= !" or "key=" at EOS)
  PresentValued, // "key=" found with a non-empty value token
};

struct PropRef {
  State state = State::Absent;
  std::size_t key_pos = std::string::npos;     // start of "key" in segment
  std::size_t value_start = std::string::npos; // first char after "="
  std::size_t value_end = std::string::npos;   // one past last value char
  std::string_view value;                      // valid iff state == PresentValued
};

static bool is_value_terminator(char c) noexcept {
  // gst-launch value tokens (in our generated strings) are not quoted. Terminate
  // at whitespace or the segment separator. This is the only place this grammar
  // is defined — every caller goes through find_prop / set_prop / get_int / get_bool.
  return std::isspace(static_cast<unsigned char>(c)) != 0 || c == '!';
}

static PropRef find_prop(std::string_view segment, std::string_view key) noexcept {
  const auto pos = segment.find(key);
  if (pos == std::string_view::npos) {
    return {};
  }
  PropRef ref;
  ref.key_pos = pos;
  ref.value_start = pos + key.size();
  ref.value_end = ref.value_start;
  while (ref.value_end < segment.size() && !is_value_terminator(segment[ref.value_end])) {
    ++ref.value_end;
  }
  if (ref.value_end == ref.value_start) {
    ref.state = State::PresentEmpty;
  } else {
    ref.state = State::PresentValued;
    ref.value = segment.substr(ref.value_start, ref.value_end - ref.value_start);
  }
  return ref;
}

// Replace the value of `key=` in `segment` with `new_value`, or append
// "key=<new_value>" if absent. Correct for numeric, boolean, and string values
// alike — the only "scanner" lives in find_prop.
static void set_prop(std::string& segment, std::string_view key, std::string_view new_value) {
  const auto ref = find_prop(segment, key);
  if (ref.state == State::Absent) {
    if (!segment.empty() && segment.back() != ' ') {
      segment.push_back(' ');
    }
    segment.append(key);
    segment.append(new_value);
    return;
  }
  segment.replace(ref.value_start, ref.value_end - ref.value_start, new_value);
}

// Return the parsed integer iff PresentValued AND fully numeric. nullopt for
// Absent, PresentEmpty, or non-numeric value. Never conflates "absent" with
// "empty" with "garbage".
static std::optional<int64_t> get_int(std::string_view segment, std::string_view key) noexcept {
  const auto ref = find_prop(segment, key);
  if (ref.state != State::PresentValued) {
    return std::nullopt;
  }
  // Accept an optional leading '-' so we don't silently reject signed properties.
  std::size_t i = 0;
  if (!ref.value.empty() && ref.value[0] == '-') {
    ++i;
    if (i == ref.value.size())
      return std::nullopt; // bare '-'
  }
  for (std::size_t j = i; j < ref.value.size(); ++j) {
    if (!std::isdigit(static_cast<unsigned char>(ref.value[j])))
      return std::nullopt;
  }
  int64_t v = 0;
  for (std::size_t j = i; j < ref.value.size(); ++j) {
    v = v * 10 + (ref.value[j] - '0');
  }
  return (i == 1) ? -v : v;
}

} // namespace gst_props

// ----- Refactored callers of the old scanners ----------------------------------------
// Each function below preserves its previous return-true-on-success / clamp semantics
// where unchanged, but now distinguishes PropPresence::PresentEmpty from Absent so the
// caller can react properly (fixes bug #5).

enum class PropPresence {
  Absent,       // property not in segment — caller may want to append a default
  PresentEmpty, // "key=" with no value — malformed; caller should treat as Absent
                // or surface a diagnostic, but never as "value was set"
  Present,      // "key=<value>" found; value was clamped in place if needed
};

static PropPresence clamp_int_prop(std::string& segment, const std::string& key, int max_val,
                                   bool clamp_zero_as_unlimited) {
  const auto ref = gst_props::find_prop(segment, key);
  switch (ref.state) {
  case gst_props::State::Absent:
    return PropPresence::Absent;
  case gst_props::State::PresentEmpty:
    return PropPresence::PresentEmpty;
  case gst_props::State::PresentValued:
    break;
  }
  const auto parsed = gst_props::get_int(segment, key);
  if (!parsed.has_value()) {
    return PropPresence::Present; // non-numeric value — leave it alone, treat as set
  }
  const int64_t val = *parsed;
  if (val > max_val || (clamp_zero_as_unlimited && val == 0)) {
    gst_props::set_prop(segment, key, std::to_string(max_val));
  }
  return PropPresence::Present;
}

// Returns the parsed num-buffers value, or nullopt if the property is absent OR has
// an empty value. Previously returned -1 for both — collapsing the distinction is
// what made bug #6 possible.
static std::optional<int64_t> parse_num_buffers_in_segment(const std::string& segment) {
  return gst_props::get_int(segment, "num-buffers=");
}

static void enforce_num_buffers_for_plugin(const std::string& pipeline, const char* plugin,
                                           int required, const char* context,
                                           int alt_required = -1) {
  size_t start = 0;
  while (start < pipeline.size()) {
    size_t bang = pipeline.find('!', start);
    std::string seg =
        (bang == std::string::npos) ? pipeline.substr(start) : pipeline.substr(start, bang - start);
    std::string trimmed = trim_copy(seg);
    if (starts_with_token(trimmed, plugin)) {
      const auto ref = gst_props::find_prop(trimmed, "num-buffers=");
      const auto val = (ref.state == gst_props::State::PresentValued)
                           ? gst_props::get_int(trimmed, "num-buffers=")
                           : std::nullopt;
      const bool matches =
          val.has_value() && (*val == required || (alt_required >= 0 && *val == alt_required));
      if (!matches) {
        std::ostringstream oss;
        oss << context << ": " << plugin << " num-buffers must be ";
        if (alt_required >= 0) {
          oss << required << " or " << alt_required;
        } else {
          oss << required;
        }
        // Distinguish the three failure modes — bug #6 used to print "got 0"
        // for both Absent and PresentEmpty, which was actively misleading.
        switch (ref.state) {
        case gst_props::State::Absent:
          oss << " (property absent on segment '" << trimmed << "').";
          break;
        case gst_props::State::PresentEmpty:
          oss << " (property present with empty value on segment '" << trimmed << "').";
          break;
        case gst_props::State::PresentValued:
          if (val.has_value()) {
            oss << " (got " << *val << ").";
          } else {
            oss << " (got non-numeric value '" << ref.value << "').";
          }
          break;
        }
        throw_session_error_simple(error_codes::kCaps, oss.str());
      }
    }
    if (bang == std::string::npos)
      break;
    start = bang + 1;
  }
}

static void dump_num_buffers_for_plugin(const std::string& pipeline, const char* plugin,
                                        const char* context) {
  if (!env_bool("SIMA_MLA_NUM_BUFFERS_DEBUG", false))
    return;
  size_t start = 0;
  int idx = 0;
  while (start < pipeline.size()) {
    size_t bang = pipeline.find('!', start);
    std::string seg =
        (bang == std::string::npos) ? pipeline.substr(start) : pipeline.substr(start, bang - start);
    std::string trimmed = trim_copy(seg);
    if (starts_with_token(trimmed, plugin)) {
      const auto val = parse_num_buffers_in_segment(trimmed);
      // -1 sentinel preserved for the debug log so existing log greps still parse.
      std::fprintf(stderr, "[DBG] %s num-buffers plugin=%s idx=%d val=%lld seg=%s\n",
                   context ? context : "Graph", plugin, idx,
                   static_cast<long long>(val.value_or(-1)), trimmed.c_str());
      ++idx;
    }
    if (bang == std::string::npos)
      break;
    start = bang + 1;
  }
}

void enforce_mla_num_buffers(const std::string& pipeline, const char* context,
                             bool allow_one = false) {
  if (env_bool("SIMA_MLA_NUM_BUFFERS_DEBUG", false)) {
    std::fprintf(stderr, "[DBG] %s enforce_mla_num_buffers allow_one=%d has_mla=%d\n",
                 context ? context : "Graph", allow_one ? 1 : 0,
                 (pipeline.find("neatprocessmla") != std::string::npos) ? 1 : 0);
  }
  if (allow_one) {
    return;
  }
  const int required = 4;
  const bool has_mla = (pipeline.find("neatprocessmla") != std::string::npos);
  const int alt = allow_one ? 1 : -1;
  if (has_mla) {
    dump_num_buffers_for_plugin(pipeline, "neatprocessmla", context);
    enforce_num_buffers_for_plugin(pipeline, "neatprocessmla", required, context, alt);
  }
  dump_num_buffers_for_plugin(pipeline, "neatprocesscvu", context);
  enforce_num_buffers_for_plugin(pipeline, "neatprocesscvu", required, context, alt);
}

static std::string clamp_queue_buffers(std::string pipeline, int max_buffers) {
  // Fix for bug #5: a `Present` result means we successfully read a value; an
  // `Absent` *or* `PresentEmpty` result both mean "no usable value set", so the
  // default should be appended. Treating PresentEmpty as "already set" (the
  // original behavior) left malformed "key=" tokens in the launch string.
  auto needs_default = [](PropPresence p) { return p != PropPresence::Present; };
  std::vector<std::string> segments;
  size_t start = 0;
  while (start < pipeline.size()) {
    size_t bang = pipeline.find('!', start);
    std::string seg =
        (bang == std::string::npos) ? pipeline.substr(start) : pipeline.substr(start, bang - start);
    std::string trimmed = trim_copy(seg);
    if (starts_with_token(trimmed, "queue") || starts_with_token(trimmed, "queue2")) {
      const PropPresence buf = clamp_int_prop(seg, "max-size-buffers=", max_buffers, true);
      const PropPresence bytes = clamp_int_prop(seg, "max-size-bytes=", 0, false);
      const PropPresence time = clamp_int_prop(seg, "max-size-time=", 0, false);
      if (needs_default(buf))
        gst_props::set_prop(seg, "max-size-buffers=", std::to_string(max_buffers));
      if (needs_default(bytes))
        gst_props::set_prop(seg, "max-size-bytes=", "0");
      if (needs_default(time))
        gst_props::set_prop(seg, "max-size-time=", "0");
    }
    segments.push_back(trim_copy(seg));
    if (bang == std::string::npos)
      break;
    start = bang + 1;
  }
  std::string out;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (i)
      out += " ! ";
    out += segments[i];
  }
  return out;
}

static int max_num_buffers_in_pipeline_strict(const std::string& pipeline);

std::string clamp_terminal_appsink_num_buffers(std::string pipeline, int num_buffers_override) {
  const int forced_min = (num_buffers_override > 0) ? std::max(2, num_buffers_override) : 2;

  auto trim = [](std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
      --end;
    }
    s = s.substr(start, end - start);
  };

  auto to_lower = [](std::string s) {
    for (char& c : s) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
  };
  std::vector<std::string> parts;
  size_t start = 0;
  while (start < pipeline.size()) {
    const size_t bang = pipeline.find('!', start);
    if (bang == std::string::npos) {
      parts.push_back(pipeline.substr(start));
      break;
    }
    parts.push_back(pipeline.substr(start, bang - start));
    start = bang + 1;
  }

  int appsink_index = -1;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    std::string seg = parts[i];
    trim(seg);
    const std::string lower = to_lower(seg);
    if (starts_with_token(lower, "appsink")) {
      appsink_index = static_cast<int>(i);
    }
  }
  if (appsink_index <= 0) {
    return pipeline;
  }

  bool changed = false;
  for (int i = appsink_index - 1; i >= 0; --i) {
    std::string seg = parts[static_cast<std::size_t>(i)];
    trim(seg);
    const std::string lower = to_lower(seg);
    if (lower.empty() || starts_with_token(lower, "identity") ||
        starts_with_token(lower, "queue") || starts_with_token(lower, "queue2") ||
        starts_with_token(lower, "capsfilter")) {
      continue;
    }

    // Sync mode intentionally collapses internal pools to a single buffer, but
    // an appsink output boundary can retain a pulled GstSample/GstBuffer until
    // after the next push/pull has started.  If the real producer immediately
    // upstream of appsink also has only one output buffer, that legitimate
    // lifetime overlap starves the producer's pool.  Keep exactly one spare at
    // this terminal boundary, generically: walk left from appsink, skip transparent
    // elements, and override the closest already-rendered num-buffers token.  That is
    // more surgical than adding a property by factory name and avoids relying on
    // registry introspection for plugins whose properties were already rendered into
    // the launch string.
    if (gst_props::find_prop(seg, "num-buffers=").state == gst_props::State::Absent) {
      continue;
    }
    if (const auto cur = parse_num_buffers_in_segment(seg); cur.has_value() && *cur >= forced_min) {
      return pipeline;
    }
    gst_props::set_prop(seg, "num-buffers=", std::to_string(forced_min));
    parts[static_cast<std::size_t>(i)] = " " + seg + " ";
    changed = true;
    break;
  }

  if (!changed) {
    return pipeline;
  }

  std::ostringstream out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) {
      out << " ! ";
    }
    std::string seg = parts[i];
    trim(seg);
    out << seg;
  }

  if (env_bool("SIMA_MLA_NUM_BUFFERS_DEBUG", false)) {
    std::fprintf(stderr, "[DBG] Graph::build(input) clamp_terminal_appsink_num_buffers min=%d\n",
                 forced_min);
  }
  return out.str();
}

std::string clamp_sync_pipeline(std::string pipeline, int num_buffers_override) {
  pipeline = clamp_queue_buffers(std::move(pipeline), 1);
  const int forced = (num_buffers_override > 0) ? num_buffers_override : 1;
  auto trim = [](std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
      ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
      --end;
    s = s.substr(start, end - start);
  };
  // The old code carried two duplicated value-token scanners here as lambdas
  // (force_bool_prop / force_num_buffers); both replaced by gst_props::set_prop
  // which uses one canonical value-token grammar. Bug #4 (force_bool_prop
  // corrupting "key=1" into "key=false1") goes away by construction.
  auto force_bool_prop = [&](std::string& seg, std::string_view key, bool value) {
    gst_props::set_prop(seg, key, value ? "true" : "false");
  };
  auto force_num_buffers = [&](std::string& seg) {
    gst_props::set_prop(seg, "num-buffers=", std::to_string(forced));
  };
  auto apply_num_buffers = [&](const std::function<bool(const std::string&)>& match) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < pipeline.size()) {
      size_t bang = pipeline.find('!', start);
      if (bang == std::string::npos) {
        parts.push_back(pipeline.substr(start));
        break;
      }
      parts.push_back(pipeline.substr(start, bang - start));
      start = bang + 1;
    }
    for (auto& part : parts) {
      std::string seg = part;
      trim(seg);
      if (!match(seg))
        continue;
      force_num_buffers(seg);
      part = " " + seg + " ";
    }
    std::ostringstream out;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i)
        out << " ! ";
      std::string seg = parts[i];
      trim(seg);
      out << seg;
    }
    pipeline = out.str();
  };

  apply_num_buffers([](const std::string& seg) {
    return seg.find("neatprocesscvu") != std::string::npos ||
           seg.find("neatprocessmla") != std::string::npos;
  });

  if (forced <= 1) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < pipeline.size()) {
      size_t bang = pipeline.find('!', start);
      if (bang == std::string::npos) {
        parts.push_back(pipeline.substr(start));
        break;
      }
      parts.push_back(pipeline.substr(start, bang - start));
      start = bang + 1;
    }
    for (auto& part : parts) {
      std::string seg = part;
      trim(seg);
      if (seg.find("neatprocessmla") == std::string::npos)
        continue;
      force_bool_prop(seg, "multi-pipeline=", false);
      part = " " + seg + " ";
    }
    std::ostringstream out;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i)
        out << " ! ";
      std::string seg = parts[i];
      trim(seg);
      out << seg;
    }
    pipeline = out.str();
  }

  bool force_boxdecode = (num_buffers_override > 0);
  if (!force_boxdecode) {
    const int max_buffers = max_num_buffers_in_pipeline_strict(pipeline);
    force_boxdecode = (max_buffers <= 2);
  }
  if (force_boxdecode) {
    apply_num_buffers(
        [](const std::string& seg) { return seg.find("neatboxdecode") != std::string::npos; });
  }

  if (forced <= 1) {
    pipeline = clamp_terminal_appsink_num_buffers(std::move(pipeline), num_buffers_override);
  }

  return pipeline;
}

std::string clamp_detess_num_buffers(std::string pipeline, int num_buffers_override) {
  // debugging resnet50: Sync detess stages can deadlock when num-buffers=1 and the
  // previous output sample lifetime overlaps the next push/pull call. Keep at least 2.
  const int forced_min = (num_buffers_override > 0) ? std::max(2, num_buffers_override) : 2;

  auto trim = [](std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
      --end;
    }
    s = s.substr(start, end - start);
  };

  auto to_lower = [](std::string s) {
    for (char& c : s) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
  };

  // Third copy of the num-buffers force-lambda in the original; consolidated
  // through gst_props::set_prop like the others above.

  std::vector<std::string> parts;
  size_t start = 0;
  while (start < pipeline.size()) {
    const size_t bang = pipeline.find('!', start);
    if (bang == std::string::npos) {
      parts.push_back(pipeline.substr(start));
      break;
    }
    parts.push_back(pipeline.substr(start, bang - start));
    start = bang + 1;
  }

  bool changed = false;
  for (auto& part : parts) {
    std::string seg = part;
    trim(seg);
    const std::string lower = to_lower(seg);
    const bool is_processcvu = lower.find("neatprocesscvu") != std::string::npos;
    const bool is_named_detess = lower.find("detess") != std::string::npos;
    const bool is_fused_detessdequant =
        lower.find("stage-id=dequantize_") != std::string::npos ||
        lower.find("name=dequantize_") != std::string::npos ||
        lower.find("stage-id=detessdequant_") != std::string::npos ||
        lower.find("name=detessdequant_") != std::string::npos;
    const bool is_detess_processcvu = is_processcvu && (is_named_detess || is_fused_detessdequant);
    if (!is_detess_processcvu) {
      continue;
    }
    // Skip only when the current value is set AND already meets the floor. Absent /
    // empty / non-numeric all fall through and get the forced value — same semantic
    // as before, but no -1 sentinel conflation.
    if (const auto cur = parse_num_buffers_in_segment(seg); cur.has_value() && *cur >= forced_min) {
      continue;
    }
    gst_props::set_prop(seg, "num-buffers=", std::to_string(forced_min));
    part = " " + seg + " ";
    changed = true;
  }

  if (!changed) {
    return pipeline;
  }

  std::ostringstream out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) {
      out << " ! ";
    }
    std::string seg = parts[i];
    trim(seg);
    out << seg;
  }

  if (env_bool("SIMA_MLA_NUM_BUFFERS_DEBUG", false)) {
    std::fprintf(stderr, "[DBG] Graph::build(input) clamp_detess_num_buffers min=%d\n", forced_min);
  }
  return out.str();
}

BuildResult build_pipeline_full(const std::vector<std::shared_ptr<Node>>& nodes,
                                bool insert_boundaries, const std::string& appsink_name,
                                bool insert_queue2, const NameTransform& name_transform,
                                const GraphOptions* sess_opt) {
  const pipeline_internal::ScopedBuildTiming timing(
      "build_pipeline_full", "nodes=" + std::to_string(nodes.size()) +
                                 " insert_boundaries=" + std::to_string(insert_boundaries ? 1 : 0) +
                                 " queue2=" + std::to_string(insert_queue2 ? 1 : 0));
  if (nodes.empty()) {
    throw_session_error_simple(error_codes::kPipelineShape, "InvalidPipeline: no nodes");
  }

  BuildResult br;
  br.diag = std::make_shared<DiagCtx>();
  br.appsink_name = apply_name_transform(name_transform, appsink_name);
  br.name_transform = name_transform;
  br.diag->queue2_enabled = insert_queue2;
  const int requested_queue_depth = sess_opt ? sess_opt->async_queue_depth : 0;
  br.diag->queue2_depth = insert_queue2 ? async_queue2_depth(requested_queue_depth) : 0;

  pipeline_internal::PipelineBuildContext build_ctx(name_transform);
  dump_node_debug_options(nodes, name_transform);

  std::ostringstream ss;

  br.diag->node_reports.reserve(nodes.size());
  if (insert_boundaries) {
    br.diag->boundaries.reserve(nodes.size() ? nodes.size() - 1 : 0);
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i]) {
      throw_session_error_simple(error_codes::kPipelineShape, "InvalidPipeline: null node");
    }

    if (i) {
      bool want_queue2 = insert_queue2;
      if (insert_queue2 && nodes[i] && nodes[i]->kind() == "Output") {
        want_queue2 = false;
      }
      if (want_queue2) {
        ss << " ! " << async_queue2_fragment(requested_queue_depth) << " ! ";
      } else {
        ss << " ! ";
      }
    }

    NodeReport nr;
    nr.index = (int)i;
    nr.kind = nodes[i]->kind();
    nr.user_label = nodes[i]->user_label();
    NodeFragment frag = make_node_fragment(nodes[i], (int)i, name_transform);
    nr.backend_fragment = apply_session_fast_path_options_to_fragment(frag.fragment, sess_opt);
    nr.elements = frag.element_names;
    br.diag->node_reports.push_back(nr);

    ss << nr.backend_fragment;

    if (insert_boundaries && i + 1 < nodes.size()) {
      const std::string bname = apply_name_transform(name_transform, "sima_b" + std::to_string(i));
      ss << " ! identity name=" << bname << " silent=true";

      // IMPORTANT: boundaries store atomic counters (BoundaryFlowCounters),
      // not plain stats. Reports use snapshot() to convert atomics -> stats.
      auto ctr = std::make_unique<pipeline_internal::BoundaryFlowCounters>();
      ctr->boundary_name = bname;
      ctr->after_node_index = (int)i;
      ctr->before_node_index = (int)(i + 1);
      br.diag->boundaries.push_back(std::move(ctr));
    }
  }

  br.diag->pipeline_string = ss.str();
  br.pipeline_string = br.diag->pipeline_string;
  return br;
}

std::vector<std::shared_ptr<Node>>
session_build_materialize_model_bound_nodes(const std::vector<std::shared_ptr<Node>>& nodes,
                                            bool sync_mode) {
  struct RequestedPostState {
    internal::ModelLineageBinding binding;
    BoxDecodeType decode_type = BoxDecodeType::Unspecified;
  };

  std::unordered_map<std::string, RequestedPostState> requested_posts;
  std::unordered_map<std::string, MaterializedLineageState> effective_lineages;

  for (const auto& node : nodes) {
    const auto* binding = node_model_lineage_binding(node);
    if (!binding || binding->requested_post == internal::RequestedPostRouteKind::Auto) {
      continue;
    }
    if (!internal::requested_post_route_supported(binding->requested_post)) {
      throw std::runtime_error("Graph::build: unsupported model-bound post request '" +
                               internal::requested_post_route_name(binding->requested_post) + "'");
    }
    BoxDecodeType requested_decode_type = BoxDecodeType::Unspecified;
    if (auto* box = dynamic_cast<SimaBoxDecode*>(node.get())) {
      requested_decode_type = box->decode_type_internal();
    }
    const auto it = requested_posts.find(binding->lineage_key);
    if (it != requested_posts.end()) {
      throw std::runtime_error("Graph::build: multiple manual post nodes for model lineage '" +
                               binding->lineage_key + "' are not supported");
    }
    requested_posts.emplace(binding->lineage_key,
                            RequestedPostState{*binding, requested_decode_type});
  }

  if (requested_posts.empty()) {
    return nodes;
  }

  for (const auto& [lineage_key, request] : requested_posts) {
    MaterializedLineageState state;
    std::string err;
    state.effective_model = internal::build_effective_model_for_requested_post(
        request.binding, request.decode_type, &state.changed, &err);
    if (!state.effective_model) {
      throw std::runtime_error("Graph::build: failed to retarget model lineage '" + lineage_key +
                               "': " + (err.empty() ? std::string("unknown error") : err));
    }
    effective_lineages.emplace(lineage_key, std::move(state));
  }

  std::vector<std::shared_ptr<Node>> out;
  out.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size();) {
    const auto* binding = node_model_lineage_binding(nodes[i]);
    if (!binding) {
      out.push_back(nodes[i]);
      ++i;
      continue;
    }
    const auto it = effective_lineages.find(binding->lineage_key);
    if (it == effective_lineages.end() || !it->second.changed) {
      out.push_back(nodes[i]);
      ++i;
      continue;
    }

    if (binding->stage_role == internal::ModelLineageStageRole::ManualPost) {
      if (auto* box = dynamic_cast<SimaBoxDecode*>(nodes[i].get())) {
        const auto route_flags = box->model_route_flags_internal();
        const auto route_tess_needed =
            route_flags.has_value() ? std::optional<bool>(route_flags->tess_needed) : std::nullopt;
        const auto route_quant_needed =
            route_flags.has_value() ? std::optional<bool>(route_flags->quant_needed) : std::nullopt;
        out.push_back(nodes::SimaBoxDecode(
            *it->second.effective_model, box->decode_type_internal(),
            box->detection_threshold_internal(), box->nms_iou_threshold_internal(),
            box->top_k_internal(),
            box->element_names(0).empty() ? std::string() : box->element_names(0).front(),
            route_tess_needed, route_quant_needed, box->original_width_internal(),
            box->original_height_internal(), /*model_width=*/0, /*model_height=*/0,
            /*resize_mode_override=*/std::nullopt, box->decode_type_option_internal()));
        ++i;
        continue;
      }
      out.push_back(nodes[i]);
      ++i;
      continue;
    }

    std::size_t j = i + 1U;
    while (j < nodes.size()) {
      const auto* next = node_model_lineage_binding(nodes[j]);
      if (!next || next->lineage_key != binding->lineage_key ||
          next->stage_role != binding->stage_role) {
        break;
      }
      ++j;
    }

    std::vector<std::shared_ptr<Node>> replacement;
    switch (binding->stage_role) {
    case internal::ModelLineageStageRole::Preprocess:
      replacement =
          internal::ModelAccess::build_preprocess_nodes(*it->second.effective_model, sync_mode);
      break;
    case internal::ModelLineageStageRole::Infer:
      replacement =
          internal::ModelAccess::build_infer_nodes(*it->second.effective_model, sync_mode);
      break;
    case internal::ModelLineageStageRole::ManualPost:
      break;
    }
    for (const auto& replacement_node : replacement) {
      out.push_back(replacement_node);
    }
    i = j;
  }

  return out;
}

void session_build_compile_contracts(BuildResult* build_result,
                                     const std::vector<std::shared_ptr<Node>>& source_nodes,
                                     const ContractCompileInput& compile_input, const char* where,
                                     std::vector<std::shared_ptr<Node>>* apply_nodes) {
  if (!build_result) {
    return;
  }

  const auto total_start = pipeline_internal::build_timing_now();
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compile_start = pipeline_internal::build_timing_now();
  auto compiled = std::make_shared<CompiledPipelineContracts>(
      compile_node_contracts(source_nodes, compile_input, &diagnostics));
  const auto compile_us = pipeline_internal::build_timing_us(compile_start);

  std::int64_t apply_us = 0;
  if (apply_nodes) {
    const auto apply_start = pipeline_internal::build_timing_now();
    std::string apply_error;
    apply_compiled_contracts(apply_nodes, *compiled, &apply_error);
    apply_us = pipeline_internal::build_timing_us(apply_start);
    if (!apply_error.empty()) {
      throw std::runtime_error(std::string(where ? where : "Graph::build") + ": " + apply_error);
    }
  }

  build_result->compiled_contracts = compiled;
  const auto render_start = pipeline_internal::build_timing_now();
  build_result->rendered_manifest =
      render_manifest_from_compiled_contracts(*compiled, compile_input, &diagnostics);
  const auto render_us = pipeline_internal::build_timing_us(render_start);
  // Carry forward compile + render diagnostics so the wrapper throws in
  // parse_pipeline_or_throw can surface the actual stage-level failure
  // messages instead of "manifest is missing" with no further detail.
  build_result->manifest_diagnostics = std::move(diagnostics);
  build_result->model_source_paths.clear();
  for (const auto& node : source_nodes) {
    const auto* binding = node_model_lineage_binding(node);
    if (!binding || binding->source_path.empty()) {
      continue;
    }
    if (std::find(build_result->model_source_paths.begin(), build_result->model_source_paths.end(),
                  binding->source_path) == build_result->model_source_paths.end()) {
      build_result->model_source_paths.push_back(binding->source_path);
    }
  }
  pipeline_internal::emit_build_timing(
      "session_build_compile_contracts",
      {{"compile", compile_us},
       {"apply", apply_us},
       {"render_manifest", render_us},
       {"total", pipeline_internal::build_timing_us(total_start)}},
      std::string("where=") + (where ? where : "Graph::build") +
          " nodes=" + std::to_string(source_nodes.size()) + " rendered_manifest=" +
          std::to_string(build_result->rendered_manifest.has_value() ? 1 : 0));
}

void enforce_sink_last(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (nodes.empty()) {
    throw_session_error_simple(error_codes::kPipelineShape, "InvalidPipeline: no nodes");
  }
  if (!nodes.back() || nodes.back()->kind() != "Output") {
    throw_session_error_simple(error_codes::kPipelineShape,
                               "InvalidPipeline: last node must be Output() for run()");
  }
}

void enforce_caps_behavior(const std::vector<std::shared_ptr<Node>>& nodes,
                           const std::string& where) {
  for (const auto& node : nodes) {
    if (!node) {
      throw_session_error_simple(error_codes::kPipelineShape, where + ": node is null");
    }
    (void)node->caps_behavior();
  }
}

static int find_output_appsink_index(const std::vector<std::shared_ptr<Node>>& nodes) {
  int found = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    if (nodes[i]->kind() != "Output")
      continue;
    if (found >= 0) {
      throw_session_error_simple(error_codes::kPipelineShape,
                                 "InvalidPipeline: multiple Output nodes found");
    }
    found = static_cast<int>(i);
  }
  return found;
}

bool has_output_appsink(const std::vector<std::shared_ptr<Node>>& nodes) {
  return find_output_appsink_index(nodes) >= 0;
}

namespace {

struct AppSinkAllocPref {
  std::uint64_t mem_type = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_GENERIC);
  std::uint64_t mem_flag = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
};

static const char* simaai_mem_type_to_str(std::uint64_t mem_type) {
  switch (mem_type) {
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_GENERIC):
    return "GST_SIMAAI_MEMORY_TARGET_GENERIC";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_EV74):
    return "GST_SIMAAI_MEMORY_TARGET_EV74";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS0):
    return "GST_SIMAAI_MEMORY_TARGET_DMS0";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS1):
    return "GST_SIMAAI_MEMORY_TARGET_DMS1";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS2):
    return "GST_SIMAAI_MEMORY_TARGET_DMS2";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS3):
    return "GST_SIMAAI_MEMORY_TARGET_DMS3";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_OCM):
    return "GST_SIMAAI_MEMORY_TARGET_OCM";
  default:
    return "GST_SIMAAI_MEMORY_TARGET_GENERIC";
  }
}

static const char* simaai_mem_flag_to_str(std::uint64_t mem_flag) {
  switch (mem_flag) {
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED):
    return "GST_SIMAAI_MEMORY_FLAG_CACHED";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_RDONLY):
    return "GST_SIMAAI_MEMORY_FLAG_RDONLY";
  case static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_DEFAULT):
    return "GST_SIMAAI_MEMORY_FLAG_DEFAULT";
  default:
    return "GST_SIMAAI_MEMORY_FLAG_DEFAULT";
  }
}

static GType simaai_allocation_meta_api_type() {
  static std::atomic<GType> cached{0};
  GType type = cached.load(std::memory_order_acquire);
  if (type != 0)
    return type;

  type = g_type_from_name("GstSimaaiAllocationMetaAPI");
  if (type == 0) {
    cached.store(0, std::memory_order_release);
    return 0;
  }
  cached.store(type, std::memory_order_release);
  return type;
}

static GstStructure* make_simaai_allocation_meta(std::uint64_t mem_type, std::uint64_t mem_flag) {
  return gst_structure_new("simaai-allocation-meta", "memory_type", G_TYPE_STRING,
                           simaai_mem_type_to_str(mem_type), "memory_flag", G_TYPE_STRING,
                           simaai_mem_flag_to_str(mem_flag), NULL);
}

static GstPadProbeReturn appsink_allocation_probe(GstPad* /*pad*/, GstPadProbeInfo* info,
                                                  gpointer user_data) {
  if (!info || !(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM)) {
    return GST_PAD_PROBE_OK;
  }
  GstQuery* query = GST_PAD_PROBE_INFO_QUERY(info);
  if (!query || GST_QUERY_TYPE(query) != GST_QUERY_ALLOCATION) {
    return GST_PAD_PROBE_OK;
  }

  auto* pref = static_cast<AppSinkAllocPref*>(user_data);
  if (!pref)
    return GST_PAD_PROBE_OK;

  const GType api_type = simaai_allocation_meta_api_type();
  if (api_type == 0)
    return GST_PAD_PROBE_OK;

  guint idx = 0;
  if (!gst_query_find_allocation_meta(query, api_type, &idx)) {
    GstStructure* meta = make_simaai_allocation_meta(pref->mem_type, pref->mem_flag);
    if (meta) {
      gst_query_add_allocation_meta(query, api_type, meta);
    }
  }

  return GST_PAD_PROBE_OK;
}

static bool infer_appsink_alloc_pref(const std::vector<std::shared_ptr<Node>>& nodes,
                                     std::uint64_t* mem_type_out, std::uint64_t* mem_flag_out) {
  if (!mem_type_out || !mem_flag_out)
    return false;
  const int sink_idx = find_output_appsink_index(nodes);
  if (sink_idx < 0)
    return false;

  if (env_bool("SIMA_APPSINK_FORCE_DMS0", false)) {
    *mem_type_out = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS0);
    *mem_flag_out = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
    return true;
  }

  // Appsink is a CPU-facing consumer; request generic memory to ensure
  // the sink can map buffers even when upstream runs on CVU/MLA.
  *mem_type_out = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_GENERIC);
  *mem_flag_out = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
  return true;
}

void configure_appsink_allocation_preference(GstElement* appsink,
                                             const std::vector<std::shared_ptr<Node>>& nodes) {
  if (!appsink)
    return;
  std::uint64_t mem_type = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_GENERIC);
  std::uint64_t mem_flag = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
  if (!infer_appsink_alloc_pref(nodes, &mem_type, &mem_flag))
    return;

  GstPad* pad = gst_element_get_static_pad(appsink, "sink");
  if (!pad)
    return;

  auto* pref = g_new0(AppSinkAllocPref, 1);
  pref->mem_type = mem_type;
  pref->mem_flag = mem_flag;
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM, appsink_allocation_probe, pref,
                    g_free);
  gst_object_unref(pad);
}

} // namespace

void session_build_configure_appsink_allocation_preference_internal(
    GstElement* appsink, const std::vector<std::shared_ptr<Node>>& nodes) {
  configure_appsink_allocation_preference(appsink, nodes);
}

void enforce_sink_last_if_present(const std::vector<std::shared_ptr<Node>>& nodes,
                                  const std::string& where) {
  if (nodes.empty()) {
    throw_session_error_simple(error_codes::kPipelineShape, where + ": no nodes");
  }
  const int sink_idx = find_output_appsink_index(nodes);
  if (sink_idx < 0)
    return;
  if (sink_idx != static_cast<int>(nodes.size() - 1)) {
    throw_session_error_simple(error_codes::kPipelineShape,
                               where + ": Output() must be the last node");
  }
}

static const Input* find_input_appsrc(const std::vector<std::shared_ptr<Node>>& nodes,
                                      int* index_out) {
  const Input* found = nullptr;
  int found_idx = -1;

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    if (auto* src = dynamic_cast<Input*>(nodes[i].get())) {
      if (found) {
        throw_session_error_simple(error_codes::kPipelineShape,
                                   "InvalidPipeline: multiple Input nodes found");
      }
      found = src;
      found_idx = static_cast<int>(i);
    }
  }

  if (index_out)
    *index_out = found_idx;
  return found;
}

static bool has_input_role(const std::vector<std::shared_ptr<Node>>& nodes, InputRole role) {
  for (const auto& n : nodes) {
    if (!n)
      continue;
    if (n->input_role() == role)
      return true;
  }
  return false;
}

void enforce_source_run_mode(const std::vector<std::shared_ptr<Node>>& nodes,
                             const std::string& where) {
  if (has_input_role(nodes, InputRole::Push)) {
    throw_session_error_simple(error_codes::kPipelineShape,
                               where + ": Input is present; use run(input) for push pipelines.");
  }
  if (!has_input_role(nodes, InputRole::Source)) {
    throw_session_error_simple(
        error_codes::kPipelineShape,
        where + ": no source input found; add a source node/group (RTSP/File/Image/Video).");
  }
}

void enforce_push_run_mode(const std::vector<std::shared_ptr<Node>>& nodes,
                           const std::string& where) {
  if (has_input_role(nodes, InputRole::Source)) {
    throw_session_error_simple(error_codes::kPipelineShape,
                               where +
                                   ": source input is present; use run() for source pipelines.");
  }
  if (!has_input_role(nodes, InputRole::Push)) {
    throw_session_error_simple(error_codes::kPipelineShape,
                               where + ": missing Input; use run() or add Input.");
  }
}

void throw_if_input_appsrc_present(const std::vector<std::shared_ptr<Node>>& nodes,
                                   const std::string& where) {
  int idx = -1;
  const Input* src = find_input_appsrc(nodes, &idx);
  if (!src)
    return;
  throw_session_error_simple(error_codes::kPipelineShape,
                             where + ": Input() is present; use the input overload (run(input), "
                                     "validate(opt, input)).");
}

void require_input_appsrc(const std::vector<std::shared_ptr<Node>>& nodes, const std::string& where,
                          const Input** out_src) {
  int idx = -1;
  const Input* src = find_input_appsrc(nodes, &idx);
  if (!src) {
    throw_session_error_simple(error_codes::kPipelineShape, where + ": missing Input() node");
  }
  if (idx != 0) {
    throw_session_error_simple(error_codes::kPipelineShape,
                               where + ": Input() must be the first node");
  }
  if (out_src)
    *out_src = src;
}

static int max_num_buffers_in_pipeline_strict(const std::string& pipeline) {
  const std::string key = "num-buffers=";
  int max_val = 0;
  size_t pos = 0;
  while ((pos = pipeline.find(key, pos)) != std::string::npos) {
    pos += key.size();
    int value = 0;
    bool found = false;
    while (pos < pipeline.size() && std::isdigit(static_cast<unsigned char>(pipeline[pos]))) {
      found = true;
      value = value * 10 + (pipeline[pos] - '0');
      ++pos;
    }
    if (found)
      max_val = std::max(max_val, value);
  }
  return max_val;
}

InputOptions resolve_appsrc_options(const InputOptions& opt, const NameTransform& name_transform) {
  InputOptions out = opt;
  // Keep logical input buffer groups stable.
  // Per-stage JSON name rewriting is disabled, so transforming this field can
  // desynchronize appsrc metadata ("decoder_1") from plugin config ("decoder").
  (void)name_transform;
  return out;
}

void configure_appsrc(GstElement* appsrc, const InputOptions& opt) {
  if (!appsrc)
    return;
  // Phase D-quick: SIMA_APPSRC_NOT_LIVE=1 forces is-live=false.  The default
  // streaming-source mode paces buffers by the GST clock, which adds queue
  // wait at queue2-0 between appsrc and the first kernel (~6 ms at bs=1, the
  // entire non-kernel overhead surfaced by Phase A per-pad probes).  Setting
  // this knob lets benchmarks/non-streaming use cases run as fast as the
  // consumer can pull, eliminating that wait.
  bool effective_is_live = opt.is_live;
  bool effective_do_timestamp = opt.do_timestamp;
  if (const char* env = std::getenv("SIMA_APPSRC_NOT_LIVE");
      env && *env && std::strcmp(env, "0") != 0) {
    effective_is_live = false;
    effective_do_timestamp = false;
  }
  g_object_set(G_OBJECT(appsrc), "is-live", effective_is_live ? TRUE : FALSE, "format",
               GST_FORMAT_TIME, "do-timestamp", effective_do_timestamp ? TRUE : FALSE, "block",
               opt.block ? TRUE : FALSE, "stream-type", opt.stream_type, "max-bytes",
               static_cast<guint64>(opt.max_bytes), nullptr);
}

static std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
  if (a == 0 || b == 0)
    return 0;
  if (a > (std::numeric_limits<std::uint64_t>::max() / b))
    return 0;
  return a * b;
}

static int tensor_dtype_bytes_from_format(std::string fmt) {
  fmt = lower_copy(fmt);
  if (fmt.find("int8") != std::string::npos || fmt.find("uint8") != std::string::npos)
    return 1;
  if (fmt.find("int16") != std::string::npos || fmt.find("uint16") != std::string::npos ||
      fmt.find("bf16") != std::string::npos || fmt.find("bfloat16") != std::string::npos ||
      fmt.find("fp16") != std::string::npos || fmt.find("float16") != std::string::npos) {
    return 2;
  }
  if (fmt.find("int32") != std::string::npos || fmt.find("fp32") != std::string::npos ||
      fmt.find("float32") != std::string::npos)
    return 4;
  if (fmt.find("fp64") != std::string::npos)
    return 8;
  return 1;
}

std::uint64_t estimate_frame_bytes_limit(const InputOptions& opt, const SampleSpec& spec) {
  if (spec.tensor_envelope_transport) {
    return static_cast<std::uint64_t>(spec.required_bytes_actual);
  }
  const InputOptions normalized = pipeline_internal::normalize_shape_bounds(opt);
  const auto limits = pipeline_internal::resolve_shape_limits(normalized, spec);
  const std::string normalized_media = resolve_input_media_type(normalized);
  const std::string media =
      lower_copy(normalized_media.empty() ? spec.media_type : normalized_media);
  const std::string fmt =
      lower_copy(normalized.format.empty() ? spec.format : normalized.format.str());
  std::uint64_t bytes = 0;
  if (media == "video/x-raw") {
    const int w = (limits.max_width > 0) ? limits.max_width : spec.width;
    const int h = (limits.max_height > 0) ? limits.max_height : spec.height;
    const int d = (limits.max_depth > 0) ? limits.max_depth : (spec.depth > 0 ? spec.depth : 1);
    if (w <= 0 || h <= 0) {
      return static_cast<std::uint64_t>(spec.required_bytes_actual);
    }
    if (fmt == "nv12" || fmt == "i420" || fmt == "iyuv") {
      // 1.5 bytes per pixel.
      const std::uint64_t area =
          checked_mul_u64(static_cast<std::uint64_t>(w), static_cast<std::uint64_t>(h));
      bytes = (area > 0) ? checked_mul_u64(area, 3) / 2 : 0;
    } else if (fmt == "gray8" || fmt == "gray") {
      bytes = checked_mul_u64(static_cast<std::uint64_t>(w), static_cast<std::uint64_t>(h));
    } else {
      const int depth = (d > 0) ? d : 3;
      bytes = checked_mul_u64(
          checked_mul_u64(static_cast<std::uint64_t>(w), static_cast<std::uint64_t>(h)),
          static_cast<std::uint64_t>(depth));
    }
  } else if (media == "application/vnd.simaai.tensor") {
    const int elem = tensor_dtype_bytes_from_format(fmt);
    if (elem > 0 && !spec.shape.empty()) {
      const std::uint64_t elems = checked_tensor_shape_elements(spec.shape);
      if (elems > 0U) {
        bytes = checked_mul_u64(elems, static_cast<std::uint64_t>(elem));
      }
    }
  }
  if (bytes == 0) {
    bytes = static_cast<std::uint64_t>(spec.required_bytes_actual);
  }
  // The graph/session guard must never shrink below the actual logical sample size.
  // Tensor byte estimates now use the full semantic shape directly; still clamp to the
  // real contiguous byte count so dynamic/sample-envelope paths remain safe.
  if (spec.required_bytes_actual > 0U) {
    bytes = std::max(bytes, static_cast<std::uint64_t>(spec.required_bytes_actual));
  }
  return bytes;
}

std::uint64_t resolve_appsrc_max_bytes(const InputOptions& opt, const SampleSpec& spec) {
  if (opt.max_bytes > 0)
    return opt.max_bytes;
  const int default_buffers = std::max(0, env_int("SIMA_APPSRC_DEFAULT_MAX_BUFFERS", 4));
  if (default_buffers <= 0)
    return 0;
  const std::uint64_t bytes = estimate_frame_bytes_limit(opt, spec);
  if (bytes == 0)
    return 0;
  std::uint64_t max_bytes = 0;
  if (bytes > 0) {
    if (bytes >
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(default_buffers)) {
      return 0;
    }
    max_bytes = bytes * static_cast<std::uint64_t>(default_buffers);
  }
  return max_bytes;
}

SampleSpec make_placeholder_spec() {
  SampleSpec spec;
  spec.kind = SampleMediaKind::Encoded;
  spec.caps_string = "application/octet-stream";
  spec.caps_key = capkey_from_spec(spec);
  return spec;
}

void configure_appsink_for_input(GstElement* appsink) {
  if (!appsink)
    return;
  g_object_set(G_OBJECT(appsink), "emit-signals", FALSE, "max-buffers", 4, "drop", FALSE, "sync",
               TRUE, "enable-last-sample", FALSE, "qos", FALSE, nullptr);
}

void configure_appsink_for_input_stream(GstElement* appsink, const InputStreamOptions& opt) {
  configure_appsink_for_input(appsink);
  if (!appsink)
    return;
  g_object_set(G_OBJECT(appsink), "max-buffers", opt.appsink_max_buffers, "drop",
               opt.appsink_drop ? TRUE : FALSE, "sync", opt.appsink_sync ? TRUE : FALSE, nullptr);
}

// Best-effort naming contract enforcement; optional via env.
void enforce_names_contract(GstElement* pipeline, const BuildResult& br) {
  if (!pipeline || !GST_IS_BIN(pipeline) || !br.diag)
    return;

  std::unordered_set<std::string> allowed;

  for (const auto& n : br.diag->node_reports) {
    for (const auto& e : n.elements) {
      allowed.insert(e);
    }
  }
  for (const auto& b : br.diag->boundaries) {
    if (b)
      allowed.insert(b->boundary_name);
  }
  if (!br.appsink_name.empty()) {
    allowed.insert(br.appsink_name);
  }

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it)
    return;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* el = GST_ELEMENT(g_value_get_object(&item));
    g_value_reset(&item);
    if (!el)
      continue;

    const char* name = GST_ELEMENT_NAME(el);
    if (name && *name) {
      const std::string n(name);
      const bool internal_ok = (n.rfind("queue", 0) == 0) || (n.rfind("typefind", 0) == 0) ||
                               (n.rfind("rtpbin", 0) == 0) || (n.rfind("decodebin", 0) == 0);

      if (!internal_ok && allowed.find(n) == allowed.end()) {
        gst_iterator_free(it);
        GraphReport rep = br.diag->snapshot_basic();
        rep.error_code = error_codes::kPipelineShape;
        rep.repro_note = "NamingContractViolation: element '" + n +
                         "' is not owned by any node.\n"
                         "Fix: ensure every fragment uses deterministic names and "
                         "element_names() matches.\n";
        throw NeatError(decorate_with_error_code(rep.error_code, rep.repro_note), std::move(rep));
      }
    }
  }

  g_value_unset(&item);
  gst_iterator_free(it);
}

} // namespace simaai::neat

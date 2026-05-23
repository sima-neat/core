// src/pipeline/internal/InputStream.cpp
#include "InputStream.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/EncodedSampleUtil.h"
#include "InputStreamUtil.h"

#include "pipeline/internal/CapsBridge.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/Input.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace simaai::neat {
using pipeline_internal::trim_copy;
using pipeline_internal::upper_copy;

using pipeline_internal::DiagCtx;

struct InputStream::State {
  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  GstElement* appsink = nullptr;
  InputOptions src_opt;
  InputStreamOptions opt;
  InputStreamOptions::DynamicCapability dynamic_capability =
      InputStreamOptions::DynamicCapability::StaticOnly;
  InputStreamOptions::ShapePolicy shape_policy = InputStreamOptions::ShapePolicy::BoundedDynamic;
  InputStreamOptions::ResolvedShapeLimits shape_limits{};
  InputStreamOptions::ByteGuardOrigin byte_guard_origin =
      InputStreamOptions::ByteGuardOrigin::Unset;
  bool allow_ingress_cvu_format_renegotiation = false;
  bool allow_dynamic_growth = false;
  std::size_t max_input_bytes_guard = 0;
  InputBufferPoolGuard pool_guard;
  GstBuffer* reusable_buffer = nullptr;
  size_t reusable_bytes = 0;
  std::shared_ptr<DiagCtx> diag;
  std::shared_ptr<void> guard;
  std::thread worker;
  std::mutex pipeline_mu;
  std::atomic<bool> running{false};
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> worker_done{true};
  std::atomic<bool> teardown_on_exit{false};
  bool use_callbacks = false;
  std::mutex cb_mu;
  std::condition_variable cb_cv;
  std::deque<GstSample*> cb_queue;
  std::size_t cb_queue_max = 0;
  std::atomic<bool> cb_eos{false};
  std::atomic<int> cb_inflight{0};
  GstAppSinkCallbacks cb_handlers{};
  struct CallbackCtx {
    std::shared_ptr<InputStream::State> st;
  };
  std::unique_ptr<CallbackCtx> cb_ctx;
  std::function<void(Sample)> callback;
  mutable std::mutex error_mu;
  std::string error;
  std::optional<CapKey> current_key;
  std::optional<CapKey> pending_key;
  int pending_count = 0;
  GstBuffer* pending_buffer = nullptr;
  std::optional<SampleSpec> pending_spec;
  std::uint64_t pending_alloc_ns = 0;
  std::uint64_t pending_map_ns = 0;
  std::uint64_t pending_copy_ns = 0;
  std::optional<SampleSpec> last_spec;
  std::function<void(const SampleSpec&, const SampleSpec&)> on_caps_change;
  std::optional<SampleSpec> current_tensor_spec;
  size_t alloc_bytes = 0;
  bool timing_enabled = false;
  GstCaps* current_caps = nullptr;
  GstVideoInfo current_vinfo{};
  std::atomic<std::uint64_t> dropped_frames{0};
  std::atomic<std::uint64_t> renegotiations{0};
  std::atomic<std::uint64_t> alloc_grows{0};
  std::atomic<std::uint64_t> growth_blocked{0};
  std::atomic<std::uint64_t> renegotiation_blocked{0};
  std::atomic<std::uint64_t> push_count{0};
  std::atomic<std::uint64_t> push_failures{0};
  std::atomic<std::uint64_t> pull_count{0};
  std::atomic<std::uint64_t> poll_count{0};
  std::atomic<std::uint64_t> alloc_ns{0};
  std::atomic<std::uint64_t> map_ns{0};
  std::atomic<std::uint64_t> copy_ns{0};
  std::atomic<std::uint64_t> push_ns{0};
  std::atomic<std::uint64_t> pull_wait_ns{0};
  std::atomic<std::uint64_t> decode_ns{0};
  std::atomic<std::int64_t> last_push_ns{0};
  std::atomic<std::int64_t> inflight{0};
  std::atomic<std::int64_t> next_input_seq{0};
  std::mutex preprocess_meta_mu;
  std::deque<std::int64_t> preprocess_meta_order;
  std::unordered_map<std::int64_t, PreprocessRuntimeMeta> preprocess_meta_by_input_seq;
  std::atomic<bool> eos_sent{false};
  std::atomic<bool> teardown_started{false};
  std::mutex stop_mu;
  std::atomic<bool> stop_started{false};
  std::atomic<int> gst_calls{0};
  std::mutex gst_mu;
  std::condition_variable gst_cv;
};

struct GstCallGuard {
  InputStream::State& st;
  explicit GstCallGuard(InputStream::State& state) : st(state) {
    st.gst_calls.fetch_add(1, std::memory_order_relaxed);
  }
  ~GstCallGuard() {
    if (st.gst_calls.fetch_sub(1, std::memory_order_relaxed) == 1) {
      std::lock_guard<std::mutex> lock(st.gst_mu);
      st.gst_cv.notify_all();
    }
  }
};

struct CallbackInflightGuard {
  std::shared_ptr<InputStream::State> st;
  explicit CallbackInflightGuard(const std::shared_ptr<InputStream::State>& state) : st(state) {
    if (st) {
      st->cb_inflight.fetch_add(1, std::memory_order_relaxed);
    }
  }
  ~CallbackInflightGuard() {
    if (!st)
      return;
    if (st->cb_inflight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(st->cb_mu);
      st->cb_cv.notify_all();
    }
  }
};

bool buffer_name_matches_expected(const std::string& expected_list, const std::string& actual) {
  const std::string expected = trim_copy(expected_list);
  if (expected.empty())
    return actual.empty();
  if (expected.find(',') == std::string::npos) {
    return expected == actual;
  }
  size_t start = 0;
  while (start < expected.size()) {
    size_t end = expected.find(',', start);
    if (end == std::string::npos)
      end = expected.size();
    const std::string item = trim_copy(expected.substr(start, end - start));
    if (!item.empty() && item == actual)
      return true;
    start = end + 1;
  }
  return false;
}

struct InputStreamDebugFlags {
  bool inputstream_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_DEBUG", false);
  bool pipeline_debug = pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false);
  bool graph_debug = pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false);
  bool holder_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_HOLDER_DEBUG", false);
  bool stop_trace = pipeline_internal::env_bool("SIMA_STOP_TRACE", false);
  bool push_ref_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_PUSH_REF_DEBUG", false);
  bool unref_on_push_fail =
      pipeline_internal::env_bool("SIMA_INPUTSTREAM_UNREF_ON_PUSH_FAIL", false);
  bool push_fail_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_PUSH_FAIL_DEBUG", false);
  bool push_fail_detail = pipeline_internal::env_bool("SIMA_INPUTSTREAM_PUSH_FAIL_DETAIL", false);
  bool drop_holder_after_push =
      pipeline_internal::env_bool("SIMA_INPUTSTREAM_DROP_HOLDER_AFTER_PUSH", true);
  bool eos_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_EOS_DEBUG", false);
  bool appsink_cb_debug = pipeline_internal::env_bool("SIMA_APPSINK_CB_DEBUG", false);
  bool appsink_drop_last_debug = pipeline_internal::env_bool("SIMA_APPSINK_DROP_LAST_DEBUG", false);
  bool inputstream_meta_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false);
  bool inputstream_pool_debug = pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false);
  bool dot_on_timeout = pipeline_internal::env_bool("SIMA_INPUTSTREAM_DOT_ON_TIMEOUT", false);
  bool use_appsink_callbacks =
      pipeline_internal::env_bool("SIMA_INPUTSTREAM_USE_APPSINK_CALLBACKS", false);
  bool stop_unblock = pipeline_internal::env_bool("SIMA_INPUTSTREAM_STOP_UNBLOCK", true);
  bool stop_flush = pipeline_internal::env_bool("SIMA_INPUTSTREAM_STOP_FLUSH", true);
  bool push_timing = pipeline_internal::env_bool("SIMA_INPUTSTREAM_PUSH_TIMING", false);
  int holder_max_inflight =
      std::max(0, pipeline_internal::env_int("SIMA_INPUTSTREAM_HOLDER_MAX_INFLIGHT", 0));
  int worker_poll_ms_default =
      std::max(10, std::atoi(pipeline_internal::env_str("SIMA_INPUTSTREAM_POLL_MS", "50").c_str()));
  int cb_stop_timeout_ms =
      std::max(0, pipeline_internal::env_int("SIMA_INPUTSTREAM_CB_STOP_TIMEOUT_MS", 2000));
  int stop_flush_timeout_ms =
      std::max(0, pipeline_internal::env_int("SIMA_INPUTSTREAM_STOP_FLUSH_TIMEOUT_MS", 500));
  int stop_timeout_ms =
      std::max(0, pipeline_internal::env_int("SIMA_INPUTSTREAM_STOP_TIMEOUT_MS", 2000));
};

const InputStreamDebugFlags& inputstream_debug_flags() {
  static const InputStreamDebugFlags flags;
  return flags;
}

bool holder_debug_enabled() {
  return inputstream_debug_flags().holder_debug;
}

bool stop_trace_enabled() {
  return inputstream_debug_flags().stop_trace;
}

bool inputstream_debug_enabled() {
  return inputstream_debug_flags().inputstream_debug;
}

bool pipeline_debug_enabled() {
  return inputstream_debug_flags().pipeline_debug;
}

bool graph_debug_enabled() {
  return inputstream_debug_flags().graph_debug;
}

bool pipeline_or_graph_debug_enabled() {
  const auto& flags = inputstream_debug_flags();
  return flags.pipeline_debug || flags.graph_debug;
}

bool push_ref_debug_enabled() {
  return inputstream_debug_flags().push_ref_debug;
}

bool unref_on_push_fail_enabled() {
  return inputstream_debug_flags().unref_on_push_fail;
}

bool push_fail_debug_enabled() {
  return inputstream_debug_flags().push_fail_debug;
}

bool push_fail_detail_enabled() {
  return inputstream_debug_flags().push_fail_detail;
}

bool drop_holder_after_push_enabled() {
  return inputstream_debug_flags().drop_holder_after_push;
}

bool inputstream_push_timing_enabled() {
  return inputstream_debug_flags().push_timing;
}

bool eos_debug_enabled() {
  return inputstream_debug_flags().eos_debug;
}

bool appsink_cb_debug_enabled() {
  return inputstream_debug_flags().appsink_cb_debug;
}

bool appsink_drop_last_debug_enabled() {
  return inputstream_debug_flags().appsink_drop_last_debug;
}

bool inputstream_meta_debug_enabled() {
  return inputstream_debug_flags().inputstream_meta_debug;
}

bool inputstream_pool_debug_enabled() {
  return inputstream_debug_flags().inputstream_pool_debug;
}

bool inputstream_dot_on_timeout_enabled() {
  return inputstream_debug_flags().dot_on_timeout;
}

bool inputstream_use_appsink_callbacks_enabled() {
  return inputstream_debug_flags().use_appsink_callbacks;
}

bool inputstream_stop_unblock_enabled() {
  return inputstream_debug_flags().stop_unblock;
}

bool inputstream_stop_flush_enabled() {
  return inputstream_debug_flags().stop_flush;
}

int holder_max_inflight() {
  return inputstream_debug_flags().holder_max_inflight;
}

int inputstream_worker_poll_ms_default() {
  return inputstream_debug_flags().worker_poll_ms_default;
}

int inputstream_cb_stop_timeout_ms() {
  return inputstream_debug_flags().cb_stop_timeout_ms;
}

int inputstream_stop_flush_timeout_ms() {
  return inputstream_debug_flags().stop_flush_timeout_ms;
}

int inputstream_stop_timeout_ms() {
  return inputstream_debug_flags().stop_timeout_ms;
}

void set_stream_error(InputStream::State& st, const std::string& msg) {
  std::lock_guard<std::mutex> lock(st.error_mu);
  st.error = msg;
}

std::string format_push_failure_error(const InputStream::State& st, const char* where,
                                      GstFlowReturn ret) {
  const char* flow_name = gst_flow_get_name(ret);
  const char* tag = where ? where : "InputStream::push";
  std::ostringstream oss;
  oss << tag << ": appsrc push failed"
      << " (flow=" << static_cast<int>(ret) << ":" << (flow_name ? flow_name : "<unknown>") << ")";
  if (st.stop_requested.load(std::memory_order_relaxed) || ret == GST_FLOW_FLUSHING ||
      ret == GST_FLOW_EOS) {
    oss << ". Hint: stream is stopping or EOS has been reached.";
  } else {
    oss << ". Hint: inspect GraphReport/Run report bus diagnostics for upstream GST errors.";
  }
  return oss.str();
}

[[noreturn]] void throw_push_failed_with_last_error(const char* where,
                                                    const std::shared_ptr<InputStream::State>& st) {
  const char* tag = where ? where : "InputStream::push";
  std::ostringstream oss;
  oss << tag << ": push failed";
  if (!st) {
    oss << " (stream state unavailable)";
  } else {
    std::lock_guard<std::mutex> lock(st->error_mu);
    if (!st->error.empty()) {
      oss << ": " << st->error;
    } else if (st->stop_requested.load(std::memory_order_relaxed)) {
      oss << ": stream is stopping";
    } else {
      oss << ": appsrc rejected buffer";
    }
  }
  throw std::runtime_error(oss.str());
}

void log_push_refcount(const char* where, GstBuffer* buffer) {
  if (!push_ref_debug_enabled())
    return;
  const int refcount = buffer ? GST_MINI_OBJECT_REFCOUNT_VALUE(buffer) : -1;
  const gsize size = buffer ? gst_buffer_get_size(buffer) : 0;
  std::fprintf(stderr, "[PUSH_REF] %s buffer=%p refcount=%d size=%" G_GSIZE_FORMAT "\n",
               where ? where : "push", static_cast<void*>(buffer), refcount, size);
}

std::string push_fail_context(const char* where, const Sample& msg, const SampleSpec& spec,
                              const InputOptions& opt,
                              const std::optional<int64_t>& input_seq_override,
                              const std::optional<int64_t>& orig_input_seq_override) {
  std::ostringstream oss;
  oss << (where ? where : "push")
      << " stream=" << (msg.stream_id.empty() ? "<empty>" : msg.stream_id)
      << " frame=" << msg.frame_id << " input_seq="
      << (input_seq_override.has_value() ? std::to_string(*input_seq_override) : "n/a")
      << " orig_input_seq="
      << (orig_input_seq_override.has_value() ? std::to_string(*orig_input_seq_override) : "n/a")
      << " port=" << (msg.port_name.empty() ? "<empty>" : msg.port_name)
      << " buffer_name=" << (opt.buffer_name.empty() ? "<default>" : opt.buffer_name)
      << " media=" << (spec.media_type.empty() ? "<none>" : spec.media_type)
      << " format=" << (spec.format.empty() ? "<none>" : spec.format) << " w=" << spec.width
      << " h=" << spec.height << " depth=" << spec.depth;
  return oss.str();
}

void maybe_drop_holder_after_push(const simaai::neat::Tensor& tensor, const char* where) {
  if (!drop_holder_after_push_enabled())
    return;
  if (pipeline_internal::drop_tensor_holder(tensor)) {
    if (holder_debug_enabled()) {
      std::fprintf(stderr, "[HOLDER] drop after push where=%s\n", where ? where : "push");
    }
  }
}

void apply_default_port_name(Sample& out, const InputOptions& opt) {
  const bool tensor_payload = sample_has_tensor_list(out);
  if (tensor_payload)
    return;
  if (!out.stream_label.empty())
    return;
  if (!opt.buffer_name.empty()) {
    out.stream_label = opt.buffer_name;
    out.port_name = opt.buffer_name;
  }
}

void log_appsink_sample_refs(const char* tag, GstSample* sample, std::size_t queue_size) {
  if (!appsink_cb_debug_enabled())
    return;
  if (!sample)
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
               "[APPSINK_CB] %s sample=%p sample_ref=%u buffer=%p buf_ref=%u "
               "writable=%s mem_readonly=%s pool=%p type=%s queue=%zu\n",
               tag ? tag : "sample", static_cast<void*>(sample), sample_ref,
               static_cast<void*>(buf), buf_ref, buf_writable ? "true" : "false",
               mem_readonly ? "true" : "false", buf ? static_cast<void*>(buf->pool) : nullptr,
               pool_type ? pool_type : "<null>", queue_size);
}

bool maybe_drop_appsink_last_sample(GstElement* appsink) {
  if (!appsink)
    return false;
  const bool dbg = appsink_drop_last_debug_enabled();
  GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(appsink), "last-sample");
  if (!pspec) {
    if (dbg) {
      std::fprintf(stderr, "[APPSINK_DROP] last-sample property not found\n");
    }
    return false;
  }
  if (!(pspec->flags & G_PARAM_WRITABLE)) {
    if (dbg) {
      std::fprintf(stderr, "[APPSINK_DROP] last-sample not writable\n");
    }
    return false;
  }
  GstSample* last = nullptr;
  g_object_get(appsink, "last-sample", &last, nullptr);
  if (!last) {
    if (dbg) {
      std::fprintf(stderr, "[APPSINK_DROP] last-sample already null\n");
    }
    return false;
  }
  g_object_set(appsink, "last-sample", nullptr, nullptr);
  if (dbg) {
    std::fprintf(stderr, "[APPSINK_DROP] cleared last-sample=%p\n", static_cast<void*>(last));
  }
  gst_sample_unref(last);
  return true;
}

struct WeakRefTag {
  int id = 0;
  const char* label = nullptr;
};

void mini_object_weak_notify(gpointer data, GstMiniObject* obj) {
  auto* tag = static_cast<WeakRefTag*>(data);
  const char* type = obj ? g_type_name(GST_MINI_OBJECT_TYPE(obj)) : "<null>";
  std::fprintf(stderr, "[WEAKREF] %s id=%d obj=%p type=%s\n",
               tag && tag->label ? tag->label : "obj", tag ? tag->id : -1, static_cast<void*>(obj),
               type ? type : "<null>");
  delete tag;
}

int next_weak_id() {
  static std::atomic<int> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

bool is_simaai_pool(GstBufferPool* pool) {
  if (!pool)
    return false;
  const char* type_name = G_OBJECT_TYPE_NAME(pool);
  if (type_name) {
    if (std::strstr(type_name, "Simaai") != nullptr ||
        std::strstr(type_name, "simaai") != nullptr) {
      return true;
    }
  }
  const char* obj_name = GST_OBJECT_NAME(pool);
  if (obj_name) {
    if (std::strstr(obj_name, "Simaai") != nullptr || std::strstr(obj_name, "simaai") != nullptr) {
      return true;
    }
  }
  return false;
}

GstFlowReturn appsink_new_sample(GstAppSink* sink, gpointer user_data) {
  auto* ctx = static_cast<InputStream::State::CallbackCtx*>(user_data);
  if (!ctx || !ctx->st)
    return GST_FLOW_FLUSHING;
  auto st = ctx->st;
  CallbackInflightGuard inflight(st);
  if (st->stop_requested.load())
    return GST_FLOW_FLUSHING;
  GstSample* sample = gst_app_sink_try_pull_sample(sink, 0);
  if (!sample)
    return GST_FLOW_OK;

  {
    std::lock_guard<std::mutex> lock(st->cb_mu);
    if (st->cb_queue_max > 0 && st->cb_queue.size() >= st->cb_queue_max) {
      log_appsink_sample_refs("drop", sample, st->cb_queue.size());
      gst_sample_unref(sample);
      return GST_FLOW_OK;
    }
    log_appsink_sample_refs("queue", sample, st->cb_queue.size());
    st->cb_queue.push_back(sample);
  }
  st->cb_cv.notify_one();
  return GST_FLOW_OK;
}

GstFlowReturn appsink_new_preroll(GstAppSink* sink, gpointer user_data) {
  auto* ctx = static_cast<InputStream::State::CallbackCtx*>(user_data);
  if (!ctx || !ctx->st)
    return GST_FLOW_OK;
  auto st = ctx->st;
  CallbackInflightGuard inflight(st);
  if (st->stop_requested.load())
    return GST_FLOW_FLUSHING;
  return GST_FLOW_OK;
}

void appsink_eos(GstAppSink* /*sink*/, gpointer user_data) {
  auto* ctx = static_cast<InputStream::State::CallbackCtx*>(user_data);
  if (!ctx || !ctx->st)
    return;
  auto st = ctx->st;
  CallbackInflightGuard inflight(st);
  st->cb_eos.store(true);
  st->cb_cv.notify_all();
}

void verify_buffer_name_override(const InputOptions& opt,
                                 const std::optional<std::string>& buffer_name_override,
                                 const char* where) {
  if (!buffer_name_override.has_value())
    return;
  if (opt.buffer_name.empty())
    return;
  const std::string expected = opt.buffer_name;
  if (buffer_name_matches_expected(expected, *buffer_name_override))
    return;
  const char* tag = where ? where : "InputStream";
  std::ostringstream oss;
  oss << tag << ": buffer-name mismatch (expected '" << expected << "', got '"
      << *buffer_name_override << "')";
  throw std::runtime_error(oss.str());
}

void release_input_buffer(GstBuffer* buffer, const char* where) {
  if (!buffer)
    return;
  track_input_pool_release(buffer, where);
  debug_input_buffer_release(buffer, where);
  gst_buffer_unref(buffer);
}

void release_input_buffer_on_push_fail(GstBuffer* buffer, const char* where) {
  if (!buffer)
    return;
  if (unref_on_push_fail_enabled()) {
    release_input_buffer(buffer, where);
  }
}

void log_push_failure(InputStream::State& st, const char* where, GstFlowReturn ret) {
  if (!push_fail_debug_enabled())
    return;
  const char* name = gst_flow_get_name(ret);
  const char* appsrc_name = st.appsrc ? gst_element_get_name(st.appsrc) : "<null>";
  const char* pipeline_name = st.pipeline ? gst_element_get_name(st.pipeline) : "<null>";
  const char* buffer_name =
      st.src_opt.buffer_name.empty() ? "<default>" : st.src_opt.buffer_name.c_str();
  GstState cur = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  GstStateChangeReturn state_ret = GST_STATE_CHANGE_FAILURE;
  if (st.pipeline) {
    state_ret = gst_element_get_state(st.pipeline, &cur, &pending, 0);
  }
  std::fprintf(stderr,
               "[PUSH_FAIL] %s ret=%d(%s) stop=%d teardown=%d appsrc=%s pipeline=%s buffer_name=%s "
               "state=%s pending=%s state_ret=%d\n",
               where ? where : "push", static_cast<int>(ret), name ? name : "<unknown>",
               st.stop_requested.load() ? 1 : 0, st.teardown_started.load() ? 1 : 0, appsrc_name,
               pipeline_name, buffer_name, gst_element_state_get_name(cur),
               gst_element_state_get_name(pending), static_cast<int>(state_ret));
}

bool handle_appsrc_push_fail(InputStream::State& st, const char* where, GstFlowReturn ret) {
  if (st.pipeline) {
    pipeline_internal::throw_if_bus_error(st.pipeline, st.diag, "InputStream::push_fail");
    pipeline_internal::drain_bus(st.pipeline, st.diag, "InputStream::push_fail");
  }
  set_stream_error(st, format_push_failure_error(st, where, ret));
  log_push_failure(st, where, ret);
  if (ret == GST_FLOW_FLUSHING || ret == GST_FLOW_EOS) {
    st.stop_requested.store(true);
    return false;
  }
  return true;
}

struct BufferUnrefGuard {
  GstBuffer** buffer = nullptr;
  const char* where = nullptr;
  BufferUnrefGuard(GstBuffer** buf, const char* tag) : buffer(buf), where(tag) {}
  BufferUnrefGuard(const BufferUnrefGuard&) = delete;
  BufferUnrefGuard& operator=(const BufferUnrefGuard&) = delete;
  ~BufferUnrefGuard() {
    if (buffer && *buffer) {
      release_input_buffer(*buffer, where);
    }
  }
  void release() {
    buffer = nullptr;
  }
};

struct BuiltBuffer {
  GstBuffer* buffer = nullptr;
  std::uint64_t alloc_ns = 0;
  std::uint64_t map_ns = 0;
  std::uint64_t copy_ns = 0;
};

InputDropInfo drop_info_from_spec(const SampleSpec& spec, const char* reason) {
  InputDropInfo info;
  info.kind = SampleKind::TensorSet;
  info.media_type = spec.media_type;
  info.format = spec.format;
  info.width = spec.width;
  info.height = spec.height;
  info.depth = spec.depth;
  info.reason = reason ? reason : "";
  return info;
}

void notify_drop(InputStream::State& st, const SampleSpec& spec, const char* reason) {
  if (!st.opt.on_input_drop)
    return;
  st.opt.on_input_drop(drop_info_from_spec(spec, reason));
}

void discard_pending_buffer(InputStream::State& st, const char* reason) {
  if (!st.pending_buffer)
    return;
  if (st.pending_spec.has_value()) {
    notify_drop(st, *st.pending_spec, reason);
  }
  release_input_buffer(st.pending_buffer, "discard_pending_buffer");
  st.pending_buffer = nullptr;
  st.pending_spec.reset();
  st.pending_alloc_ns = 0;
  st.pending_map_ns = 0;
  st.pending_copy_ns = 0;
}

void queue_pending_buffer(InputStream::State& st, BuiltBuffer pending, const SampleSpec& spec,
                          const char* reason) {
  if (!pending.buffer) {
    throw std::runtime_error(std::string(reason ? reason : "InputStream") +
                             ": pending buffer missing");
  }
  if (st.pending_buffer) {
    if (st.pending_spec.has_value()) {
      notify_drop(st, *st.pending_spec, "pending_regime_overwrite");
    }
    release_input_buffer(st.pending_buffer, "queue_pending_buffer:overwrite");
  }
  st.pending_buffer = pending.buffer;
  st.pending_spec = spec;
  st.pending_alloc_ns = pending.alloc_ns;
  st.pending_map_ns = pending.map_ns;
  st.pending_copy_ns = pending.copy_ns;
}

bool flush_pending_buffer(InputStream::State& st, const char* where) {
  if (!st.pending_buffer)
    return true;
  GstBuffer* buf = st.pending_buffer;
  st.pending_buffer = nullptr;
  st.pending_spec.reset();
  const bool timings = st.timing_enabled;
  if (timings) {
    st.alloc_ns.fetch_add(st.pending_alloc_ns, std::memory_order_relaxed);
    st.map_ns.fetch_add(st.pending_map_ns, std::memory_order_relaxed);
    st.copy_ns.fetch_add(st.pending_copy_ns, std::memory_order_relaxed);
  }
  st.pending_alloc_ns = 0;
  st.pending_map_ns = 0;
  st.pending_copy_ns = 0;

  std::chrono::steady_clock::time_point t_push_start{};
  if (timings)
    t_push_start = std::chrono::steady_clock::now();
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st.appsrc), buf);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (timings) {
    st.push_count.fetch_add(1, std::memory_order_relaxed);
    st.push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
  }

  if (ret != GST_FLOW_OK) {
    release_input_buffer_on_push_fail(buf, "flush_pending_buffer:push_fail");
    if (timings) {
      st.push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    handle_appsrc_push_fail(st, "flush_pending_buffer", ret);
    return false;
  }
  st.inflight.fetch_add(1, std::memory_order_relaxed);
  const auto push_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end.time_since_epoch()).count();
  st.last_push_ns.store(static_cast<std::int64_t>(push_ns), std::memory_order_relaxed);
  return true;
}

BuiltBuffer build_buffer_with_fill(
    InputStream::State& st, const char* where, const std::function<size_t(uint8_t*, size_t)>& fill,
    size_t required_bytes, const std::optional<int64_t>& frame_id_override,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const std::optional<std::string>& stream_id_override,
    const std::optional<std::string>& buffer_name_override,
    const SampleTimingOverrides& timing_override, const std::function<void(GstBuffer**)>& prepare,
    bool record_timings, const char* op_tag, bool release_reuse_buffer_on_fail, int input_width,
    int input_height) {
  const char* tag = (op_tag && *op_tag) ? op_tag : "build_buffer_with_fill";
  verify_buffer_name_override(st.src_opt, buffer_name_override, where);
  std::chrono::steady_clock::time_point t_alloc_start{};
  if (record_timings)
    t_alloc_start = std::chrono::steady_clock::now();
  GstBuffer* buf = nullptr;
  if (st.opt.reuse_input_buffer) {
    if (!st.reusable_buffer || st.reusable_bytes != st.alloc_bytes) {
      if (st.reusable_buffer) {
        release_input_buffer(st.reusable_buffer, (std::string(tag) + ":drop_reusable").c_str());
        st.reusable_buffer = nullptr;
        st.reusable_bytes = 0;
      }
      st.reusable_buffer = allocate_input_buffer(st.alloc_bytes, st.src_opt, st.pool_guard);
      st.reusable_bytes = st.alloc_bytes;
    }
    buf = st.reusable_buffer;
  } else {
    buf = allocate_input_buffer(st.alloc_bytes, st.src_opt, st.pool_guard);
  }
  std::chrono::steady_clock::time_point t_alloc_end{};
  if (record_timings)
    t_alloc_end = std::chrono::steady_clock::now();
  if (!buf) {
    throw std::runtime_error(std::string(where) + ": failed to allocate GstBuffer");
  }

  if (required_bytes > 0) {
    if (required_bytes > st.alloc_bytes) {
      if (!st.opt.reuse_input_buffer || release_reuse_buffer_on_fail) {
        release_input_buffer(buf, (std::string(tag) + ":oversize").c_str());
      }
      std::ostringstream msg;
      msg << where << ": input exceeds allocated buffer size"
          << " (required=" << required_bytes << ", allocated=" << st.alloc_bytes << "). "
          << "Fix: increase RunAdvancedOptions::max_input_bytes or "
          << "Model::Options::input_max_* limits.";
      throw std::runtime_error(msg.str());
    }
    gst_buffer_resize(buf, 0, required_bytes);
  }

  GstMapInfo mi{};
  std::chrono::steady_clock::time_point t_map_start{};
  if (record_timings)
    t_map_start = std::chrono::steady_clock::now();
  if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
    if (!st.opt.reuse_input_buffer || release_reuse_buffer_on_fail) {
      release_input_buffer(buf, (std::string(tag) + ":map_fail").c_str());
    }
    throw std::runtime_error(std::string(where) + ": failed to map GstBuffer");
  }
  std::chrono::steady_clock::time_point t_map_end{};
  if (record_timings)
    t_map_end = std::chrono::steady_clock::now();

  const size_t filled = fill(static_cast<uint8_t*>(mi.data), mi.size);
  if (filled < mi.size) {
    std::memset(static_cast<uint8_t*>(mi.data) + filled, 0, mi.size - filled);
  }
  gst_buffer_unmap(buf, &mi);
  std::chrono::steady_clock::time_point t_copy_end{};
  if (record_timings)
    t_copy_end = std::chrono::steady_clock::now();

  if (prepare) {
    try {
      prepare(&buf);
    } catch (...) {
      if (!st.opt.reuse_input_buffer || release_reuse_buffer_on_fail) {
        release_input_buffer(buf, (std::string(tag) + ":prepare_fail").c_str());
      }
      throw;
    }
  }

  buf = attach_simaai_meta_inplace(buf, st.src_opt, st.pool_guard, where, frame_id_override,
                                   stream_id_override, buffer_name_override);
  if (!buf) {
    if (!st.opt.reuse_input_buffer || release_reuse_buffer_on_fail) {
      release_input_buffer(buf, (std::string(tag) + ":attach_meta_fail").c_str());
    }
    throw std::runtime_error(std::string(where) + ": failed to attach GstSimaMeta");
  }
  update_simaai_meta_fields(buf, frame_id_override, input_seq_override, orig_input_seq_override,
                            stream_id_override, buffer_name_override, timing_override.pts_ns);
  if (!write_sample_timing_to_gst_buffer(buf, timing_override)) {
    if (!st.opt.reuse_input_buffer || release_reuse_buffer_on_fail) {
      release_input_buffer(buf, (std::string(tag) + ":sample_timing_fail").c_str());
    }
    throw std::runtime_error(std::string(where) + ": failed to write sample timing metadata");
  }
  if (!has_simaai_preprocess_meta(buf) && input_width > 0 && input_height > 0) {
    (void)apply_simaai_preprocess_meta_template(buf, st.src_opt, input_width, input_height);
  }
  if (inputstream_meta_debug_enabled()) {
    dump_sima_meta(buf, "copy_path_meta");
    dump_sima_meta_full(buf, "copy_path_meta_full");
    dump_buffer_memories(buf, "copy_path_meta");
  }

  BuiltBuffer out;
  out.buffer = st.opt.reuse_input_buffer ? gst_buffer_ref(buf) : buf;
  if (record_timings) {
    out.alloc_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_alloc_end - t_alloc_start).count());
    out.map_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_map_end - t_map_start).count());
    out.copy_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_copy_end - t_map_end).count());
  }
  return out;
}

void apply_video_meta_or_throw(GstBuffer** buffer, const SampleSpec& spec, const char* where) {
  std::string err;
  if (!pipeline_internal::attach_video_meta(buffer, spec, &err)) {
    const char* tag = where ? where : "InputStream::apply_video_meta";
    throw std::runtime_error(std::string(tag) + ": " + err);
  }
}

void apply_tensor_size_or_throw(GstBuffer** buffer, const SampleSpec& spec, const char* where) {
  std::string err;
  if (!pipeline_internal::apply_tensor_size(buffer, spec, &err)) {
    const char* tag = where ? where : "InputStream::apply_tensor_size";
    throw std::runtime_error(std::string(tag) + ": " + err);
  }
}

bool tensor_spec_matches(const SampleSpec& a, const SampleSpec& b) {
  if (a.media_type != b.media_type)
    return false;
  const std::string a_format = normalize_caps_format_for_media(a.media_type, a.format);
  const std::string b_format = normalize_caps_format_for_media(b.media_type, b.format);
  return upper_copy(a_format) == upper_copy(b_format) && a.dtype == b.dtype && a.shape == b.shape;
}

void ensure_alloc_for_bytes(InputStream::State& st, size_t bytes, const char* where) {
  if (bytes == 0 || bytes <= st.alloc_bytes)
    return;

  const char* tag = where ? where : "InputStream::ensure_alloc";
  if (st.max_input_bytes_guard > 0 && bytes > st.max_input_bytes_guard) {
    st.growth_blocked.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream msg;
    msg << tag << ": input exceeds max_input_bytes"
        << " (required=" << bytes << ", max_input_bytes=" << st.max_input_bytes_guard
        << "). Fix: increase RunAdvancedOptions::max_input_bytes or raise "
        << "Model::Options::input_max_width/input_max_height/input_max_depth.";
    throw std::runtime_error(msg.str());
  }

  if (st.alloc_bytes == 0) {
    st.alloc_bytes = bytes;
  } else if (!st.allow_dynamic_growth) {
    st.growth_blocked.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream msg;
    msg << tag << ": input exceeds allocated buffer size"
        << " (required=" << bytes << ", allocated=" << st.alloc_bytes << "). "
        << "Fix: increase max bounds or rebuild with dynamic-capable ingress.";
    throw std::runtime_error(msg.str());
  } else {
    st.alloc_grows.fetch_add(1, std::memory_order_relaxed);
    st.alloc_bytes = bytes;
  }

  st.pool_guard.pool.reset();
  if (st.reusable_buffer) {
    release_input_buffer(st.reusable_buffer, "ensure_alloc_for_bytes:drop_reusable");
    st.reusable_buffer = nullptr;
    st.reusable_bytes = 0;
  }
}

enum class CapsDecision {
  Push,
  Queue,
  Flush,
};

// Caps policy:
// - Caps always reflect actual frame geometry.
// - Renegotiate only on format/width/height changes.
// - Stability guard queues the latest pending sample until K consecutive frames
//   confirm the new regime (older pending samples are dropped).
// - Raw video buffers must carry GstVideoMeta so OSS plugins interpret
//   offsets/strides correctly.

// =====================================================================================
// Split implementation chunks
// =====================================================================================

// Appsrc must carry GstSimaMeta; holder paths preserve it.
void attach_required_meta(GstBuffer* buffer, const InputOptions& opt, InputBufferPoolGuard& guard,
                          const char* where) {
  buffer = attach_simaai_meta_inplace(buffer, opt, guard, where, std::nullopt, std::nullopt,
                                      std::nullopt);
  if (!buffer) {
    release_input_buffer(buffer, "attach_required_meta");
    throw std::runtime_error(std::string(where) + ": failed to attach GstSimaMeta");
  }
}

bool InputStream::push_with_fill(const char* where,
                                 const std::function<size_t(uint8_t*, size_t)>& fill,
                                 size_t required_bytes,
                                 const std::optional<int64_t>& frame_id_override,
                                 const std::optional<int64_t>& input_seq_override,
                                 const std::optional<int64_t>& orig_input_seq_override,
                                 const std::optional<std::string>& stream_id_override,
                                 const std::optional<std::string>& buffer_name_override,
                                 const SampleTimingOverrides& timing_override,
                                 const std::function<void(GstBuffer**)>& prepare, int input_width,
                                 int input_height) {
  auto st = state_;
  if (!st || !st->pipeline) {
    throw std::runtime_error(std::string(where) + ": stream is closed");
  }
  if (!st->appsrc) {
    throw std::runtime_error(std::string(where) + ": appsrc not available (no Input)");
  }
  const bool timings = st->timing_enabled;
  const bool push_timing = inputstream_debug_flags().push_timing;

  const auto t_build_start =
      push_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  BuiltBuffer built = build_buffer_with_fill(
      *st, where, fill, required_bytes, frame_id_override, input_seq_override,
      orig_input_seq_override, stream_id_override, buffer_name_override, timing_override, prepare,
      timings || push_timing, "push_with_fill", true, input_width, input_height);
  const auto t_build_end =
      push_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  GstBuffer* buf = built.buffer;
  GstBuffer* push_src = buf;

  std::chrono::steady_clock::time_point t_push_start{};
  if (timings || push_timing)
    t_push_start = std::chrono::steady_clock::now();
  GstBuffer* push_buf = push_src;
  if (st->opt.reuse_input_buffer && push_src == buf) {
    push_buf = gst_buffer_ref(buf);
  }
  const size_t buf_bytes = gst_buffer_get_size(push_buf);
  log_push_refcount(where, push_buf);
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st->appsrc), push_buf);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (timings) {
    st->push_count.fetch_add(1, std::memory_order_relaxed);
    st->alloc_ns.fetch_add(built.alloc_ns, std::memory_order_relaxed);
    st->map_ns.fetch_add(built.map_ns, std::memory_order_relaxed);
    st->copy_ns.fetch_add(built.copy_ns, std::memory_order_relaxed);
    st->push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
  }
  if (push_timing) {
    const auto push_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start).count();
    const auto build_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_build_end - t_build_start).count();
    std::fprintf(stderr,
                 "[INPUTSTREAM_PUSH_TIMING] %s build_ns=%lld alloc_ns=%llu map_ns=%llu "
                 "copy_ns=%llu appsrc_push_ns=%lld bytes=%zu ret=%d\n",
                 where ? where : "InputStream::push_with_fill", static_cast<long long>(build_ns),
                 static_cast<unsigned long long>(built.alloc_ns),
                 static_cast<unsigned long long>(built.map_ns),
                 static_cast<unsigned long long>(built.copy_ns), static_cast<long long>(push_ns),
                 buf_bytes, static_cast<int>(ret));
  }

  if (ret != GST_FLOW_OK) {
    if (st->opt.reuse_input_buffer) {
      release_input_buffer_on_push_fail(push_buf, "push_with_fill:push_fail_ref");
    } else {
      release_input_buffer_on_push_fail(buf, "push_with_fill:push_fail");
    }
    if (timings) {
      st->push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    handle_appsrc_push_fail(*st, where ? where : "push_with_fill", ret);
    return false;
  }
  st->inflight.fetch_add(1, std::memory_order_relaxed);
  const auto push_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end.time_since_epoch()).count();
  st->last_push_ns.store(static_cast<std::int64_t>(push_ns), std::memory_order_relaxed);
  return true;
}

} // namespace simaai::neat

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "InputStream.h"
#include "InputStreamUtil.h"

#include "nodes/io/Input.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/HolderLoanGate.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using pipeline_internal::DiagCtx;
using pipeline_internal::trim_copy;
using pipeline_internal::upper_copy;

struct TensorSetOutputDecodeSignature {
  guint caps_hash = 0;
  guint tensor_count = 0;
  guint descriptor_size = 0;
  guint memory_count = 0;
  bool copy_output = false;
  std::string stage_key;
  std::uint64_t meta_hash = 0;

  bool operator==(const TensorSetOutputDecodeSignature& other) const {
    return caps_hash == other.caps_hash && tensor_count == other.tensor_count &&
           descriptor_size == other.descriptor_size && memory_count == other.memory_count &&
           copy_output == other.copy_output && stage_key == other.stage_key &&
           meta_hash == other.meta_hash;
  }
};

struct CachedTensorSetOutputDecode {
  bool valid = false;
  TensorSetOutputDecodeSignature signature;
  Sample sample_template;
  TensorList tensor_templates;
  std::vector<Segment> runtime_segments;
};

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
  std::shared_ptr<void> lifetime_token;
  pipeline_internal::HolderLoanGatePtr holder_loan_gate;
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
  std::mutex output_decode_mu;
  CachedTensorSetOutputDecode tensor_set_output_decode_cache;
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

struct WeakRefTag {
  int id = 0;
  const char* label = nullptr;
};

struct BuiltBuffer {
  GstBuffer* buffer = nullptr;
  std::uint64_t alloc_ns = 0;
  std::uint64_t map_ns = 0;
  std::uint64_t copy_ns = 0;
};

enum class CapsDecision {
  Push,
  Queue,
  Flush,
};

bool buffer_name_matches_expected(const std::string& expected_list, const std::string& actual);
bool holder_debug_enabled();
bool stop_trace_enabled();
bool inputstream_debug_enabled();
bool pipeline_debug_enabled();
bool graph_debug_enabled();
bool pipeline_or_graph_debug_enabled();
bool push_ref_debug_enabled();
bool unref_on_push_fail_enabled();
bool push_fail_debug_enabled();
bool push_fail_detail_enabled();
bool drop_holder_after_push_enabled();
bool inputstream_push_timing_enabled();
bool eos_debug_enabled();
bool appsink_cb_debug_enabled();
bool appsink_drop_last_debug_enabled();
bool inputstream_meta_debug_enabled();
bool inputstream_pool_debug_enabled();
bool inputstream_dot_on_timeout_enabled();
bool inputstream_use_appsink_callbacks_enabled();
bool inputstream_stop_unblock_enabled();
bool inputstream_stop_flush_enabled();
int holder_max_inflight();
int inputstream_worker_poll_ms_default();
int inputstream_cb_stop_timeout_ms();
int inputstream_stop_flush_timeout_ms();
int inputstream_stop_timeout_ms();

void set_stream_error(InputStream::State& st, const std::string& msg);
std::string format_push_failure_error(const InputStream::State& st, const char* where,
                                      GstFlowReturn ret);
[[noreturn]] void throw_push_failed_with_last_error(const char* where,
                                                    const std::shared_ptr<InputStream::State>& st);
void log_push_refcount(const char* where, GstBuffer* buffer);
std::string push_fail_context(const char* where, const Sample& msg, const SampleSpec& spec,
                              const InputOptions& opt,
                              const std::optional<int64_t>& input_seq_override,
                              const std::optional<int64_t>& orig_input_seq_override);
void maybe_drop_holder_after_push(const simaai::neat::Tensor& tensor, const char* where);

void apply_default_port_name(Sample& out, const InputOptions& opt);
void log_appsink_sample_refs(const char* tag, GstSample* sample, std::size_t queue_size);
bool maybe_drop_appsink_last_sample(GstElement* appsink);
void mini_object_weak_notify(gpointer data, GstMiniObject* obj);
int next_weak_id();
bool is_simaai_pool(GstBufferPool* pool);

GstFlowReturn appsink_new_sample(GstAppSink* sink, gpointer user_data);
GstFlowReturn appsink_new_preroll(GstAppSink* sink, gpointer user_data);
void appsink_eos(GstAppSink* sink, gpointer user_data);

void verify_buffer_name_override(const InputOptions& opt,
                                 const std::optional<std::string>& buffer_name_override,
                                 const char* where);
void release_input_buffer(GstBuffer* buffer, const char* where);
void release_input_buffer_on_push_fail(GstBuffer* buffer, const char* where);
void log_push_failure(InputStream::State& st, const char* where, GstFlowReturn ret);
bool handle_appsrc_push_fail(InputStream::State& st, const char* where, GstFlowReturn ret);

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

InputDropInfo drop_info_from_spec(const SampleSpec& spec, const char* reason);
void notify_drop(InputStream::State& st, const SampleSpec& spec, const char* reason);
void discard_pending_buffer(InputStream::State& st, const char* reason);
void queue_pending_buffer(InputStream::State& st, BuiltBuffer pending, const SampleSpec& spec,
                          const char* reason);
bool flush_pending_buffer(InputStream::State& st, const char* where);
BuiltBuffer build_buffer_with_fill(
    InputStream::State& st, const char* where, const std::function<size_t(uint8_t*, size_t)>& fill,
    size_t required_bytes, const std::optional<int64_t>& frame_id_override,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const std::optional<std::string>& stream_id_override,
    const std::optional<std::string>& buffer_name_override,
    const SampleTimingOverrides& timing_override, const std::function<void(GstBuffer**)>& prepare,
    bool record_timings, const char* op_tag, bool release_reuse_buffer_on_fail,
    int input_width = -1, int input_height = -1);

void apply_video_meta_or_throw(GstBuffer** buffer, const SampleSpec& spec, const char* where);
void apply_tensor_size_or_throw(GstBuffer** buffer, const SampleSpec& spec, const char* where);
bool tensor_spec_matches(const SampleSpec& a, const SampleSpec& b);
void ensure_alloc_for_bytes(InputStream::State& st, size_t bytes, const char* where);

bool apply_caps_or_throw(InputStream::State& st, const SampleSpec& caps_spec, const char* where);
CapsDecision maybe_update_caps_for_spec(InputStream::State& st, const SampleSpec& spec,
                                        const char* where);

void attach_required_meta(GstBuffer* buffer, const InputOptions& opt, InputBufferPoolGuard& guard,
                          const char* where);

Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt,
                                 InputStream::State* st);
Sample sample_from_gst_envelope(GstSample* sample, const char* where, bool copy_output,
                                const std::optional<OutputTensorOverride>* override_opt,
                                InputStream::State* st);
void maybe_restore_cached_preprocess_meta(InputStream::State& st, GstSample* sample);
void maybe_restore_cached_preprocess_meta_on_sample(InputStream::State& st, Sample* sample);
Sample decode_sample_from_inputstream_state(InputStream::State& st, GstSample* sample,
                                            const char* where);

} // namespace simaai::neat

#include "gst/GstLatestByStreamMux.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/RealtimeLinkOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/SampleTimingGstUtil.h"
#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr const char* kFactoryName = "neatlatestbystreammux";
constexpr const char* kLoanValidField = "neat-latest-mux-loan-valid";
constexpr const char* kLoanNamespaceField = "neat-latest-mux-loan-namespace";
constexpr const char* kLoanStreamIdField = "neat-latest-mux-loan-stream-id";
constexpr const char* kLoanFrameIdField = "neat-latest-mux-loan-frame-id";

GST_DEBUG_CATEGORY_STATIC(gst_latest_by_stream_mux_debug_category);
#define GST_CAT_DEFAULT gst_latest_by_stream_mux_debug_category

using GstLatestByStreamMux = struct _GstLatestByStreamMux;
using GstLatestByStreamMuxClass = struct _GstLatestByStreamMuxClass;
GType gst_latest_by_stream_mux_get_type();
#define GST_TYPE_LATEST_BY_STREAM_MUX (gst_latest_by_stream_mux_get_type())
#define GST_LATEST_BY_STREAM_MUX(obj)                                                              \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_LATEST_BY_STREAM_MUX, GstLatestByStreamMux))

bool loan_debug_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_LOAN_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

bool loan_meta_debug_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_LOAN_META_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

using HolderLoanGatePtr = simaai::neat::pipeline_internal::HolderLoanGatePtr;

struct StreamLoanState {
  StreamLoanState(int credit_limit, HolderLoanGatePtr total_credit_gate)
      : gate(std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(credit_limit)),
        total_gate(std::move(total_credit_gate)) {}

  HolderLoanGatePtr gate;
  HolderLoanGatePtr total_gate;
  std::atomic<std::uint64_t> registered{0};
  std::atomic<std::uint64_t> released_by_output{0};
  std::atomic<std::uint64_t> released_without_output{0};
  std::atomic<std::uint64_t> missing_key{0};
};

std::shared_ptr<StreamLoanState> make_stream_loan_state(int credit_limit,
                                                        HolderLoanGatePtr total_credit_gate) {
  return std::make_shared<StreamLoanState>(credit_limit, std::move(total_credit_gate));
}

bool loan_gate_enabled(const HolderLoanGatePtr& gate) {
  return gate && gate->enabled();
}

bool loan_state_enabled(const std::shared_ptr<StreamLoanState>& state) {
  return state && (loan_gate_enabled(state->gate) || loan_gate_enabled(state->total_gate));
}

bool loan_state_has_credit(const std::shared_ptr<StreamLoanState>& state) {
  if (!loan_state_enabled(state)) {
    return true;
  }
  const auto available = [](const HolderLoanGatePtr& gate) {
    return !loan_gate_enabled(gate) || gate->inflight() < gate->credit_limit();
  };
  return available(state->total_gate) && available(state->gate);
}

bool try_acquire_loan_state(const std::shared_ptr<StreamLoanState>& state) {
  if (!loan_state_enabled(state)) {
    return false;
  }
  const bool acquire_total = loan_gate_enabled(state->total_gate);
  if (acquire_total && !state->total_gate->try_acquire()) {
    return false;
  }
  if (loan_gate_enabled(state->gate) && !state->gate->try_acquire()) {
    if (acquire_total) {
      state->total_gate->release();
    }
    return false;
  }
  return true;
}

void release_loan_state(const std::shared_ptr<StreamLoanState>& state) {
  if (!state) {
    return;
  }
  if (loan_gate_enabled(state->gate)) {
    state->gate->release();
  }
  if (loan_gate_enabled(state->total_gate)) {
    state->total_gate->release();
  }
}

struct PendingSlot {
  GstPad* pad = nullptr;
  guint index = 0;
  GstBuffer* pending = nullptr; // owned by the slot while queued
  GstCaps* caps = nullptr;      // latest caps seen on this stream
  bool eos = false;
  bool have_segment = false;
  bool releasing = false;
  guint active_chains = 0;
  guint64 next_frame_id = 0;
  guint64 emitted = 0;
  // Monotonic age of the current empty -> pending transition. Replacement by
  // a newer frame keeps this ticket, so a hot producer cannot lose its place
  // while still preserving latest-only storage.
  guint64 ready_ticket = 0;
  bool has_ready_ticket = false;
  // Virtual service count used only for scheduling. Unlike emitted (which is
  // a truthful output statistic), this count starts at the current service
  // frontier when a stream first joins or restarts. Established streams retain
  // their deficit across temporary absence/credit blocking and can therefore
  // consume spare capacity until the active streams are level again.
  guint64 fair_service_count = 0;
  bool fairness_active = false;
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> replaced{0};
  std::atomic<std::uint64_t> no_credit_skips{0};
  std::shared_ptr<StreamLoanState> loan_state;
};

using SlotVector = std::vector<PendingSlot*>;
using StringVector = std::vector<std::string>;
using IntVector = std::vector<int>;

struct _GstLatestByStreamMux {
  GstElement parent;
  GstPad* srcpad = nullptr;

  GMutex lock;
  GCond cond;
  SlotVector slots;
  StringVector stream_ids;
  IntVector stream_inflight_limits;
  HolderLoanGatePtr total_loan_gate;
  int max_inflight_total = 0;
  guint next_pad_index = 0;
  guint rr_index = 0;
  guint64 next_ready_ticket = 1;
  guint64 service_frontier = 0;
  bool lifetime_guard_enabled;
  bool started = false;
  bool stopping = false;
  bool flushing = false;
  bool stream_start_sent = false;
  bool segment_sent = false;
  bool have_pending_segment = false;
  bool stats_reported = false;
  GstSegment pending_segment;
  GstCaps* current_caps = nullptr;
  GThread* worker = nullptr;
  std::uint64_t loan_namespace = 0;
  std::uint64_t stats_pushes = 0;
};

struct _GstLatestByStreamMuxClass {
  GstElementClass parent_class;
};

struct LoanKey {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;

  bool operator==(const LoanKey& other) const {
    return namespace_id == other.namespace_id && frame_id == other.frame_id &&
           input_seq == other.input_seq && orig_input_seq == other.orig_input_seq &&
           stream_id == other.stream_id;
  }
};

struct LoanKeyHash {
  std::size_t operator()(const LoanKey& key) const noexcept {
    std::size_t h = std::hash<std::uint64_t>{}(key.namespace_id);
    const auto mix = [&h](std::size_t value) {
      h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    };
    mix(std::hash<std::string>{}(key.stream_id));
    mix(std::hash<std::int64_t>{}(key.frame_id));
    mix(std::hash<std::int64_t>{}(key.input_seq));
    mix(std::hash<std::int64_t>{}(key.orig_input_seq));
    return h;
  }
};

struct LoanDropGuard {
  LoanKey key;
  std::uint64_t sequence = 0;
  std::mutex mutex;
  std::uint64_t carriers = 1;
  bool armed = true;
};

struct GstLatestMuxLoanGuardMeta {
  GstMeta meta;
  std::shared_ptr<LoanDropGuard> guard;
  bool retired = false;
  bool terminal_claimed = false;
};

struct LoanEntry {
  GstLatestByStreamMux* mux = nullptr; // strong GObject ref while registered
  std::shared_ptr<StreamLoanState> state;
  std::uint64_t sequence = 0;
  std::uint64_t ref_count = 1;
  bool terminal_replacing = false;
  std::atomic<bool> released{false};
  // The admission gate protects decoder-backed input residency, whereas the
  // registry entry also carries timing until the derived terminal result is
  // matched.  Replacing hardware stages can finish with the decoder buffer
  // before that result reaches appsink, so these lifetimes must be independent.
  std::mutex resources_mutex;
  bool credit_released = false;
  simaai::neat::SampleTimingOverrides timing;
};

std::mutex& loan_registry_mutex() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<LoanKey, std::shared_ptr<LoanEntry>, LoanKeyHash>& loan_registry() {
  static std::unordered_map<LoanKey, std::shared_ptr<LoanEntry>, LoanKeyHash> registry;
  return registry;
}

std::atomic<std::uint64_t>& loan_sequence_counter() {
  static std::atomic<std::uint64_t> counter{1};
  return counter;
}

std::atomic<std::uint64_t>& mux_namespace_counter() {
  static std::atomic<std::uint64_t> counter{1};
  return counter;
}

bool ensure_sima_meta_structure_mutable(GstBuffer* buffer, GstStructure** structure_out) {
  if (!buffer || !structure_out) {
    return false;
  }
  // GstCustomMeta parents its GstStructure to the owning buffer's refcount.
  // Consequently an existing structure is mutable exactly when the buffer is
  // writable. The terminal probe already takes GStreamer's copy-on-write path;
  // copying every field into a replacement GstCustomMeta here is redundant and
  // particularly expensive for ProcessCVU metadata with many fields.
  if (!gst_buffer_is_writable(buffer)) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  if (!meta) {
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    return false;
  }

  *structure_out = s;
  return true;
}

bool stamp_latest_mux_loan_key(GstBuffer* buffer, const LoanKey& key) {
  if (!buffer || key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  GstStructure* s = nullptr;
  if (!ensure_sima_meta_structure_mutable(buffer, &s)) {
    return false;
  }
  gst_structure_set(s, kLoanValidField, G_TYPE_BOOLEAN, TRUE, kLoanNamespaceField, G_TYPE_UINT64,
                    static_cast<guint64>(key.namespace_id), kLoanStreamIdField, G_TYPE_STRING,
                    key.stream_id.c_str(), kLoanFrameIdField, G_TYPE_INT64,
                    static_cast<gint64>(key.frame_id), nullptr);
  if (key.input_seq >= 0) {
    gst_structure_set(s, "neat-latest-mux-loan-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(key.input_seq), nullptr);
  }
  if (key.orig_input_seq >= 0) {
    gst_structure_set(s, "neat-latest-mux-loan-orig-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(key.orig_input_seq), nullptr);
  }
  return true;
}

bool read_latest_mux_loan_key(GstBuffer* buffer, LoanKey* key) {
  if (!buffer || !key) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    return false;
  }
  gboolean valid = FALSE;
  if (gst_structure_get_boolean(s, kLoanValidField, &valid) != TRUE || valid != TRUE) {
    return false;
  }
  const char* stream = gst_structure_get_string(s, kLoanStreamIdField);
  guint64 namespace_id = 0;
  gint64 frame = -1;
  if (gst_structure_get_uint64(s, kLoanNamespaceField, &namespace_id) != TRUE ||
      namespace_id == 0 || !stream || !*stream ||
      gst_structure_get_int64(s, kLoanFrameIdField, &frame) != TRUE || frame < 0) {
    return false;
  }
  key->namespace_id = static_cast<std::uint64_t>(namespace_id);
  key->stream_id = stream;
  key->frame_id = static_cast<std::int64_t>(frame);
  gint64 input_seq = -1;
  if (gst_structure_get_int64(s, "neat-latest-mux-loan-input-seq", &input_seq) == TRUE) {
    key->input_seq = static_cast<std::int64_t>(input_seq);
  }
  gint64 orig_input_seq = -1;
  if (gst_structure_get_int64(s, "neat-latest-mux-loan-orig-input-seq", &orig_input_seq) == TRUE) {
    key->orig_input_seq = static_cast<std::int64_t>(orig_input_seq);
  }
  return true;
}

bool loan_entry_credit_released(const std::shared_ptr<LoanEntry>& entry) {
  if (!entry) {
    return true;
  }
  std::lock_guard<std::mutex> lock(entry->resources_mutex);
  return entry->credit_released;
}

struct ClaimedLoanCreditRelease {
  std::shared_ptr<StreamLoanState> state;
  GstLatestByStreamMux* wake_mux = nullptr;
  bool claimed = false;
};

ClaimedLoanCreditRelease claim_loan_entry_credit_release(const std::shared_ptr<LoanEntry>& entry) {
  ClaimedLoanCreditRelease release;
  if (!entry) {
    return release;
  }
  std::lock_guard<std::mutex> lock(entry->resources_mutex);
  if (entry->credit_released) {
    return release;
  }
  entry->credit_released = true;
  release.state = entry->state;
  if (entry->mux) {
    release.wake_mux = GST_LATEST_BY_STREAM_MUX(gst_object_ref(GST_OBJECT(entry->mux)));
  }
  release.claimed = true;
  return release;
}

void apply_claimed_loan_credit_release(ClaimedLoanCreditRelease release) {
  if (!release.claimed) {
    return;
  }
  if (loan_state_enabled(release.state)) {
    release_loan_state(release.state);
  }
  if (release.wake_mux) {
    g_mutex_lock(&release.wake_mux->lock);
    g_cond_broadcast(&release.wake_mux->cond);
    g_mutex_unlock(&release.wake_mux->lock);
    gst_object_unref(GST_OBJECT(release.wake_mux));
  }
}

bool release_loan_entry_credit(const std::shared_ptr<LoanEntry>& entry) {
  ClaimedLoanCreditRelease release = claim_loan_entry_credit_release(entry);
  const bool claimed = release.claimed;
  apply_claimed_loan_credit_release(std::move(release));
  return claimed;
}

bool loan_entry_belongs_to_mux(const std::shared_ptr<LoanEntry>& entry, GstLatestByStreamMux* mux) {
  if (!entry || !mux) {
    return false;
  }
  std::lock_guard<std::mutex> lock(entry->resources_mutex);
  return entry->mux == mux;
}

void release_loan_entry_resources(const std::shared_ptr<LoanEntry>& entry, bool by_output) {
  if (!entry) {
    return;
  }

  // This is a no-op when a known raw-input completion boundary already
  // returned admission.  The terminal/teardown outcome counters intentionally
  // remain final-entry counters rather than credit-release timing counters.
  (void)release_loan_entry_credit(entry);
  if (loan_state_enabled(entry->state)) {
    if (by_output) {
      entry->state->released_by_output.fetch_add(1, std::memory_order_relaxed);
    } else {
      entry->state->released_without_output.fetch_add(1, std::memory_order_relaxed);
    }
  }

  GstLatestByStreamMux* owned_mux = nullptr;
  {
    std::lock_guard<std::mutex> lock(entry->resources_mutex);
    owned_mux = std::exchange(entry->mux, nullptr);
  }
  if (owned_mux) {
    gst_object_unref(GST_OBJECT(owned_mux));
  }
}

void release_loan_entry(const std::shared_ptr<LoanEntry>& entry, bool by_output) {
  if (!entry) {
    return;
  }
  bool expected = false;
  if (!entry->released.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
    return;
  }
  entry->ref_count = 0;
  release_loan_entry_resources(entry, by_output);
}

void erase_loan_keys_for_entry_locked(const std::shared_ptr<LoanEntry>& entry) {
  if (!entry) {
    return;
  }
  auto& registry = loan_registry();
  for (auto it = registry.begin(); it != registry.end();) {
    if (it->second == entry) {
      it = registry.erase(it);
    } else {
      ++it;
    }
  }
}

bool consume_loan_ref_locked(const std::shared_ptr<LoanEntry>& entry) {
  if (!entry || entry->released.load(std::memory_order_acquire)) {
    return false;
  }
  if (entry->ref_count > 1U) {
    --entry->ref_count;
    return false;
  }
  entry->ref_count = 0;
  entry->released.store(true, std::memory_order_release);
  erase_loan_keys_for_entry_locked(entry);
  return true;
}

bool consume_all_loan_refs_locked(const std::shared_ptr<LoanEntry>& entry) {
  if (!entry || entry->released.load(std::memory_order_acquire)) {
    return false;
  }
  entry->ref_count = 0;
  entry->released.store(true, std::memory_order_release);
  erase_loan_keys_for_entry_locked(entry);
  return true;
}

void release_all_loans_for_mux(GstLatestByStreamMux* self) {
  if (!self) {
    return;
  }
  std::vector<std::shared_ptr<LoanEntry>> entries;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    for (auto it = registry.begin(); it != registry.end();) {
      if (loan_entry_belongs_to_mux(it->second, self)) {
        entries.push_back(it->second);
        it = registry.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (const auto& entry : entries) {
    release_loan_entry(entry, /*by_output=*/false);
  }
}

void restore_loan_identity_and_timing_on_buffer(const std::shared_ptr<LoanEntry>& entry,
                                                const LoanKey& key, GstBuffer* buffer) {
  if (!entry || !buffer) {
    return;
  }

  if (!entry->timing.empty()) {
    // Restore native PTS/DTS/duration first. This preserves the old behavior
    // even when a caller supplies a non-writable buffer and GstSimaMeta cannot
    // be updated below.
    (void)simaai::neat::pipeline_internal::sample_timing_gst_detail::apply_to_buffer_header(
        buffer, entry->timing);
  }

  // Replacement-stage output pools may recycle scalar GstSimaMeta fields from
  // an older stream even though the current result is valid.  Once terminal
  // matching has selected the authoritative registry entry, rewrite both the
  // public routing fields and the private (now-completed) mux key from that
  // entry.  This keeps the Sample observed by GstAppSink consistent with the
  // loan whose timing was restored and leaves the canonical private fields as
  // diagnostics while marking the already-consumed key invalid downstream.
  GstStructure* s = nullptr;
  if (!ensure_sima_meta_structure_mutable(buffer, &s) || !s) {
    return;
  }
  // Use the same writable GstStructure transaction for timing and identity.
  // Reacquiring timing metadata through write_sample_timing_to_gst_buffer()
  // used to copy the complete ProcessCVU structure once, then this identity
  // pass copied it a second time on GStreamer versions without a public
  // structure-mutability macro.
  if (!entry->timing.empty()) {
    (void)simaai::neat::pipeline_internal::sample_timing_gst_detail::write_to_structure(
        s, entry->timing);
  }
  gst_structure_set(s, "stream-id", G_TYPE_STRING, key.stream_id.c_str(), "orig-stream-id",
                    G_TYPE_STRING, key.stream_id.c_str(), "frame-id", G_TYPE_INT64,
                    static_cast<gint64>(key.frame_id), kLoanValidField, G_TYPE_BOOLEAN, FALSE,
                    kLoanNamespaceField, G_TYPE_UINT64, static_cast<guint64>(key.namespace_id),
                    kLoanStreamIdField, G_TYPE_STRING, key.stream_id.c_str(), kLoanFrameIdField,
                    G_TYPE_INT64, static_cast<gint64>(key.frame_id), nullptr);
  if (key.input_seq >= 0) {
    gst_structure_set(s, "input-seq", G_TYPE_INT64, static_cast<gint64>(key.input_seq),
                      "neat-latest-mux-loan-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(key.input_seq), nullptr);
  } else {
    gst_structure_remove_fields(s, "input-seq", "neat-latest-mux-loan-input-seq", nullptr);
  }
  if (key.orig_input_seq >= 0) {
    gst_structure_set(s, "orig-input-seq", G_TYPE_INT64, static_cast<gint64>(key.orig_input_seq),
                      "neat-latest-mux-loan-orig-input-seq", G_TYPE_INT64,
                      static_cast<gint64>(key.orig_input_seq), nullptr);
  } else {
    gst_structure_remove_fields(s, "orig-input-seq", "neat-latest-mux-loan-orig-input-seq",
                                nullptr);
  }
}

bool read_stream_key(GstBuffer* buffer, std::string* stream_id);
bool read_stream_frame_key(GstBuffer* buffer, std::string* stream_id, std::int64_t* frame_id,
                           std::int64_t* input_seq, std::int64_t* orig_input_seq);

bool release_dropped_loan_for_key_impl(const LoanKey& key, std::uint64_t sequence,
                                       const char* mode);

GType latest_mux_loan_guard_meta_api_get_type() {
  static const GType type = [] {
    // This lifecycle marker is independent of the source buffer's memory.
    // Tagging it as a memory reference makes deep buffer copies omit it, which
    // would release decoder credit before a copied model output completes.
    static const gchar* tags[] = {nullptr};
    return gst_meta_api_type_register("GstLatestMuxLoanGuardMetaAPI", tags);
  }();
  return type;
}

gboolean latest_mux_loan_guard_meta_init(GstMeta* meta, gpointer, GstBuffer*) {
  auto* guard_meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(meta);
  new (&guard_meta->guard) std::shared_ptr<LoanDropGuard>();
  guard_meta->retired = false;
  guard_meta->terminal_claimed = false;
  return TRUE;
}

void retire_loan_guard_meta_carrier(GstLatestMuxLoanGuardMeta* meta, const char* mode) {
  if (!meta || !meta->guard) {
    return;
  }
  const std::shared_ptr<LoanDropGuard> guard = meta->guard;
  bool release_remaining = false;
  std::uint64_t previous = 0;
  bool armed = false;
  {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (meta->retired) {
      return;
    }
    meta->retired = true;
    previous = guard->carriers;
    if (guard->carriers > 0) {
      --guard->carriers;
    }
    if (guard->carriers == 0 && guard->armed) {
      guard->armed = false;
      release_remaining = true;
    }
    armed = guard->armed;
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr, "[latestmux][loan] guard-%s frame=%lld carriers=%llu->%llu armed=%d\n",
                 mode ? mode : "retire", static_cast<long long>(guard->key.frame_id),
                 static_cast<unsigned long long>(previous),
                 static_cast<unsigned long long>(previous > 0 ? previous - 1 : 0), armed ? 1 : 0);
  }
  if (release_remaining) {
    (void)release_dropped_loan_for_key_impl(guard->key, guard->sequence,
                                            mode ? mode : "buffer-lifetime-drop");
  }
}

void latest_mux_loan_guard_meta_free(GstMeta* raw_meta, GstBuffer*) {
  auto* meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(raw_meta);
  retire_loan_guard_meta_carrier(meta, "buffer-lifetime-drop");
  meta->guard.~shared_ptr();
}

const GstMetaInfo* latest_mux_loan_guard_meta_get_info();

gboolean latest_mux_loan_guard_meta_transform(GstBuffer* destination, GstMeta* meta, GstBuffer*,
                                              GQuark, gpointer) {
  auto* source = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(meta);
  if (!destination || !source->guard) {
    return FALSE;
  }
  const std::shared_ptr<LoanDropGuard> guard = source->guard;
  std::lock_guard<std::mutex> lock(guard->mutex);
  if (source->retired || source->terminal_claimed || !guard->armed) {
    return FALSE;
  }
  auto* copy = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(
      gst_buffer_add_meta(destination, latest_mux_loan_guard_meta_get_info(), nullptr));
  if (!copy) {
    return FALSE;
  }
  copy->guard = guard;
  copy->terminal_claimed = false;
  const std::uint64_t previous = guard->carriers++;
  if (loan_debug_enabled()) {
    std::fprintf(stderr, "[latestmux][loan] guard-copy frame=%lld carriers=%llu->%llu\n",
                 static_cast<long long>(guard->key.frame_id),
                 static_cast<unsigned long long>(previous),
                 static_cast<unsigned long long>(previous + 1));
  }
  return TRUE;
}

const GstMetaInfo* latest_mux_loan_guard_meta_get_info() {
  static const GstMetaInfo* info =
      gst_meta_register(latest_mux_loan_guard_meta_api_get_type(), "GstLatestMuxLoanGuardMeta",
                        sizeof(GstLatestMuxLoanGuardMeta), latest_mux_loan_guard_meta_init,
                        latest_mux_loan_guard_meta_free, latest_mux_loan_guard_meta_transform);
  return info;
}

std::shared_ptr<LoanDropGuard> loan_drop_guard_for_buffer(GstBuffer* buffer) {
  if (!buffer) {
    return {};
  }
  auto* meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(
      gst_buffer_get_meta(buffer, latest_mux_loan_guard_meta_api_get_type()));
  return meta ? meta->guard : std::shared_ptr<LoanDropGuard>{};
}

struct LoanGuardTerminalClaim {
  std::shared_ptr<LoanDropGuard> guard;
  bool present = false;
  bool claimed = false;
};

LoanGuardTerminalClaim claim_loan_drop_guard_for_terminal(GstBuffer* buffer,
                                                          std::uint64_t namespace_hint) {
  LoanGuardTerminalClaim result;
  if (!buffer) {
    return result;
  }
  auto* meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(
      gst_buffer_get_meta(buffer, latest_mux_loan_guard_meta_api_get_type()));
  if (!meta) {
    return result;
  }
  result.present = true;
  result.guard = meta->guard;
  if (!result.guard) {
    return result;
  }
  std::lock_guard<std::mutex> lock(result.guard->mutex);
  if (meta->retired || meta->terminal_claimed || !result.guard->armed ||
      (namespace_hint != 0 && result.guard->key.namespace_id != namespace_hint)) {
    return result;
  }
  meta->terminal_claimed = true;
  result.claimed = true;
  return result;
}

bool attach_loan_drop_guard(GstBuffer* buffer, const LoanKey& key, std::uint64_t sequence) {
  if (!buffer || sequence == 0 || key.namespace_id == 0 || key.stream_id.empty() ||
      key.frame_id < 0 || loan_drop_guard_for_buffer(buffer)) {
    return false;
  }
  auto guard = std::make_shared<LoanDropGuard>();
  guard->key = key;
  guard->sequence = sequence;
  auto* meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(
      gst_buffer_add_meta(buffer, latest_mux_loan_guard_meta_get_info(), nullptr));
  if (!meta) {
    return false;
  }
  meta->guard = std::move(guard);
  return true;
}

void disarm_loan_drop_guard(GstBuffer* buffer, const LoanKey& key, std::uint64_t sequence) {
  const auto guard = loan_drop_guard_for_buffer(buffer);
  if (guard && guard->sequence == sequence && guard->key == key) {
    std::lock_guard<std::mutex> lock(guard->mutex);
    guard->armed = false;
  }
}

void retire_terminal_loan_guard_carrier(GstBuffer* buffer, const LoanKey& key,
                                        std::uint64_t sequence) {
  if (!buffer) {
    return;
  }
  auto* meta = reinterpret_cast<GstLatestMuxLoanGuardMeta*>(
      gst_buffer_get_meta(buffer, latest_mux_loan_guard_meta_api_get_type()));
  if (!meta || !meta->guard || meta->guard->sequence != sequence || !(meta->guard->key == key)) {
    return;
  }
  retire_loan_guard_meta_carrier(meta, "terminal");
}

bool release_dropped_loan_for_key_impl(const LoanKey& key, std::uint64_t sequence,
                                       const char* mode) {
  if (sequence == 0 || key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  bool final_release = false;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    const auto it = loan_registry().find(key);
    if (it == loan_registry().end() || !it->second || it->second->sequence != sequence) {
      return false;
    }
    entry = it->second;
    // No buffer carrying this loan remains, so no retained fan-out branch can
    // still reach a terminal completion. Finalize every logical reference in
    // one sequence-checked registry transaction.
    final_release = consume_all_loan_refs_locked(entry);
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] drop-release ns=%llu stream=%s frame=%lld sequence=%llu "
                 "mode=%s final=%d\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), static_cast<unsigned long long>(sequence),
                 mode ? mode : "buffer-drop", final_release ? 1 : 0);
  }
  if (final_release) {
    release_loan_entry_resources(entry, /*by_output=*/false);
  }
  return true;
}

bool release_loan_for_key_impl(const LoanKey& key, const char* mode,
                               GstBuffer* terminal_buffer = nullptr,
                               std::uint64_t expected_sequence = 0) {
  if (key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  bool final_release = false;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto it = registry.find(key);
    if (it == registry.end()) {
      return false;
    }
    entry = it->second;
    if (!entry || (expected_sequence != 0 && entry->sequence != expected_sequence)) {
      return false;
    }
    sequence = entry->sequence;
    final_release = consume_loan_ref_locked(entry);
  }
  // Terminal metadata mutation can allocate, copy, and rewrite GstMeta.  Keep
  // that work outside the process-wide registry lock: the selected entry and
  // key are immutable and the shared_ptr keeps their timing alive after a
  // final consume erases the registry entry.
  restore_loan_identity_and_timing_on_buffer(entry, key, terminal_buffer);
  if (final_release) {
    disarm_loan_drop_guard(terminal_buffer, key, sequence);
  } else {
    // This terminal has accounted for one logical fan-out reference. Remove
    // its physical carrier without disarming the guard shared by other copied
    // buffers, so a sibling dropped before the terminal can still release the
    // remaining retained references when its carrier is destroyed.
    retire_terminal_loan_guard_carrier(terminal_buffer, key, sequence);
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] release ns=%llu stream=%s frame=%lld input_seq=%lld "
                 "orig_input_seq=%lld mode=%s final=%d\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), static_cast<long long>(key.input_seq),
                 static_cast<long long>(key.orig_input_seq), mode ? mode : "key",
                 final_release ? 1 : 0);
  }
  if (final_release) {
    release_loan_entry_resources(entry, /*by_output=*/true);
  }
  return true;
}

G_DEFINE_TYPE_WITH_CODE(GstLatestByStreamMux, gst_latest_by_stream_mux, GST_TYPE_ELEMENT,
                        GST_DEBUG_CATEGORY_INIT(gst_latest_by_stream_mux_debug_category,
                                                "neatlatestbystreammux", 0,
                                                "Neat live latest-by-stream mux"));

enum {
  PROP_0,
  PROP_STREAM_IDS,
  PROP_STREAM_INFLIGHT_LIMITS,
  PROP_MAX_INFLIGHT_TOTAL,
};

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink_%u", GST_PAD_SINK, GST_PAD_REQUEST, GST_STATIC_CAPS_ANY);
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

bool debug_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

bool stats_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_STATS");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

std::uint64_t stats_interval() {
  static const std::uint64_t value = []() -> std::uint64_t {
    const gchar* env = g_getenv("SIMA_LATEST_MUX_STATS_EVERY");
    if (!env || !*env) {
      return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env) {
      return 0;
    }
    return static_cast<std::uint64_t>(parsed);
  }();
  return value;
}

bool read_stream_frame_key(GstBuffer* buffer, std::string* stream_id, std::int64_t* frame_id,
                           std::int64_t* input_seq = nullptr,
                           std::int64_t* orig_input_seq = nullptr) {
  if (!buffer || !stream_id || !frame_id) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    return false;
  }
  const char* stream = gst_structure_get_string(s, "orig-stream-id");
  if (!stream || !*stream) {
    stream = gst_structure_get_string(s, "stream-id");
  }
  if (!stream || !*stream) {
    return false;
  }
  *stream_id = stream;
  // Terminal replacement buffers may preserve the input sequence while
  // omitting or rewriting frame-id. Populate sequence outputs independently;
  // the boolean return continues to mean that the full stream+frame key needed
  // for mux-loan registration is present.
  if (input_seq) {
    gint64 seq = -1;
    if (gst_structure_get_int64(s, "input-seq", &seq) == TRUE) {
      *input_seq = static_cast<std::int64_t>(seq);
    }
  }
  if (orig_input_seq) {
    gint64 seq = -1;
    if (gst_structure_get_int64(s, "orig-input-seq", &seq) == TRUE) {
      *orig_input_seq = static_cast<std::int64_t>(seq);
    }
  }
  gint64 frame = -1;
  gboolean sample_frame_valid = FALSE;
  if (gst_structure_get_boolean(s, "sample-frame-id-valid", &sample_frame_valid) == TRUE &&
      sample_frame_valid == TRUE) {
    (void)gst_structure_get_int64(s, "sample-frame-id", &frame);
  }
  if (frame < 0) {
    (void)gst_structure_get_int64(s, "frame-id", &frame);
  }
  if (frame < 0) {
    return false;
  }
  *frame_id = static_cast<std::int64_t>(frame);
  return true;
}

bool read_stream_key(GstBuffer* buffer, std::string* stream_id) {
  if (!buffer || !stream_id) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    return false;
  }
  const char* stream = gst_structure_get_string(s, "orig-stream-id");
  if (!stream || !*stream) {
    stream = gst_structure_get_string(s, "stream-id");
  }
  if (!stream || !*stream) {
    return false;
  }
  *stream_id = stream;
  return true;
}

std::uint64_t register_loan_for_key(GstLatestByStreamMux* self,
                                    const std::shared_ptr<StreamLoanState>& state,
                                    const LoanKey& key, GstBuffer* buffer) {
  if (!self || !loan_state_enabled(state) || key.namespace_id == 0 || key.stream_id.empty() ||
      key.frame_id < 0) {
    return 0;
  }

  auto entry = std::make_shared<LoanEntry>();
  entry->mux = GST_LATEST_BY_STREAM_MUX(gst_object_ref(GST_OBJECT(self)));
  entry->state = state;
  entry->sequence = loan_sequence_counter().fetch_add(1, std::memory_order_relaxed);
  g_mutex_lock(&self->lock);
  entry->terminal_replacing = !self->lifetime_guard_enabled;
  g_mutex_unlock(&self->lock);
  entry->timing.frame_id = key.frame_id;
  if (buffer) {
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))) {
      entry->timing.pts_ns = static_cast<std::uint64_t>(GST_BUFFER_PTS(buffer));
    }
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))) {
      entry->timing.dts_ns = static_cast<std::uint64_t>(GST_BUFFER_DTS(buffer));
    }
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))) {
      entry->timing.duration_ns = static_cast<std::uint64_t>(GST_BUFFER_DURATION(buffer));
    }
  }

  std::shared_ptr<LoanEntry> replaced;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto it = registry.find(key);
    if (it != registry.end()) {
      replaced = it->second;
      it->second = entry;
    } else {
      registry.emplace(std::move(key), entry);
    }
  }
  state->registered.fetch_add(1, std::memory_order_relaxed);
  if (replaced) {
    release_loan_entry(replaced, /*by_output=*/false);
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] register ns=%llu stream=%s frame=%lld input_seq=%lld "
                 "orig_input_seq=%lld inflight=%d limit=%d\n",
                 static_cast<unsigned long long>(self->loan_namespace), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), static_cast<long long>(key.input_seq),
                 static_cast<long long>(key.orig_input_seq),
                 state->gate ? state->gate->inflight() : 0,
                 state->gate ? state->gate->credit_limit() : 0);
  }
  return entry->sequence;
}

void release_acquired_loan_without_output(const std::shared_ptr<StreamLoanState>& state) {
  if (!loan_state_enabled(state)) {
    return;
  }
  release_loan_state(state);
  state->released_without_output.fetch_add(1, std::memory_order_relaxed);
}

GstBuffer* detach_pending_locked(PendingSlot* slot) {
  // The caller holds the owning mux lock. Final unref is intentionally left to
  // the caller after unlocking because GstMiniObject destroy callbacks may
  // re-enter the mux.
  if (!slot) {
    return nullptr;
  }
  slot->ready_ticket = 0;
  slot->has_ready_ticket = false;
  return std::exchange(slot->pending, nullptr);
}

void unref_buffers(const std::vector<GstBuffer*>& buffers) {
  for (GstBuffer* buffer : buffers) {
    if (buffer) {
      gst_buffer_unref(buffer);
    }
  }
}

void clear_slot_caps(PendingSlot* slot) {
  if (!slot || !slot->caps) {
    return;
  }
  gst_caps_unref(slot->caps);
  slot->caps = nullptr;
}

std::string stream_id_for_slot(GstLatestByStreamMux* self, const PendingSlot* slot) {
  if (!slot) {
    return "stream0";
  }
  std::string out;
  if (self) {
    g_mutex_lock(&self->lock);
    if (slot->index < self->stream_ids.size() && !self->stream_ids[slot->index].empty()) {
      out = self->stream_ids[slot->index];
    }
    g_mutex_unlock(&self->lock);
  }
  if (!out.empty()) {
    return out;
  }
  return "stream" + std::to_string(slot->index);
}

void stamp_stream_meta(GstLatestByStreamMux* self, GstBuffer** inout, const PendingSlot* slot) {
  if (!inout || !*inout || !slot) {
    return;
  }

  GstBuffer* buffer = *inout;
  const std::string stream_id = stream_id_for_slot(self, slot);
  std::optional<int64_t> frame_id;
  if (GstCustomMeta* existing = gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
    if (GstStructure* s = gst_custom_meta_get_structure(existing)) {
      gint64 existing_frame = -1;
      if (gst_structure_get_int64(s, "frame-id", &existing_frame) == TRUE && existing_frame >= 0) {
        frame_id = static_cast<int64_t>(existing_frame);
      }
    }
  }
  if (!frame_id.has_value()) {
    frame_id = simaai::neat::next_input_frame_id();
  }

  GstBuffer* writable = gst_buffer_make_writable(buffer);
  if (!writable) {
    return;
  }
  buffer = writable;

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (s) {
    if (!gst_structure_has_field(s, "frame-id")) {
      gst_structure_set(s, "frame-id", G_TYPE_INT64, static_cast<gint64>(*frame_id), nullptr);
    }
    gst_structure_set(s, "stream-id", G_TYPE_STRING, stream_id.c_str(), nullptr);
    gst_structure_set(s, "orig-stream-id", G_TYPE_STRING, stream_id.c_str(), nullptr);
  }

  simaai::neat::SampleTimingOverrides timing;
  timing.frame_id = frame_id;
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))) {
    timing.pts_ns = static_cast<std::uint64_t>(GST_BUFFER_PTS(buffer));
  }
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))) {
    timing.dts_ns = static_cast<std::uint64_t>(GST_BUFFER_DTS(buffer));
  }
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))) {
    timing.duration_ns = static_cast<std::uint64_t>(GST_BUFFER_DURATION(buffer));
  }
  (void)simaai::neat::write_sample_timing_to_gst_buffer(buffer, timing);
  *inout = buffer;
}

bool all_slots_drained_eos_locked(const GstLatestByStreamMux* self) {
  if (!self || self->slots.empty()) {
    return false;
  }
  for (const PendingSlot* slot : self->slots) {
    // EOS is serialized after the final chain call on each sink pad, but the
    // mux may still own that pad's last buffer while terminal credit is
    // exhausted.  Do not let the aggregate EOS overtake such a tail frame.
    if (slot && (!slot->eos || slot->pending)) {
      return false;
    }
  }
  return true;
}

PendingSlot* take_next_slot_locked(GstLatestByStreamMux* self, bool* loan_acquired) {
  if (loan_acquired) {
    *loan_acquired = false;
  }
  if (!self || self->slots.empty()) {
    return nullptr;
  }
  const guint n = static_cast<guint>(self->slots.size());
  PendingSlot* selected = nullptr;
  guint selected_idx = 0;
  for (guint offset = 0; offset < n; ++offset) {
    const guint idx = (self->rr_index + offset) % n;
    PendingSlot* slot = self->slots[idx];
    if (!slot || !slot->pending || !slot->has_ready_ticket) {
      continue;
    }

    if (!loan_state_has_credit(slot->loan_state)) {
      slot->no_credit_skips.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    // Max-min service fairness gives an established lagging stream more than
    // one prompt turn when spare capacity exists. This closes accumulated
    // phase/credit deficits instead of merely preserving them forever.
    // ready_ticket keeps equal-count streams deterministic and preserves the
    // age of an empty -> pending transition across latest-only replacement.
    if (!selected || slot->fair_service_count < selected->fair_service_count ||
        (slot->fair_service_count == selected->fair_service_count &&
         slot->ready_ticket < selected->ready_ticket)) {
      selected = slot;
      selected_idx = idx;
    }
  }

  if (!selected) {
    return nullptr;
  }

  if (loan_state_enabled(selected->loan_state)) {
    // Only this mux worker acquires the shared total gate and each per-stream
    // gate. The availability check above therefore cannot become false
    // (terminal threads only release credit), but keep acquisition defensive
    // in case that ownership contract changes.
    if (!try_acquire_loan_state(selected->loan_state)) {
      selected->no_credit_skips.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (loan_acquired) {
      *loan_acquired = true;
    }
  }

  self->rr_index = (selected_idx + 1U) % n;
  selected->ready_ticket = 0;
  selected->has_ready_ticket = false;
  ++selected->fair_service_count;
  self->service_frontier = std::max(self->service_frontier, selected->fair_service_count);
  ++selected->emitted;
  return selected;
}

GstFlowReturn push_buffer(GstLatestByStreamMux* self, GstBuffer* buffer) {
  if (!self || !buffer) {
    return GST_FLOW_OK;
  }
  return gst_pad_push(self->srcpad, buffer);
}

void print_slot_stats(GstLatestByStreamMux* self, const char* reason, bool once);

gpointer worker_main(gpointer data) {
  auto* self = GST_LATEST_BY_STREAM_MUX(data);
  if (stats_enabled()) {
    std::fprintf(stderr, "[latestmux][stats] worker-start every=%llu\n",
                 static_cast<unsigned long long>(stats_interval()));
    std::fflush(stderr);
  }
  while (true) {
    GstBuffer* buffer = nullptr;
    GstCaps* caps = nullptr;
    std::shared_ptr<StreamLoanState> loan_state;
    bool loan_acquired = false;
    bool push_eos = false;

    g_mutex_lock(&self->lock);
    while (!self->stopping) {
      if (self->flushing) {
        g_cond_wait(&self->cond, &self->lock);
        continue;
      }
      PendingSlot* slot = take_next_slot_locked(self, &loan_acquired);
      if (slot && slot->pending) {
        buffer = slot->pending;
        slot->pending = nullptr;
        // Publish the empty-slot transition before the synchronous downstream
        // push so state and pad teardown waiters can observe the ownership move.
        g_cond_broadcast(&self->cond);
        loan_state = slot->loan_state;
        if (slot->caps) {
          caps = gst_caps_ref(slot->caps);
        }
        break;
      }
      if (all_slots_drained_eos_locked(self)) {
        push_eos = true;
        self->stopping = true;
        break;
      }
      g_cond_wait(&self->cond, &self->lock);
    }
    const bool done = self->stopping && !buffer && !push_eos;
    g_mutex_unlock(&self->lock);

    if (buffer) {
      bool push_caps = false;
      bool push_segment = false;
      GstSegment segment_copy;
      g_mutex_lock(&self->lock);
      if (caps && (!self->current_caps || !gst_caps_is_equal(caps, self->current_caps))) {
        if (self->current_caps) {
          gst_caps_unref(self->current_caps);
        }
        self->current_caps = gst_caps_ref(caps);
        push_caps = true;
      }
      if (!self->segment_sent && self->have_pending_segment) {
        gst_segment_copy_into(&self->pending_segment, &segment_copy);
        self->segment_sent = true;
        push_segment = true;
      }
      g_mutex_unlock(&self->lock);
      if (push_caps) {
        (void)gst_pad_push_event(self->srcpad, gst_event_new_caps(caps));
      }
      if (push_segment) {
        (void)gst_pad_push_event(self->srcpad, gst_event_new_segment(&segment_copy));
      }
      if (caps) {
        gst_caps_unref(caps);
      }
      std::string stream_id;
      std::int64_t frame_id = -1;
      std::int64_t input_seq = -1;
      std::int64_t orig_input_seq = -1;
      bool loan_registered = false;
      if (loan_acquired) {
        if (read_stream_frame_key(buffer, &stream_id, &frame_id, &input_seq, &orig_input_seq)) {
          const LoanKey loan_key{self->loan_namespace, stream_id, frame_id, input_seq,
                                 orig_input_seq};
          if (stamp_latest_mux_loan_key(buffer, loan_key)) {
            const std::uint64_t sequence =
                register_loan_for_key(self, loan_state, loan_key, buffer);
            bool lifetime_guard_enabled = true;
            g_mutex_lock(&self->lock);
            lifetime_guard_enabled = self->lifetime_guard_enabled;
            g_mutex_unlock(&self->lock);
            if (sequence == 0) {
              release_acquired_loan_without_output(loan_state);
            } else if (!lifetime_guard_enabled ||
                       attach_loan_drop_guard(buffer, loan_key, sequence)) {
              loan_registered = true;
            } else {
              (void)release_dropped_loan_for_key_impl(loan_key, sequence, "guard-attach-failure");
            }
          } else {
            release_acquired_loan_without_output(loan_state);
            if (loan_debug_enabled()) {
              std::fprintf(stderr,
                           "[latestmux][loan] failed to stamp loan key stream=%s frame=%lld; "
                           "released credit\n",
                           stream_id.c_str(), static_cast<long long>(frame_id));
            }
          }
        } else {
          if (loan_state) {
            loan_state->missing_key.fetch_add(1, std::memory_order_relaxed);
          }
          release_acquired_loan_without_output(loan_state);
          if (loan_debug_enabled()) {
            std::fprintf(stderr, "[latestmux][loan] missing stream/frame key; released credit\n");
          }
        }
      }
      const std::uint64_t interval = stats_interval();
      if (stats_enabled() && interval > 0) {
        const std::uint64_t pushes = ++self->stats_pushes;
        if (pushes % interval == 0) {
          print_slot_stats(self, "periodic", /*once=*/false);
        }
      }
      const GstFlowReturn flow = push_buffer(self, buffer);
      if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING) {
        GST_WARNING_OBJECT(self, "downstream push returned %s", gst_flow_get_name(flow));
      }
      if (loan_registered && flow != GST_FLOW_OK) {
        (void)release_loan_for_key_impl(
            LoanKey{self->loan_namespace, stream_id, frame_id, input_seq, orig_input_seq},
            "push-failure");
      }
      continue;
    }
    if (push_eos) {
      gst_pad_push_event(self->srcpad, gst_event_new_eos());
      continue;
    }
    if (done) {
      break;
    }
  }
  return nullptr;
}

GstFlowReturn sink_chain(GstPad* pad, GstObject* parent, GstBuffer* buffer) {
  auto* self = GST_LATEST_BY_STREAM_MUX(parent);
  auto* slot = static_cast<PendingSlot*>(gst_pad_get_element_private(pad));
  if (!self || !slot || !buffer) {
    if (buffer) {
      gst_buffer_unref(buffer);
    }
    return GST_FLOW_ERROR;
  }

  g_mutex_lock(&self->lock);
  if (slot->releasing || self->flushing || self->stopping) {
    g_mutex_unlock(&self->lock);
    gst_buffer_unref(buffer);
    return GST_FLOW_FLUSHING;
  }
  ++slot->active_chains;
  g_mutex_unlock(&self->lock);
  struct ChainGuard {
    GstLatestByStreamMux* mux;
    PendingSlot* slot;
    ~ChainGuard() {
      g_mutex_lock(&mux->lock);
      if (slot->active_chains > 0) {
        --slot->active_chains;
      }
      g_cond_broadcast(&mux->cond);
      g_mutex_unlock(&mux->lock);
    }
  } chain_guard{self, slot};

  stamp_stream_meta(self, &buffer, slot);

  GstBuffer* replaced = nullptr;
  g_mutex_lock(&self->lock);
  if (slot->releasing || self->flushing || self->stopping) {
    g_mutex_unlock(&self->lock);
    gst_buffer_unref(buffer);
    return GST_FLOW_FLUSHING;
  }
  const std::uint64_t received_before = slot->received.fetch_add(1, std::memory_order_relaxed);
  if (!slot->fairness_active) {
    // A first frame (or the first frame after STREAM_START) joins at the
    // current maximum service count. It owes no work from before it existed.
    slot->fair_service_count = self->service_frontier;
    slot->fairness_active = true;
  }
  if (!slot->pending) {
    slot->ready_ticket = self->next_ready_ticket++;
    slot->has_ready_ticket = true;
  }
  if (slot->pending) {
    slot->replaced.fetch_add(1, std::memory_order_relaxed);
  }
  if (stats_enabled() && received_before == 0) {
    const char* stream_id = nullptr;
    if (slot->index < self->stream_ids.size() && !self->stream_ids[slot->index].empty()) {
      stream_id = self->stream_ids[slot->index].c_str();
    }
    std::fprintf(stderr, "[latestmux][stats] first-chain slot=%u stream=%s\n", slot->index,
                 stream_id ? stream_id : "unknown");
    std::fflush(stderr);
  }
  replaced = std::exchange(slot->pending, buffer); // take ownership from chain
  g_cond_signal(&self->cond);
  g_mutex_unlock(&self->lock);
  if (replaced) {
    gst_buffer_unref(replaced);
  }
  return GST_FLOW_OK;
}

gboolean forward_event_once(GstLatestByStreamMux* self, GstEvent* event, bool* sent_flag) {
  if (!self || !event || !sent_flag) {
    if (event) {
      gst_event_unref(event);
    }
    return false;
  }
  bool should_push = false;
  g_mutex_lock(&self->lock);
  if (!*sent_flag) {
    *sent_flag = true;
    should_push = true;
  }
  g_mutex_unlock(&self->lock);
  if (should_push) {
    return gst_pad_push_event(self->srcpad, event) == TRUE;
  }
  gst_event_unref(event);
  return true;
}

gboolean sink_event(GstPad* pad, GstObject* parent, GstEvent* event) {
  auto* self = GST_LATEST_BY_STREAM_MUX(parent);
  auto* slot = static_cast<PendingSlot*>(gst_pad_get_element_private(pad));
  if (!self || !slot || !event) {
    if (event) {
      gst_event_unref(event);
    }
    return false;
  }

  switch (GST_EVENT_TYPE(event)) {
  case GST_EVENT_STREAM_START: {
    g_mutex_lock(&self->lock);
    // A new upstream stream epoch must not inherit an arbitrarily old deficit
    // from a disconnected predecessor using the same request pad. Apply the
    // new frontier on its first buffer; an old pending buffer, if any, keeps
    // the predecessor's service count until then.
    slot->fairness_active = false;
    g_mutex_unlock(&self->lock);
    gst_event_unref(event);
    const std::string stream_id = std::string(GST_ELEMENT_NAME(self)) + ":src";
    return forward_event_once(self, gst_event_new_stream_start(stream_id.c_str()),
                              &self->stream_start_sent);
  }
  case GST_EVENT_CAPS: {
    GstCaps* caps = nullptr;
    gst_event_parse_caps(event, &caps);
    if (caps) {
      g_mutex_lock(&self->lock);
      clear_slot_caps(slot);
      slot->caps = gst_caps_ref(caps);
      g_mutex_unlock(&self->lock);
      gst_event_unref(event);
      return TRUE;
    }
    break;
  }
  case GST_EVENT_SEGMENT: {
    const GstSegment* segment = nullptr;
    gst_event_parse_segment(event, &segment);
    if (segment) {
      g_mutex_lock(&self->lock);
      gst_segment_copy_into(segment, &self->pending_segment);
      self->have_pending_segment = true;
      g_cond_broadcast(&self->cond);
      g_mutex_unlock(&self->lock);
      gst_event_unref(event);
      return TRUE;
    }
    break;
  }
  case GST_EVENT_FLUSH_START: {
    std::vector<GstBuffer*> detached;
    g_mutex_lock(&self->lock);
    self->flushing = true;
    detached.reserve(self->slots.size());
    for (PendingSlot* s : self->slots) {
      if (GstBuffer* buffer = detach_pending_locked(s)) {
        detached.push_back(buffer);
      }
      clear_slot_caps(s);
    }
    if (self->current_caps) {
      gst_caps_unref(self->current_caps);
      self->current_caps = nullptr;
    }
    self->segment_sent = false;
    self->have_pending_segment = false;
    gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
    g_cond_broadcast(&self->cond);
    g_mutex_unlock(&self->lock);
    unref_buffers(detached);
    release_all_loans_for_mux(self);
    return gst_pad_push_event(self->srcpad, event) == TRUE;
  }
  case GST_EVENT_FLUSH_STOP:
    g_mutex_lock(&self->lock);
    self->flushing = false;
    self->segment_sent = false;
    self->have_pending_segment = false;
    gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
    for (PendingSlot* s : self->slots) {
      if (s) {
        s->eos = false;
      }
    }
    g_cond_broadcast(&self->cond);
    g_mutex_unlock(&self->lock);
    return gst_pad_push_event(self->srcpad, event) == TRUE;
  case GST_EVENT_EOS:
    gst_event_unref(event);
    g_mutex_lock(&self->lock);
    slot->eos = true;
    g_cond_broadcast(&self->cond);
    g_mutex_unlock(&self->lock);
    return true;
  default:
    break;
  }

  return gst_pad_event_default(pad, parent, event);
}

GstPad* request_new_pad(GstElement* element, GstPadTemplate* templ, const gchar* req_name,
                        const GstCaps*) {
  auto* self = GST_LATEST_BY_STREAM_MUX(element);
  if (!self || !templ) {
    return nullptr;
  }

  guint requested_index = 0;
  bool have_requested_index = false;
  if (req_name && g_str_has_prefix(req_name, "sink_")) {
    guint parsed = 0;
    bool ok = false;
    for (const char* p = req_name + 5; *p; ++p) {
      if (*p < '0' || *p > '9') {
        ok = false;
        break;
      }
      ok = true;
      parsed = parsed * 10U + static_cast<guint>(*p - '0');
    }
    if (ok) {
      requested_index = parsed;
      have_requested_index = true;
    }
  }

  // Terminal loans require a completion boundary that calls the public
  // release helper. The fused graph renderer always supplies an explicit
  // per-stream list; standalone element users that omit it must remain
  // ordinary GStreamer pipelines rather than silently stalling after four
  // buffers with no release probe.
  int stream_inflight_limit = 0;
  HolderLoanGatePtr total_loan_gate;
  g_mutex_lock(&self->lock);
  const guint index = have_requested_index ? requested_index : self->next_pad_index;
  self->next_pad_index = std::max(self->next_pad_index, index + 1U);
  if (index < self->stream_inflight_limits.size()) {
    stream_inflight_limit = self->stream_inflight_limits[index];
  }
  total_loan_gate = self->total_loan_gate;
  g_mutex_unlock(&self->lock);

  gchar* name = nullptr;
  if (req_name && *req_name) {
    name = g_strdup(req_name);
  } else {
    name = g_strdup_printf("sink_%u", index);
  }

  GstPad* pad = gst_pad_new_from_template(templ, name);
  g_free(name);
  if (!pad) {
    return nullptr;
  }

  auto* slot = new PendingSlot();
  slot->pad = pad;
  slot->index = index;
  slot->loan_state = make_stream_loan_state(stream_inflight_limit, std::move(total_loan_gate));
  gst_pad_set_element_private(pad, slot);
  gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(sink_chain));
  gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));

  g_mutex_lock(&self->lock);
  self->slots.push_back(slot);
  g_mutex_unlock(&self->lock);

  gst_element_add_pad(element, pad);
  if (debug_enabled()) {
    GST_INFO_OBJECT(self, "requested %s stream=stream%u", GST_PAD_NAME(pad), index);
  }
  return pad;
}

void release_pad(GstElement* element, GstPad* pad) {
  auto* self = GST_LATEST_BY_STREAM_MUX(element);
  if (!self || !pad) {
    return;
  }
  auto* slot = static_cast<PendingSlot*>(gst_pad_get_element_private(pad));

  GstBuffer* detached = nullptr;
  g_mutex_lock(&self->lock);
  if (slot) {
    slot->releasing = true;
  }
  auto it = std::find(self->slots.begin(), self->slots.end(), slot);
  if (it != self->slots.end()) {
    self->slots.erase(it);
  }
  detached = detach_pending_locked(slot);
  clear_slot_caps(slot);
  g_cond_broadcast(&self->cond);
  while (slot && slot->active_chains > 0) {
    g_cond_wait(&self->cond, &self->lock);
  }
  g_mutex_unlock(&self->lock);
  if (detached) {
    gst_buffer_unref(detached);
  }

  gst_pad_set_element_private(pad, nullptr);
  gst_element_remove_pad(element, pad);
  delete slot;
}

void reset_slot_stats(PendingSlot* slot) {
  if (!slot) {
    return;
  }
  slot->received.store(0, std::memory_order_relaxed);
  slot->replaced.store(0, std::memory_order_relaxed);
  slot->no_credit_skips.store(0, std::memory_order_relaxed);
  if (slot->loan_state) {
    slot->loan_state->registered.store(0, std::memory_order_relaxed);
    slot->loan_state->released_by_output.store(0, std::memory_order_relaxed);
    slot->loan_state->released_without_output.store(0, std::memory_order_relaxed);
    slot->loan_state->missing_key.store(0, std::memory_order_relaxed);
  }
}

struct SlotStatsSnapshot {
  guint index = 0;
  std::string stream_id;
  guint64 received = 0;
  guint64 replaced = 0;
  guint64 emitted = 0;
  guint64 fair_service_count = 0;
  guint64 ready_ticket = 0;
  guint64 no_credit_skips = 0;
  bool pending = false;
  bool eos = false;
  std::uint64_t loans_registered = 0;
  std::uint64_t loans_released_by_output = 0;
  std::uint64_t loans_released_without_output = 0;
  std::uint64_t missing_key = 0;
  int loan_inflight = 0;
  int loan_limit = 0;
};

void print_slot_stats(GstLatestByStreamMux* self, const char* reason, bool once) {
  if (!self || !stats_enabled()) {
    return;
  }

  std::vector<SlotStatsSnapshot> rows;
  guint64 service_frontier = 0;
  g_mutex_lock(&self->lock);
  if (once && self->stats_reported) {
    g_mutex_unlock(&self->lock);
    return;
  }
  if (once) {
    self->stats_reported = true;
  }
  service_frontier = self->service_frontier;
  rows.reserve(self->slots.size());
  for (const PendingSlot* slot : self->slots) {
    if (!slot) {
      continue;
    }
    SlotStatsSnapshot row;
    row.index = slot->index;
    if (slot->index < self->stream_ids.size() && !self->stream_ids[slot->index].empty()) {
      row.stream_id = self->stream_ids[slot->index];
    } else {
      row.stream_id = "stream" + std::to_string(slot->index);
    }
    row.received = slot->received.load(std::memory_order_relaxed);
    row.replaced = slot->replaced.load(std::memory_order_relaxed);
    row.emitted = slot->emitted;
    row.fair_service_count = slot->fair_service_count;
    row.ready_ticket = slot->ready_ticket;
    row.no_credit_skips = slot->no_credit_skips.load(std::memory_order_relaxed);
    row.pending = slot->pending != nullptr;
    row.eos = slot->eos;
    if (slot->loan_state) {
      row.loans_registered = slot->loan_state->registered.load(std::memory_order_relaxed);
      row.loans_released_by_output =
          slot->loan_state->released_by_output.load(std::memory_order_relaxed);
      row.loans_released_without_output =
          slot->loan_state->released_without_output.load(std::memory_order_relaxed);
      row.missing_key = slot->loan_state->missing_key.load(std::memory_order_relaxed);
      if (slot->loan_state->gate) {
        row.loan_inflight = slot->loan_state->gate->inflight();
        row.loan_limit = slot->loan_state->gate->credit_limit();
      }
    }
    rows.push_back(std::move(row));
  }
  g_mutex_unlock(&self->lock);

  std::fprintf(stderr, "[latestmux][stats] reason=%s slots=%zu service_frontier=%llu\n",
               reason ? reason : "unknown", rows.size(),
               static_cast<unsigned long long>(service_frontier));
  for (const auto& row : rows) {
    std::fprintf(
        stderr,
        "[latestmux][stats] slot=%u stream=%s chain=%llu replaced=%llu emitted=%llu "
        "fair_service_count=%llu ready_ticket=%llu pending=%d eos=%d no_credit_skips=%llu "
        "loans_registered=%llu "
        "loans_released_output=%llu loans_released_without_output=%llu missing_key=%llu "
        "loan_inflight=%d loan_limit=%d\n",
        row.index, row.stream_id.c_str(), static_cast<unsigned long long>(row.received),
        static_cast<unsigned long long>(row.replaced), static_cast<unsigned long long>(row.emitted),
        static_cast<unsigned long long>(row.fair_service_count),
        static_cast<unsigned long long>(row.ready_ticket), row.pending ? 1 : 0, row.eos ? 1 : 0,
        static_cast<unsigned long long>(row.no_credit_skips),
        static_cast<unsigned long long>(row.loans_registered),
        static_cast<unsigned long long>(row.loans_released_by_output),
        static_cast<unsigned long long>(row.loans_released_without_output),
        static_cast<unsigned long long>(row.missing_key), row.loan_inflight, row.loan_limit);
  }
  std::fflush(stderr);
}

void print_slot_stats_once(GstLatestByStreamMux* self, const char* reason) {
  print_slot_stats(self, reason, /*once=*/true);
}

GstStateChangeReturn change_state(GstElement* element, GstStateChange transition) {
  auto* self = GST_LATEST_BY_STREAM_MUX(element);
  switch (transition) {
  case GST_STATE_CHANGE_READY_TO_PAUSED: {
    std::vector<GstBuffer*> detached;
    g_mutex_lock(&self->lock);
    self->stopping = false;
    self->flushing = false;
    self->stream_start_sent = false;
    self->segment_sent = false;
    self->have_pending_segment = false;
    self->stats_reported = false;
    self->stats_pushes = 0;
    self->rr_index = 0;
    self->next_ready_ticket = 1;
    self->service_frontier = 0;
    gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
    detached.reserve(self->slots.size());
    for (PendingSlot* slot : self->slots) {
      if (slot) {
        slot->eos = false;
        slot->emitted = 0;
        slot->fair_service_count = 0;
        slot->fairness_active = false;
        reset_slot_stats(slot);
        if (GstBuffer* buffer = detach_pending_locked(slot)) {
          detached.push_back(buffer);
        }
        clear_slot_caps(slot);
      }
    }
    if (self->current_caps) {
      gst_caps_unref(self->current_caps);
      self->current_caps = nullptr;
    }
    if (!self->worker) {
      self->worker = g_thread_new("neatlatestmux", worker_main, self);
    }
    g_mutex_unlock(&self->lock);
    unref_buffers(detached);
    break;
  }
  case GST_STATE_CHANGE_PAUSED_TO_READY: {
    // Mark the mux as stopping and wake its worker before the parent transition
    // deactivates pads.
    std::vector<GstBuffer*> detached;
    g_mutex_lock(&self->lock);
    self->stopping = true;
    detached.reserve(self->slots.size());
    for (PendingSlot* slot : self->slots) {
      if (GstBuffer* buffer = detach_pending_locked(slot)) {
        detached.push_back(buffer);
      }
      clear_slot_caps(slot);
    }
    if (self->current_caps) {
      gst_caps_unref(self->current_caps);
      self->current_caps = nullptr;
    }
    self->segment_sent = false;
    self->have_pending_segment = false;
    gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
    g_cond_broadcast(&self->cond);
    g_mutex_unlock(&self->lock);
    unref_buffers(detached);
    break;
  }
  default:
    break;
  }

  GstStateChangeReturn ret =
      GST_ELEMENT_CLASS(gst_latest_by_stream_mux_parent_class)->change_state(element, transition);

  switch (transition) {
  case GST_STATE_CHANGE_PAUSED_TO_READY: {
    if (self->worker) {
      g_thread_join(self->worker);
      self->worker = nullptr;
    }
    // The worker may have acquired and registered its final loan while its
    // downstream push was completing. Join before the registry scan so no
    // post-scan entry can retain mux credit or a strong element reference.
    release_all_loans_for_mux(self);
    print_slot_stats_once(self, "paused-to-ready");
    break;
  }
  default:
    break;
  }
  return ret;
}

void finalize(GObject* object) {
  auto* self = GST_LATEST_BY_STREAM_MUX(object);
  std::vector<GstBuffer*> detached;
  g_mutex_lock(&self->lock);
  self->stopping = true;
  detached.reserve(self->slots.size());
  for (PendingSlot* slot : self->slots) {
    if (GstBuffer* buffer = detach_pending_locked(slot)) {
      detached.push_back(buffer);
    }
    clear_slot_caps(slot);
  }
  g_cond_broadcast(&self->cond);
  g_mutex_unlock(&self->lock);
  unref_buffers(detached);
  release_all_loans_for_mux(self);
  if (self->worker) {
    g_thread_join(self->worker);
    self->worker = nullptr;
  }
  print_slot_stats_once(self, "finalize");
  if (self->current_caps) {
    gst_caps_unref(self->current_caps);
    self->current_caps = nullptr;
  }
  for (PendingSlot* slot : self->slots) {
    clear_slot_caps(slot);
    delete slot;
  }
  self->slots.clear();
  self->stream_ids.clear();
  self->stream_inflight_limits.clear();
  self->total_loan_gate.reset();
  self->total_loan_gate.~HolderLoanGatePtr();
  self->stream_inflight_limits.~IntVector();
  self->stream_ids.~StringVector();
  self->slots.~SlotVector();
  g_cond_clear(&self->cond);
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_latest_by_stream_mux_parent_class)->finalize(object);
}

void parse_stream_ids(GstLatestByStreamMux* self, const gchar* csv) {
  if (!self) {
    return;
  }
  std::vector<std::string> parsed;
  if (csv && *csv) {
    gchar** parts = g_strsplit(csv, ",", -1);
    for (gchar** it = parts; it && *it; ++it) {
      gchar* stripped = g_strstrip(*it);
      parsed.emplace_back(stripped ? stripped : "");
    }
    g_strfreev(parts);
  }
  g_mutex_lock(&self->lock);
  self->stream_ids = std::move(parsed);
  g_mutex_unlock(&self->lock);
}

std::string join_stream_ids(GstLatestByStreamMux* self) {
  if (!self) {
    return {};
  }
  std::string out;
  g_mutex_lock(&self->lock);
  for (std::size_t i = 0; i < self->stream_ids.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += self->stream_ids[i];
  }
  g_mutex_unlock(&self->lock);
  return out;
}

void parse_stream_inflight_limits(GstLatestByStreamMux* self, const gchar* csv) {
  if (!self) {
    return;
  }
  std::vector<int> parsed;
  if (csv && *csv) {
    gchar** parts = g_strsplit(csv, ",", -1);
    for (gchar** it = parts; it && *it; ++it) {
      gchar* stripped = g_strstrip(*it);
      char* end = nullptr;
      const long value = stripped ? std::strtol(stripped, &end, 10) : 0;
      if (!stripped || end == stripped || *end != '\0' || value < 0 ||
          value > std::numeric_limits<int>::max()) {
        g_strfreev(parts);
        GST_ERROR_OBJECT(self,
                         "stream-inflight-limits must be a comma-separated list of non-negative "
                         "integers");
        return;
      }
      parsed.push_back(static_cast<int>(value));
    }
    g_strfreev(parts);
  }
  g_mutex_lock(&self->lock);
  // Properties are construction-time configuration. Refuse to mutate gates
  // after request pads exist; replacing a gate with live loans would strand
  // terminal completion credits.
  if (!self->slots.empty()) {
    g_mutex_unlock(&self->lock);
    GST_ERROR_OBJECT(self, "stream-inflight-limits cannot change after request pads exist");
    return;
  }
  self->stream_inflight_limits = std::move(parsed);
  g_mutex_unlock(&self->lock);
}

std::string join_stream_inflight_limits(GstLatestByStreamMux* self) {
  if (!self) {
    return {};
  }
  std::string out;
  g_mutex_lock(&self->lock);
  for (std::size_t i = 0; i < self->stream_inflight_limits.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += std::to_string(self->stream_inflight_limits[i]);
  }
  g_mutex_unlock(&self->lock);
  return out;
}

void set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
  auto* self = GST_LATEST_BY_STREAM_MUX(object);
  switch (prop_id) {
  case PROP_STREAM_IDS:
    parse_stream_ids(self, g_value_get_string(value));
    break;
  case PROP_STREAM_INFLIGHT_LIMITS:
    parse_stream_inflight_limits(self, g_value_get_string(value));
    break;
  case PROP_MAX_INFLIGHT_TOTAL: {
    g_mutex_lock(&self->lock);
    if (!self->slots.empty()) {
      g_mutex_unlock(&self->lock);
      GST_ERROR_OBJECT(self, "max-inflight-total cannot change after request pads exist");
      break;
    }
    self->max_inflight_total = g_value_get_int(value);
    self->total_loan_gate =
        std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(self->max_inflight_total);
    g_mutex_unlock(&self->lock);
    break;
  }
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

void get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
  auto* self = GST_LATEST_BY_STREAM_MUX(object);
  switch (prop_id) {
  case PROP_STREAM_IDS: {
    const std::string joined = join_stream_ids(self);
    g_value_set_string(value, joined.c_str());
    break;
  }
  case PROP_STREAM_INFLIGHT_LIMITS: {
    const std::string joined = join_stream_inflight_limits(self);
    g_value_set_string(value, joined.c_str());
    break;
  }
  case PROP_MAX_INFLIGHT_TOTAL:
    g_mutex_lock(&self->lock);
    g_value_set_int(value, self->max_inflight_total);
    g_mutex_unlock(&self->lock);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

void gst_latest_by_stream_mux_class_init(GstLatestByStreamMuxClass* klass) {
  auto* gobject_class = G_OBJECT_CLASS(klass);
  gobject_class->set_property = set_property;
  gobject_class->get_property = get_property;
  gobject_class->finalize = finalize;

  auto* element_class = GST_ELEMENT_CLASS(klass);
  element_class->request_new_pad = request_new_pad;
  element_class->release_pad = release_pad;
  element_class->change_state = change_state;
  gst_element_class_set_static_metadata(
      element_class, "Neat realtime per-stream mux", "Generic",
      "Keeps the latest buffer for each live stream and schedules ready streams fairly", "SiMa.ai");
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_template));

  g_object_class_install_property(
      gobject_class, PROP_STREAM_IDS,
      g_param_spec_string("stream-ids", "Stream IDs",
                          "Comma-separated stream ids, indexed by request pad number", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_STREAM_INFLIGHT_LIMITS,
      g_param_spec_string(
          "stream-inflight-limits", "Per-stream inflight limits",
          "Comma-separated decoder-backed raw-input admission limits (0 disables; omitted "
          "entries default to 0), indexed by request pad number",
          "", static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_MAX_INFLIGHT_TOTAL,
      g_param_spec_int("max-inflight-total", "Total inflight limit",
                       "Mux-wide decoder-backed raw-input admission limit across all streams "
                       "(0 disables)",
                       0, std::numeric_limits<int>::max(), 0,
                       static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

void gst_latest_by_stream_mux_init(GstLatestByStreamMux* self) {
  new (&self->slots) SlotVector();
  new (&self->stream_ids) StringVector();
  new (&self->stream_inflight_limits) IntVector();
  new (&self->total_loan_gate)
      HolderLoanGatePtr(std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(0));
  self->max_inflight_total = 0;
  self->lifetime_guard_enabled = true;
  do {
    self->loan_namespace = mux_namespace_counter().fetch_add(1, std::memory_order_relaxed);
  } while (self->loan_namespace == 0);
  g_mutex_init(&self->lock);
  g_cond_init(&self->cond);
  self->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
  gst_element_add_pad(GST_ELEMENT(self), self->srcpad);
}

struct PublicBufferIdentity {
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
};

PublicBufferIdentity public_buffer_identity(GstBuffer* buffer) {
  PublicBufferIdentity identity;
  std::string current_stream_id;
  std::string original_stream_id;
  std::string buffer_name;
  if (buffer) {
    GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
    GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    if (s) {
      if (const char* stream = gst_structure_get_string(s, "stream-id"); stream && *stream) {
        current_stream_id = stream;
      }
      if (const char* stream = gst_structure_get_string(s, "orig-stream-id"); stream && *stream) {
        original_stream_id = stream;
      }
      if (const char* name = gst_structure_get_string(s, "buffer-name"); name && *name) {
        buffer_name = name;
      }
    }
  }

  // Match the same effective identity that InputStreamPull exposes. Replacing
  // output pools can retain an old orig-stream-id while refreshing stream-id;
  // a framework buffer-name in stream-id is the exception and defers to orig.
  identity.stream_id =
      (current_stream_id.empty() || (!buffer_name.empty() && current_stream_id == buffer_name))
          ? original_stream_id
          : current_stream_id;
  if (identity.stream_id.empty()) {
    identity.stream_id = current_stream_id;
  }
  if (!identity.stream_id.empty()) {
    std::string ignored_stream;
    (void)read_stream_frame_key(buffer, &ignored_stream, &identity.frame_id, &identity.input_seq,
                                &identity.orig_input_seq);
  }
  return identity;
}

} // namespace

namespace simaai::neat {

bool register_latest_by_stream_mux() {
  static std::once_flag once;
  static bool registered = false;
  std::call_once(once, []() {
    registered = gst_element_register(nullptr, kFactoryName, GST_RANK_NONE,
                                      GST_TYPE_LATEST_BY_STREAM_MUX) == TRUE;
  });
  return registered;
}

std::uint64_t latest_by_stream_mux_namespace(GstElement* element) {
  if (!element || !G_TYPE_CHECK_INSTANCE_TYPE(element, GST_TYPE_LATEST_BY_STREAM_MUX)) {
    return 0;
  }
  return GST_LATEST_BY_STREAM_MUX(element)->loan_namespace;
}

namespace pipeline_internal {

bool set_latest_by_stream_mux_lifetime_guard_enabled(GstElement* element, bool enabled) {
  if (!element || !G_TYPE_CHECK_INSTANCE_TYPE(element, GST_TYPE_LATEST_BY_STREAM_MUX)) {
    return false;
  }
  auto* self = GST_LATEST_BY_STREAM_MUX(element);
  g_mutex_lock(&self->lock);
  if (self->worker != nullptr) {
    g_mutex_unlock(&self->lock);
    return false;
  }
  self->lifetime_guard_enabled = enabled;
  g_mutex_unlock(&self->lock);
  return true;
}

bool release_loan_for_key(const LoanKey& key, const char* mode) {
  if (key.namespace_id != 0) {
    return release_loan_for_key_impl(key, mode);
  }
  if (key.stream_id.empty() || (key.frame_id < 0 && key.input_seq < 0 && key.orig_input_seq < 0)) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  bool found = false;
  bool ambiguous = false;
  bool final_release = false;
  const auto seq_matches = [](const LoanKey& candidate, const LoanKey& query) {
    if (query.input_seq >= 0 &&
        (candidate.input_seq == query.input_seq || candidate.orig_input_seq == query.input_seq)) {
      return true;
    }
    if (query.orig_input_seq >= 0 && (candidate.input_seq == query.orig_input_seq ||
                                      candidate.orig_input_seq == query.orig_input_seq)) {
      return true;
    }
    return false;
  };
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto pick = [&](const auto& predicate) {
      for (const auto& item : registry) {
        if (!item.second || item.first.stream_id != key.stream_id || !predicate(item.first)) {
          continue;
        }
        if (found) {
          ambiguous = true;
          break;
        }
        selected = item.first;
        entry = item.second;
        found = true;
      }
    };
    if (key.input_seq >= 0 || key.orig_input_seq >= 0) {
      pick([&](const LoanKey& candidate) { return seq_matches(candidate, key); });
    }
    if (!found && !ambiguous && key.frame_id >= 0) {
      pick([&](const LoanKey& candidate) { return candidate.frame_id == key.frame_id; });
    }
    if (found && !ambiguous) {
      final_release = consume_loan_ref_locked(entry);
    }
  }
  if (!found || ambiguous) {
    if (ambiguous && loan_debug_enabled()) {
      std::fprintf(
          stderr, "[latestmux][loan] ambiguous unqualified release stream=%s frame=%lld mode=%s\n",
          key.stream_id.c_str(), static_cast<long long>(key.frame_id), mode ? mode : "key");
    }
    return false;
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] release ns=%llu stream=%s frame=%lld input_seq=%lld "
                 "orig_input_seq=%lld mode=%s "
                 "(unqualified) final=%d\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id), static_cast<long long>(key.input_seq),
                 static_cast<long long>(key.orig_input_seq), mode ? mode : "key",
                 final_release ? 1 : 0);
  }
  if (final_release) {
    release_loan_entry_resources(entry, /*by_output=*/true);
  }
  return true;
}

bool retain_loan_for_key(const LoanKey& key, std::size_t extra_refs, const char* mode) {
  if (extra_refs == 0U || key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::uint64_t refs = 0;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto it = registry.find(key);
    if (it == registry.end() || !it->second ||
        it->second->released.load(std::memory_order_acquire)) {
      return false;
    }
    it->second->ref_count += static_cast<std::uint64_t>(extra_refs);
    refs = it->second->ref_count;
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] retain ns=%llu stream=%s frame=%lld extra=%zu refs=%llu "
                 "mode=%s\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), extra_refs,
                 static_cast<unsigned long long>(refs), mode ? mode : "retain");
  }
  return true;
}

bool release_oldest_loan_for_stream(const std::string& stream_id, std::int64_t frame_id,
                                    std::int64_t input_seq, std::int64_t orig_input_seq,
                                    std::uint64_t namespace_hint,
                                    GstBuffer* terminal_buffer = nullptr,
                                    bool terminal_replacing_only = false) {
  if (stream_id.empty()) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  bool found = false;
  bool sole_fallback = false;
  bool ambiguous_namespace = false;
  bool final_release = false;
  std::uint64_t namespace_id = 0;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  int best_identity_rank = -1;
  std::shared_ptr<LoanEntry> sole_entry;
  LoanKey sole_key;
  std::size_t distinct_candidates = 0;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    for (const auto& item : registry) {
      const auto& key = item.first;
      if (key.stream_id != stream_id || !item.second ||
          (namespace_hint != 0 && key.namespace_id != namespace_hint)) {
        continue;
      }

      if (!terminal_replacing_only && namespace_hint != 0 && item.second != sole_entry) {
        // A namespace-qualified terminal probe is an exact mux boundary. Keep
        // a count for the legacy sole-loan fallback used by guard-enabled,
        // identity-preserving chains.
        if (!sole_entry) {
          sole_entry = item.second;
          sole_key = key;
          distinct_candidates = 1U;
        } else {
          ++distinct_candidates;
        }
      }

      if (terminal_replacing_only && !item.second->terminal_replacing) {
        continue;
      }

      // Transforms commonly rewrite frame-id but preserve input-seq. Never let
      // a coincidental/stale frame-id beat the stronger sequence identity.
      const bool sequence_matches =
          (input_seq >= 0 && (key.input_seq == input_seq || key.orig_input_seq == input_seq)) ||
          (orig_input_seq >= 0 &&
           (key.input_seq == orig_input_seq || key.orig_input_seq == orig_input_seq));
      const bool has_sequence_identity = input_seq >= 0 || orig_input_seq >= 0;
      const bool frame_matches =
          !has_sequence_identity && frame_id >= 0 && key.frame_id == frame_id;
      // Replacement output pools can recycle frame/input scalars that happen
      // to name a newer live loan. Replacing chains guarantee order within one
      // source stream, not across all source streams, so their authoritative
      // contract is the per-stream FIFO regardless of those scalar values.
      const int identity_rank =
          terminal_replacing_only ? 0 : (sequence_matches ? 2 : (frame_matches ? 1 : -1));
      if (identity_rank < 0) {
        continue;
      }
      if (namespace_id == 0) {
        namespace_id = key.namespace_id;
      } else if (namespace_id != key.namespace_id) {
        ambiguous_namespace = true;
        break;
      }
      if (!found || identity_rank > best_identity_rank ||
          (identity_rank == best_identity_rank && item.second->sequence < best_sequence)) {
        selected = item.first;
        entry = item.second;
        best_sequence = item.second->sequence;
        best_identity_rank = identity_rank;
        found = true;
      }
    }
    if (!terminal_replacing_only && !found && namespace_hint != 0 && distinct_candidates == 1U &&
        sole_entry) {
      selected = sole_key;
      entry = sole_entry;
      found = true;
      sole_fallback = true;
    }
    if (found && !ambiguous_namespace) {
      final_release = consume_loan_ref_locked(entry);
    }
  }
  if (!found || ambiguous_namespace) {
    if (ambiguous_namespace && loan_debug_enabled()) {
      std::fprintf(stderr,
                   "[latestmux][loan] ambiguous stream-fifo release stream=%s across mux "
                   "namespaces\n",
                   stream_id.c_str());
    }
    return false;
  }
  // The registry entry is authoritative after either sequence/frame matching
  // or the namespace-bounded sole-loan fallback.  Terminal buffers can come
  // from recycled output pools, so restore their routing identity and timing
  // before exposing them to the application.  Do not hold the process-wide
  // registry lock while mutating GstMeta; entry is retained by shared_ptr and
  // its identity/timing are immutable after registration.
  restore_loan_identity_and_timing_on_buffer(entry, selected, terminal_buffer);
  if (loan_debug_enabled()) {
    std::fprintf(
        stderr, "[latestmux][loan] release ns=%llu stream=%s frame=%lld mode=%s final=%d\n",
        static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
        static_cast<long long>(selected.frame_id),
        terminal_replacing_only
            ? (best_identity_rank > 0 ? "stream-terminal-identity" : "stream-terminal-fifo")
            : (sole_fallback ? "stream-sole-fallback" : "stream-identity"),
        final_release ? 1 : 0);
  }
  if (final_release) {
    release_loan_entry_resources(entry, /*by_output=*/true);
  }
  return true;
}

bool release_replacing_loan_credit_for_identity(const std::string& stream_id, std::int64_t frame_id,
                                                std::int64_t input_seq, std::int64_t orig_input_seq,
                                                std::uint64_t namespace_hint) {
  if (stream_id.empty() || namespace_hint == 0) {
    return false;
  }

  const bool has_sequence_identity = input_seq >= 0 || orig_input_seq >= 0;
  const bool has_frame_identity = !has_sequence_identity && frame_id >= 0;
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  ClaimedLoanCreditRelease credit_release;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    for (const auto& item : loan_registry()) {
      const LoanKey& key = item.first;
      const auto& candidate = item.second;
      if (key.namespace_id != namespace_hint || key.stream_id != stream_id || !candidate ||
          !candidate->terminal_replacing || candidate->released.load(std::memory_order_acquire) ||
          loan_entry_credit_released(candidate)) {
        continue;
      }
      const bool sequence_matches =
          (input_seq >= 0 && (key.input_seq == input_seq || key.orig_input_seq == input_seq)) ||
          (orig_input_seq >= 0 &&
           (key.input_seq == orig_input_seq || key.orig_input_seq == orig_input_seq));
      const bool frame_matches = frame_id >= 0 && key.frame_id == frame_id;
      if ((has_sequence_identity && !sequence_matches) || (has_frame_identity && !frame_matches)) {
        continue;
      }
      // A replacing fused consumer is ordered and non-dropping. ProcessCVU's
      // output carries the current frame context, so prefer that exact
      // sequence/frame. Older plugin shapes that preserve only stream identity
      // use the ordered per-stream FIFO. Timing remains in the registry for the
      // later terminal FIFO match.
      if (!entry || candidate->sequence < best_sequence) {
        entry = candidate;
        selected = key;
        best_sequence = candidate->sequence;
      }
    }
    // Claim while the registry selection is serialized. ProcessCVU normally
    // pushes one src buffer at a time, but this also makes concurrent probe
    // callbacks advance to distinct outstanding frames instead of all
    // selecting the same oldest entry.
    credit_release = claim_loan_entry_credit_release(entry);
  }
  if (!credit_release.claimed) {
    return false;
  }

  apply_claimed_loan_credit_release(std::move(credit_release));
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] source-consumed ns=%llu stream=%s frame=%lld "
                 "sequence=%llu credit_released=%d\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id),
                 static_cast<unsigned long long>(best_sequence), 1);
  }
  return true;
}

void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id) {
  (void)release_loan_for_key(LoanKey{0, stream_id, frame_id}, "key");
}

bool release_latest_by_stream_mux_raw_input_credit_for_buffer(GstBuffer* buffer,
                                                              std::uint64_t namespace_hint) {
  const PublicBufferIdentity identity = public_buffer_identity(buffer);
  return release_replacing_loan_credit_for_identity(identity.stream_id, identity.frame_id,
                                                    identity.input_seq, identity.orig_input_seq,
                                                    namespace_hint);
}

bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer, std::uint64_t namespace_hint) {
  // A lifecycle carrier is authoritative even if its scalar private/public
  // GstSimaMeta fields are stale. Claim it first: guarded identity-preserving
  // chains do not need the replacing-chain public metadata parse at all, and
  // the one-shot claim remains the serialization point for concurrent terminal
  // calls through the same physical buffer.
  const LoanGuardTerminalClaim guard_claim =
      claim_loan_drop_guard_for_terminal(buffer, namespace_hint);
  if (guard_claim.present) {
    const auto& guard = guard_claim.guard;
    if (!guard_claim.claimed || !guard) {
      return false;
    }
    return release_loan_for_key_impl(guard->key, "lifetime-meta", buffer, guard->sequence);
  }

  const PublicBufferIdentity identity = public_buffer_identity(buffer);
  const std::string& public_stream_id = identity.stream_id;
  const std::int64_t public_frame_id = identity.frame_id;
  const std::int64_t public_input_seq = identity.input_seq;
  const std::int64_t public_orig_input_seq = identity.orig_input_seq;
  const bool have_public_stream = !public_stream_id.empty();

  // Buffer-replacing fused stages intentionally disable lifecycle guards. The
  // public stream identity remains authoritative even when frame/input fields
  // are replaced or recycled. Match within that stream so async cross-stream
  // completion cannot stamp stream A's timing/identity on stream B's result.
  if (namespace_hint != 0 && have_public_stream &&
      release_oldest_loan_for_stream(public_stream_id, public_frame_id, public_input_seq,
                                     public_orig_input_seq, namespace_hint, buffer,
                                     /*terminal_replacing_only=*/true)) {
    return true;
  }

  LoanKey key;
  const bool have_private_key = read_latest_mux_loan_key(buffer, &key);
  if (have_private_key) {
    const bool public_has_sequence = public_input_seq >= 0 || public_orig_input_seq >= 0;
    const bool public_sequence_matches =
        (public_input_seq >= 0 &&
         (key.input_seq == public_input_seq || key.orig_input_seq == public_input_seq)) ||
        (public_orig_input_seq >= 0 &&
         (key.input_seq == public_orig_input_seq || key.orig_input_seq == public_orig_input_seq));
    const bool public_identity_matches =
        !have_public_stream ||
        (key.stream_id == public_stream_id &&
         (public_has_sequence ? public_sequence_matches
                              : (public_frame_id < 0 || key.frame_id == public_frame_id)));
    if (public_identity_matches && (namespace_hint == 0 || key.namespace_id == namespace_hint) &&
        release_loan_for_key_impl(key, "meta", buffer)) {
      return true;
    }

    // Some hardware transforms recycle output buffers without removing the
    // private mux fields previously stamped on that GstBuffer. The key can
    // therefore be syntactically valid but name an already-completed frame.
    // At the terminal probe we also have the immutable namespace of the exact
    // mux feeding this pipeline. Reuse the same bounded fallback as the
    // private-key-absent path. Use the current public stream fields rather
    // than the recycled private stream key; without a matching public identity
    // it releases only when this namespace and stream have exactly one
    // distinct outstanding loan.
    if (namespace_hint == 0) {
      return false;
    }
    if (!have_public_stream) {
      return false;
    }
    return release_oldest_loan_for_stream(public_stream_id, public_frame_id, public_input_seq,
                                          public_orig_input_seq, namespace_hint, buffer);
  }

  if (have_public_stream) {
    // Output transforms are allowed to replace the original frame identity.
    // Parse it when present so the normal exact path remains preferred, but a
    // namespace-qualified terminal probe can safely fall back to the sole
    // outstanding loan for this stream when it is missing or no longer
    // matches.
    if (loan_meta_debug_enabled()) {
      static std::atomic<int> fallback_logs{0};
      const int seen = fallback_logs.fetch_add(1, std::memory_order_relaxed);
      if (seen < 4) {
        GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
        GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
        gchar* raw = s ? gst_structure_to_string(s) : nullptr;
        std::fprintf(stderr,
                     "[latestmux][loan] terminal private key absent; using stream-fifo "
                     "stream=%s raw=%s\n",
                     public_stream_id.c_str(), raw ? raw : "<no GstSimaMeta>");
        if (raw) {
          g_free(raw);
        }
      }
    }
    return release_oldest_loan_for_stream(public_stream_id, public_frame_id, public_input_seq,
                                          public_orig_input_seq, namespace_hint, buffer);
  }

  if (loan_meta_debug_enabled()) {
    static std::atomic<int> missing_logs{0};
    const int seen = missing_logs.fetch_add(1, std::memory_order_relaxed);
    if (seen < 8) {
      GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
      GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
      gchar* raw = s ? gst_structure_to_string(s) : nullptr;
      std::fprintf(stderr, "[latestmux][loan] terminal missing private loan key raw=%s\n",
                   raw ? raw : "<no GstSimaMeta>");
      if (raw) {
        g_free(raw);
      }
    }
  }
  return false;
}

std::vector<RealtimeFrameCredit> realtime_frame_credits_for_sample(const Sample& sample) {
  std::vector<RealtimeFrameCredit> credits;
  auto add_credit = [&credits](const RealtimeFrameCredit& credit) {
    if ((credit.frame_id < 0 || credit.stream_id.empty()) && credit.input_seq < 0 &&
        credit.orig_input_seq < 0) {
      return;
    }
    const auto found =
        std::find_if(credits.begin(), credits.end(), [&](const RealtimeFrameCredit& existing) {
          return existing.namespace_id == credit.namespace_id &&
                 existing.frame_id == credit.frame_id && existing.stream_id == credit.stream_id &&
                 existing.input_seq == credit.input_seq &&
                 existing.orig_input_seq == credit.orig_input_seq &&
                 existing.graph_private == credit.graph_private;
        });
    if (found == credits.end()) {
      credits.push_back(credit);
    }
  };
  for (const auto& credit : attached_realtime_frame_credits_from_sample(sample)) {
    add_credit(credit);
  }
  auto add_tensor_key = [&](const Tensor& tensor) {
    // Latest-by-stream mux credits are registered under the mux instance namespace stamped on
    // GstSimaMeta. Prefer that exact key so drops/stops can release credits unambiguously when
    // multiple muxes in the process reuse the same stream/frame ids.
    if (!tensor.storage || tensor.storage->kind != StorageKind::GstSample) {
      return false;
    }
    const std::shared_ptr<void> holder = holder_from_tensor(tensor);
    GstBuffer* buffer = holder ? buffer_from_tensor_holder(holder) : nullptr;
    if (!buffer) {
      return false;
    }
    LoanKey key;
    const bool found = read_latest_mux_loan_key(buffer, &key);
    gst_buffer_unref(buffer);
    if (found) {
      add_credit(RealtimeFrameCredit{key.namespace_id, key.stream_id, key.frame_id, key.input_seq,
                                     key.orig_input_seq});
    }
    return found;
  };
  auto collect_stamped_keys = [&](auto&& self, const Sample& s) -> bool {
    bool found_stamped_key = false;
    if (s.tensor) {
      found_stamped_key = add_tensor_key(*s.tensor) || found_stamped_key;
    }
    for (const auto& tensor : s.tensors) {
      found_stamped_key = add_tensor_key(tensor) || found_stamped_key;
    }
    for (const auto& field : s.fields) {
      found_stamped_key = self(self, field) || found_stamped_key;
    }
    return found_stamped_key;
  };
  const bool has_stamped_latest_mux_key = collect_stamped_keys(collect_stamped_keys, sample);
  auto add_fallback_key = [&](const Sample& s) {
    add_credit(RealtimeFrameCredit{0, s.stream_id, s.frame_id, s.input_seq, s.orig_input_seq});
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    add_fallback_key(s);
    for (const auto& field : s.fields) {
      self(self, field);
    }
  };
  if (!has_stamped_latest_mux_key) {
    walk(walk, sample);
  }
  return credits;
}

bool retain_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                   std::size_t extra_refs, const char* mode) {
  bool retained_any = retain_registered_realtime_frame_credits(credits, extra_refs, mode);
  if (credits.empty() || extra_refs == 0U) {
    return retained_any;
  }
  for (const auto& credit : credits) {
    if (credit.graph_private || credit.namespace_id == 0) {
      continue;
    }
    retained_any =
        retain_loan_for_key(LoanKey{credit.namespace_id, credit.stream_id, credit.frame_id,
                                    credit.input_seq, credit.orig_input_seq},
                            extra_refs, mode) ||
        retained_any;
  }
  return retained_any;
}

void release_realtime_frame_credits_impl(const std::vector<RealtimeFrameCredit>& credits,
                                         const char* mode, bool by_output) {
  for (const auto& credit : credits) {
    if (credit.graph_private) {
      (void)release_registered_realtime_frame_credit(credit, mode, by_output);
      continue;
    }
    if (credit.namespace_id != 0) {
      /*
       * Non-zero keys collected from GstSimaMeta are latest-by-stream mux loan
       * keys, not C++ realtime-credit registry namespaces. Keep those domains
       * separate: the graph credit, when present, is carried as an exact
       * graph-private sidecar credit and was handled above.
       */
      (void)release_loan_for_key(LoanKey{credit.namespace_id, credit.stream_id, credit.frame_id,
                                         credit.input_seq, credit.orig_input_seq},
                                 mode);
      continue;
    }
    if (release_registered_realtime_frame_credit(credit, mode, by_output)) {
      continue;
    }
    if (!credit.stream_id.empty() && credit.frame_id >= 0) {
      (void)release_loan_for_key(LoanKey{credit.namespace_id, credit.stream_id, credit.frame_id,
                                         credit.input_seq, credit.orig_input_seq},
                                 mode);
    } else if (!credit.stream_id.empty() && (credit.input_seq >= 0 || credit.orig_input_seq >= 0)) {
      (void)release_loan_for_key(LoanKey{credit.namespace_id, credit.stream_id, credit.frame_id,
                                         credit.input_seq, credit.orig_input_seq},
                                 mode);
    }
  }
}

void release_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                    const char* mode) {
  release_realtime_frame_credits_impl(credits, mode, /*by_output=*/true);
}

void release_realtime_frame_credits_without_output(const std::vector<RealtimeFrameCredit>& credits,
                                                   const char* mode) {
  release_realtime_frame_credits_impl(credits, mode, /*by_output=*/false);
}

void release_realtime_frame_credits_for_sample(const Sample& sample, const char* mode) {
  release_realtime_frame_credits(realtime_frame_credits_for_sample(sample), mode);
}

} // namespace pipeline_internal

} // namespace simaai::neat

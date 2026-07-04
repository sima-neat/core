#include "gst/GstLatestByStreamMux.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"

#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
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
#define GST_TYPE_LATEST_BY_STREAM_MUX (gst_latest_by_stream_mux_get_type())
#define GST_LATEST_BY_STREAM_MUX(obj)                                                              \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_LATEST_BY_STREAM_MUX, GstLatestByStreamMux))

int configured_max_inflight_per_stream() {
  static const int value = []() {
    const gchar* env = g_getenv("SIMA_LATEST_MUX_MAX_INFLIGHT_PER_STREAM");
    if (!env || !*env) {
      return 1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env) {
      return 1;
    }
    if (parsed <= 0) {
      return 0;
    }
    if (parsed > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(parsed);
  }();
  return value;
}

bool loan_debug_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_LOAN_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

bool loan_meta_debug_enabled() {
  const gchar* env = g_getenv("SIMA_LATEST_MUX_LOAN_META_DEBUG");
  return env && *env && g_strcmp0(env, "0") != 0 && g_ascii_strcasecmp(env, "false") != 0;
}

struct StreamLoanState {
  explicit StreamLoanState(int credit_limit)
      : gate(std::make_shared<simaai::neat::pipeline_internal::HolderLoanGate>(credit_limit)) {}

  simaai::neat::pipeline_internal::HolderLoanGatePtr gate;
  std::atomic<std::uint64_t> registered{0};
  std::atomic<std::uint64_t> released_by_output{0};
  std::atomic<std::uint64_t> released_without_output{0};
  std::atomic<std::uint64_t> missing_key{0};
};

std::shared_ptr<StreamLoanState> make_stream_loan_state() {
  return std::make_shared<StreamLoanState>(configured_max_inflight_per_stream());
}

struct PendingSlot {
  GstPad* pad = nullptr;
  guint index = 0;
  GstBuffer* pending = nullptr; // owned by the slot while queued
  GstCaps* caps = nullptr;      // latest caps seen on this stream
  bool eos = false;
  bool have_segment = false;
  guint64 next_frame_id = 0;
  guint64 emitted = 0;
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> replaced{0};
  std::atomic<std::uint64_t> no_credit_skips{0};
  std::shared_ptr<StreamLoanState> loan_state = make_stream_loan_state();
};

using SlotVector = std::vector<PendingSlot*>;
using StringVector = std::vector<std::string>;

struct _GstLatestByStreamMux {
  GstElement parent;
  GstPad* srcpad = nullptr;

  GMutex lock;
  GCond cond;
  SlotVector slots;
  StringVector stream_ids;
  guint next_pad_index = 0;
  guint rr_index = 0;
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

  bool operator==(const LoanKey& other) const {
    return namespace_id == other.namespace_id && frame_id == other.frame_id &&
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
    return h;
  }
};

struct LoanEntry {
  GstLatestByStreamMux* mux = nullptr; // strong GObject ref while registered
  std::shared_ptr<StreamLoanState> state;
  std::uint64_t sequence = 0;
  std::atomic<bool> released{false};
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
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  bool added_meta = false;
  if (!meta) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    added_meta = true;
  }
  if (!meta) {
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    return false;
  }

  bool structure_mutable = added_meta;
#if defined(GST_STRUCTURE_IS_MUTABLE)
  structure_mutable = GST_STRUCTURE_IS_MUTABLE(s);
#elif defined(GST_STRUCTURE_IS_WRITABLE)
  structure_mutable = GST_STRUCTURE_IS_WRITABLE(s);
#endif
  if (!structure_mutable) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    GstStructure* snapshot = gst_structure_copy(s);
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    if (!s) {
      if (snapshot) {
        gst_structure_free(snapshot);
      }
      return false;
    }
    if (snapshot) {
      const gint n_fields = gst_structure_n_fields(snapshot);
      for (gint i = 0; i < n_fields; ++i) {
        const char* fname = gst_structure_nth_field_name(snapshot, i);
        if (!fname) {
          continue;
        }
        const GValue* val = gst_structure_get_value(snapshot, fname);
        if (val) {
          gst_structure_set_value(s, fname, val);
        }
      }
      gst_structure_free(snapshot);
    }
  }

  *structure_out = s;
  return true;
}

bool stamp_latest_mux_loan_key(GstBuffer* buffer, std::uint64_t namespace_id,
                               const std::string& stream_id, std::int64_t frame_id) {
  if (!buffer || namespace_id == 0 || stream_id.empty() || frame_id < 0) {
    return false;
  }
  GstStructure* s = nullptr;
  if (!ensure_sima_meta_structure_mutable(buffer, &s)) {
    return false;
  }
  gst_structure_set(s, kLoanValidField, G_TYPE_BOOLEAN, TRUE, kLoanNamespaceField, G_TYPE_UINT64,
                    static_cast<guint64>(namespace_id), kLoanStreamIdField, G_TYPE_STRING,
                    stream_id.c_str(), kLoanFrameIdField, G_TYPE_INT64,
                    static_cast<gint64>(frame_id), nullptr);
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
  return true;
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
  if (entry->state && entry->state->gate) {
    entry->state->gate->release();
    if (by_output) {
      entry->state->released_by_output.fetch_add(1, std::memory_order_relaxed);
    } else {
      entry->state->released_without_output.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (entry->mux) {
    g_mutex_lock(&entry->mux->lock);
    g_cond_broadcast(&entry->mux->cond);
    g_mutex_unlock(&entry->mux->lock);
    gst_object_unref(GST_OBJECT(entry->mux));
    entry->mux = nullptr;
  }
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
      if (it->second && it->second->mux == self) {
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

bool release_loan_for_key_impl(const LoanKey& key, const char* mode) {
  if (key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto it = registry.find(key);
    if (it == registry.end()) {
      return false;
    }
    entry = it->second;
    registry.erase(it);
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr, "[latestmux][loan] release ns=%llu stream=%s frame=%lld mode=%s\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), mode ? mode : "key");
  }
  release_loan_entry(entry, /*by_output=*/true);
  return true;
}

GType gst_latest_by_stream_mux_get_type();

G_DEFINE_TYPE_WITH_CODE(GstLatestByStreamMux, gst_latest_by_stream_mux, GST_TYPE_ELEMENT,
                        GST_DEBUG_CATEGORY_INIT(gst_latest_by_stream_mux_debug_category,
                                                "neatlatestbystreammux", 0,
                                                "Neat live latest-by-stream mux"));

enum {
  PROP_0,
  PROP_STREAM_IDS,
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

bool read_stream_frame_key(GstBuffer* buffer, std::string* stream_id, std::int64_t* frame_id) {
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
  gint64 frame = -1;
  gboolean sample_frame_valid = FALSE;
  if (gst_structure_get_boolean(s, "sample-frame-id-valid", &sample_frame_valid) == TRUE &&
      sample_frame_valid == TRUE) {
    (void)gst_structure_get_int64(s, "sample-frame-id", &frame);
  }
  if (frame < 0) {
    (void)gst_structure_get_int64(s, "frame-id", &frame);
  }
  if (!stream || !*stream || frame < 0) {
    return false;
  }
  *stream_id = stream;
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

void register_loan_for_key(GstLatestByStreamMux* self,
                           const std::shared_ptr<StreamLoanState>& state,
                           const std::string& stream_id, std::int64_t frame_id) {
  if (!self || !state || !state->gate || !state->gate->enabled() || stream_id.empty() ||
      frame_id < 0) {
    return;
  }

  auto entry = std::make_shared<LoanEntry>();
  entry->mux = GST_LATEST_BY_STREAM_MUX(gst_object_ref(GST_OBJECT(self)));
  entry->state = state;
  entry->sequence = loan_sequence_counter().fetch_add(1, std::memory_order_relaxed);

  std::shared_ptr<LoanEntry> replaced;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    LoanKey key{self->loan_namespace, stream_id, frame_id};
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
                 "[latestmux][loan] register ns=%llu stream=%s frame=%lld inflight=%d "
                 "limit=%d\n",
                 static_cast<unsigned long long>(self->loan_namespace), stream_id.c_str(),
                 static_cast<long long>(frame_id), state->gate->inflight(),
                 state->gate->credit_limit());
  }
}

void release_acquired_loan_without_output(const std::shared_ptr<StreamLoanState>& state) {
  if (!state || !state->gate || !state->gate->enabled()) {
    return;
  }
  state->gate->release();
  state->released_without_output.fetch_add(1, std::memory_order_relaxed);
}

void clear_pending(PendingSlot* slot) {
  if (!slot || !slot->pending) {
    return;
  }
  gst_buffer_unref(slot->pending);
  slot->pending = nullptr;
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

bool all_slots_eos_locked(const GstLatestByStreamMux* self) {
  if (!self || self->slots.empty()) {
    return false;
  }
  for (const PendingSlot* slot : self->slots) {
    if (slot && !slot->eos) {
      return false;
    }
  }
  return true;
}

bool slot_has_loan_credit(const PendingSlot* slot) {
  if (!slot || !slot->loan_state || !slot->loan_state->gate || !slot->loan_state->gate->enabled()) {
    return true;
  }
  return slot->loan_state->gate->inflight() < slot->loan_state->gate->credit_limit();
}

PendingSlot* take_next_slot_locked(GstLatestByStreamMux* self, bool* loan_acquired) {
  if (loan_acquired) {
    *loan_acquired = false;
  }
  if (!self || self->slots.empty()) {
    return nullptr;
  }
  const guint n = static_cast<guint>(self->slots.size());
  PendingSlot* best = nullptr;
  guint best_index = 0;
  guint64 best_emitted = 0;
  for (guint offset = 0; offset < n; ++offset) {
    const guint idx = (self->rr_index + offset) % n;
    PendingSlot* slot = self->slots[idx];
    if (!slot || !slot->pending) {
      continue;
    }
    if (!slot_has_loan_credit(slot)) {
      slot->no_credit_skips.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (!best || slot->emitted < best_emitted) {
      best = slot;
      best_index = idx;
      best_emitted = slot->emitted;
    }
  }
  if (best) {
    if (best->loan_state && best->loan_state->gate && best->loan_state->gate->enabled()) {
      if (!best->loan_state->gate->try_acquire()) {
        return nullptr;
      }
      if (loan_acquired) {
        *loan_acquired = true;
      }
    }
    self->rr_index = (best_index + 1U) % n;
    ++best->emitted;
  }
  return best;
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
    std::fprintf(stderr, "[latestmux][stats] worker-start every=%llu max_inflight=%d\n",
                 static_cast<unsigned long long>(stats_interval()),
                 configured_max_inflight_per_stream());
    std::fflush(stderr);
  }
  while (true) {
    GstBuffer* buffer = nullptr;
    GstCaps* caps = nullptr;
    std::shared_ptr<StreamLoanState> loan_state;
    bool loan_acquired = false;
    bool push_eos = false;

    g_mutex_lock(&self->lock);
    while (!self->stopping && !self->flushing) {
      PendingSlot* slot = take_next_slot_locked(self, &loan_acquired);
      if (slot && slot->pending) {
        buffer = slot->pending;
        slot->pending = nullptr;
        loan_state = slot->loan_state;
        if (slot->caps) {
          caps = gst_caps_ref(slot->caps);
        }
        break;
      }
      if (all_slots_eos_locked(self)) {
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
      bool loan_registered = false;
      if (loan_acquired) {
        if (read_stream_frame_key(buffer, &stream_id, &frame_id)) {
          if (stamp_latest_mux_loan_key(buffer, self->loan_namespace, stream_id, frame_id)) {
            register_loan_for_key(self, loan_state, stream_id, frame_id);
            loan_registered = true;
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
        (void)release_loan_for_key_impl(LoanKey{self->loan_namespace, stream_id, frame_id},
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

  stamp_stream_meta(self, &buffer, slot);

  g_mutex_lock(&self->lock);
  if (self->flushing || self->stopping) {
    g_mutex_unlock(&self->lock);
    gst_buffer_unref(buffer);
    return GST_FLOW_FLUSHING;
  }
  const std::uint64_t received_before = slot->received.fetch_add(1, std::memory_order_relaxed);
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
  clear_pending(slot);
  slot->pending = buffer; // take ownership from chain
  g_cond_signal(&self->cond);
  g_mutex_unlock(&self->lock);
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
  case GST_EVENT_FLUSH_START:
    g_mutex_lock(&self->lock);
    self->flushing = true;
    for (PendingSlot* s : self->slots) {
      clear_pending(s);
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
    release_all_loans_for_mux(self);
    return gst_pad_push_event(self->srcpad, event) == TRUE;
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

  g_mutex_lock(&self->lock);
  const guint index = have_requested_index ? requested_index : self->next_pad_index;
  self->next_pad_index = std::max(self->next_pad_index, index + 1U);
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

  g_mutex_lock(&self->lock);
  auto it = std::find(self->slots.begin(), self->slots.end(), slot);
  if (it != self->slots.end()) {
    self->slots.erase(it);
  }
  clear_pending(slot);
  clear_slot_caps(slot);
  g_cond_broadcast(&self->cond);
  g_mutex_unlock(&self->lock);

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
  g_mutex_lock(&self->lock);
  if (once && self->stats_reported) {
    g_mutex_unlock(&self->lock);
    return;
  }
  if (once) {
    self->stats_reported = true;
  }
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

  std::fprintf(stderr, "[latestmux][stats] reason=%s slots=%zu\n", reason ? reason : "unknown",
               rows.size());
  for (const auto& row : rows) {
    std::fprintf(
        stderr,
        "[latestmux][stats] slot=%u stream=%s chain=%llu replaced=%llu emitted=%llu "
        "pending=%d eos=%d no_credit_skips=%llu loans_registered=%llu "
        "loans_released_output=%llu loans_released_without_output=%llu missing_key=%llu "
        "loan_inflight=%d loan_limit=%d\n",
        row.index, row.stream_id.c_str(), static_cast<unsigned long long>(row.received),
        static_cast<unsigned long long>(row.replaced), static_cast<unsigned long long>(row.emitted),
        row.pending ? 1 : 0, row.eos ? 1 : 0, static_cast<unsigned long long>(row.no_credit_skips),
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
  case GST_STATE_CHANGE_READY_TO_PAUSED:
    g_mutex_lock(&self->lock);
    self->stopping = false;
    self->flushing = false;
    self->stream_start_sent = false;
    self->segment_sent = false;
    self->have_pending_segment = false;
    self->stats_reported = false;
    self->stats_pushes = 0;
    gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
    for (PendingSlot* slot : self->slots) {
      if (slot) {
        slot->eos = false;
        slot->emitted = 0;
        reset_slot_stats(slot);
        clear_pending(slot);
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
    break;
  default:
    break;
  }

  GstStateChangeReturn ret =
      GST_ELEMENT_CLASS(gst_latest_by_stream_mux_parent_class)->change_state(element, transition);

  switch (transition) {
  case GST_STATE_CHANGE_PAUSED_TO_READY:
    g_mutex_lock(&self->lock);
    self->stopping = true;
    for (PendingSlot* slot : self->slots) {
      clear_pending(slot);
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
    release_all_loans_for_mux(self);
    if (self->worker) {
      g_thread_join(self->worker);
      self->worker = nullptr;
    }
    print_slot_stats_once(self, "paused-to-ready");
    break;
  default:
    break;
  }
  return ret;
}

void finalize(GObject* object) {
  auto* self = GST_LATEST_BY_STREAM_MUX(object);
  g_mutex_lock(&self->lock);
  self->stopping = true;
  for (PendingSlot* slot : self->slots) {
    clear_pending(slot);
    clear_slot_caps(slot);
  }
  g_cond_broadcast(&self->cond);
  g_mutex_unlock(&self->lock);
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
    clear_pending(slot);
    clear_slot_caps(slot);
    delete slot;
  }
  self->slots.clear();
  self->stream_ids.clear();
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

void set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
  auto* self = GST_LATEST_BY_STREAM_MUX(object);
  switch (prop_id) {
  case PROP_STREAM_IDS:
    parse_stream_ids(self, g_value_get_string(value));
    break;
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
  gst_element_class_set_static_metadata(element_class, "Neat latest-by-stream mux", "Generic",
                                        "Keeps one latest buffer per live stream and pushes fairly",
                                        "SiMa.ai");
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_template));

  g_object_class_install_property(
      gobject_class, PROP_STREAM_IDS,
      g_param_spec_string("stream-ids", "Stream IDs",
                          "Comma-separated stream ids, indexed by request pad number", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

void gst_latest_by_stream_mux_init(GstLatestByStreamMux* self) {
  new (&self->slots) SlotVector();
  new (&self->stream_ids) StringVector();
  do {
    self->loan_namespace = mux_namespace_counter().fetch_add(1, std::memory_order_relaxed);
  } while (self->loan_namespace == 0);
  g_mutex_init(&self->lock);
  g_cond_init(&self->cond);
  self->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_segment_init(&self->pending_segment, GST_FORMAT_TIME);
  gst_element_add_pad(GST_ELEMENT(self), self->srcpad);
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

namespace pipeline_internal {

bool release_loan_for_key(const LoanKey& key, const char* mode) {
  if (key.namespace_id != 0) {
    return release_loan_for_key_impl(key, mode);
  }
  if (key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  bool found = false;
  bool ambiguous = false;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    for (const auto& item : registry) {
      if (item.first.stream_id != key.stream_id || item.first.frame_id != key.frame_id ||
          !item.second) {
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
    if (found && !ambiguous) {
      registry.erase(selected);
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
                 "[latestmux][loan] release ns=%llu stream=%s frame=%lld mode=%s "
                 "(unqualified)\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id), mode ? mode : "key");
  }
  release_loan_entry(entry, /*by_output=*/true);
  return true;
}

bool release_oldest_loan_for_stream(const std::string& stream_id) {
  if (stream_id.empty()) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  bool found = false;
  bool ambiguous_namespace = false;
  std::uint64_t namespace_id = 0;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    for (const auto& item : registry) {
      if (item.first.stream_id != stream_id || !item.second) {
        continue;
      }
      if (namespace_id == 0) {
        namespace_id = item.first.namespace_id;
      } else if (namespace_id != item.first.namespace_id) {
        ambiguous_namespace = true;
        break;
      }
      if (!found || item.second->sequence < best_sequence) {
        selected = item.first;
        entry = item.second;
        best_sequence = item.second->sequence;
        found = true;
      }
    }
    if (found && !ambiguous_namespace) {
      registry.erase(selected);
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
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] release ns=%llu stream=%s frame=%lld mode=stream-fifo\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id));
  }
  release_loan_entry(entry, /*by_output=*/true);
  return true;
}

void release_latest_by_stream_mux_loan(const std::string& stream_id, std::int64_t frame_id) {
  (void)release_loan_for_key(LoanKey{0, stream_id, frame_id}, "key");
}

bool release_latest_by_stream_mux_loan_for_buffer(GstBuffer* buffer) {
  if (configured_max_inflight_per_stream() <= 0) {
    return false;
  }
  LoanKey key;
  if (read_latest_mux_loan_key(buffer, &key)) {
    return release_loan_for_key(key, "meta");
  }

  std::string stream_id;
  if (read_stream_key(buffer, &stream_id)) {
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
                     stream_id.c_str(), raw ? raw : "<no GstSimaMeta>");
        if (raw) {
          g_free(raw);
        }
      }
    }
    return release_oldest_loan_for_stream(stream_id);
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
  auto add_key = [&credits](const Sample& s) {
    if (s.frame_id < 0 || s.stream_id.empty()) {
      return;
    }
    const RealtimeFrameCredit credit{0, s.stream_id, s.frame_id};
    const auto found =
        std::find_if(credits.begin(), credits.end(), [&](const RealtimeFrameCredit& existing) {
          return existing.namespace_id == credit.namespace_id &&
                 existing.frame_id == credit.frame_id && existing.stream_id == credit.stream_id;
        });
    if (found == credits.end()) {
      credits.push_back(credit);
    }
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    add_key(s);
    for (const auto& field : s.fields) {
      self(self, field);
    }
  };
  walk(walk, sample);
  return credits;
}

void release_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                    const char* mode) {
  for (const auto& credit : credits) {
    (void)release_loan_for_key(
        LoanKey{credit.namespace_id, credit.stream_id, credit.frame_id}, mode);
  }
}

void release_realtime_frame_credits_for_sample(const Sample& sample, const char* mode) {
  release_realtime_frame_credits(realtime_frame_credits_for_sample(sample), mode);
}

} // namespace pipeline_internal

} // namespace simaai::neat

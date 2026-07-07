#include "gst/GstLatestByStreamMux.h"

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"

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

struct LoanEntry {
  GstLatestByStreamMux* mux = nullptr; // strong GObject ref while registered
  std::shared_ptr<StreamLoanState> state;
  std::uint64_t sequence = 0;
  std::uint64_t ref_count = 1;
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

void release_loan_entry_resources(const std::shared_ptr<LoanEntry>& entry, bool by_output) {
  if (!entry) {
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
  bool final_release = false;
  {
    std::lock_guard<std::mutex> lock(loan_registry_mutex());
    auto& registry = loan_registry();
    auto it = registry.find(key);
    if (it == registry.end()) {
      return false;
    }
    entry = it->second;
    final_release = consume_loan_ref_locked(entry);
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

bool force_release_loan_for_key_impl(const LoanKey& key, const char* mode) {
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
    if (entry) {
      entry->ref_count = 0;
    }
    erase_loan_keys_for_entry_locked(entry);
  }
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] force-release ns=%llu stream=%s frame=%lld input_seq=%lld "
                 "orig_input_seq=%lld mode=%s\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), static_cast<long long>(key.input_seq),
                 static_cast<long long>(key.orig_input_seq), mode ? mode : "key");
  }
  release_loan_entry(entry, /*by_output=*/true);
  return true;
}

GQuark latest_mux_buffer_loan_release_quark() {
  static GQuark q = g_quark_from_static_string("sima-latest-mux-buffer-loan-release-v1");
  return q;
}

void release_loan_qdata(gpointer data) {
  std::unique_ptr<LoanKey> key(static_cast<LoanKey*>(data));
  if (key) {
    (void)force_release_loan_for_key_impl(*key, "buffer-finalize");
  }
}

bool attach_buffer_loan_release(GstBuffer* buffer, const LoanKey& key) {
  if (!buffer || key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  auto* owned_key = new (std::nothrow) LoanKey(key);
  if (!owned_key) {
    return false;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buffer), latest_mux_buffer_loan_release_quark(),
                            owned_key, release_loan_qdata);
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
                           const std::shared_ptr<StreamLoanState>& state, const LoanKey& key) {
  if (!self || !state || !state->gate || !state->gate->enabled() || key.namespace_id == 0 ||
      key.stream_id.empty() || key.frame_id < 0) {
    return;
  }

  auto entry = std::make_shared<LoanEntry>();
  entry->mux = GST_LATEST_BY_STREAM_MUX(gst_object_ref(GST_OBJECT(self)));
  entry->state = state;
  entry->sequence = loan_sequence_counter().fetch_add(1, std::memory_order_relaxed);

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
                 static_cast<long long>(key.orig_input_seq), state->gate->inflight(),
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
    while (!self->stopping) {
      if (self->flushing) {
        g_cond_wait(&self->cond, &self->lock);
        continue;
      }
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
      std::int64_t input_seq = -1;
      std::int64_t orig_input_seq = -1;
      bool loan_registered = false;
      if (loan_acquired) {
        if (read_stream_frame_key(buffer, &stream_id, &frame_id, &input_seq, &orig_input_seq)) {
          const LoanKey loan_key{self->loan_namespace, stream_id, frame_id, input_seq,
                                 orig_input_seq};
          if (stamp_latest_mux_loan_key(buffer, loan_key) &&
              attach_buffer_loan_release(buffer, loan_key)) {
            register_loan_for_key(self, loan_state, loan_key);
            loan_registered = true;
          } else {
            release_acquired_loan_without_output(loan_state);
            if (loan_debug_enabled()) {
              std::fprintf(stderr,
                           "[latestmux][loan] failed to arm loan release stream=%s frame=%lld; "
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

bool release_oldest_loan_for_stream(const std::string& stream_id) {
  if (stream_id.empty()) {
    return false;
  }
  std::shared_ptr<LoanEntry> entry;
  LoanKey selected;
  bool found = false;
  bool ambiguous_namespace = false;
  bool final_release = false;
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
  if (loan_debug_enabled()) {
    std::fprintf(stderr,
                 "[latestmux][loan] release ns=%llu stream=%s frame=%lld mode=stream-fifo "
                 "final=%d\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id), final_release ? 1 : 0);
  }
  if (final_release) {
    release_loan_entry_resources(entry, /*by_output=*/true);
  }
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

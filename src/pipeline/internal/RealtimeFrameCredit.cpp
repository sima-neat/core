#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/RealtimeFrameCredit.h"

#include "pipeline/internal/EnvUtil.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

struct CreditKey {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;

  bool operator==(const CreditKey& other) const {
    return namespace_id == other.namespace_id && frame_id == other.frame_id &&
           stream_id == other.stream_id;
  }
};

enum class CreditSeqKind { InputSeq, OrigInputSeq };

struct CreditAliasKey {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t seq = -1;
  CreditSeqKind kind = CreditSeqKind::InputSeq;

  bool operator==(const CreditAliasKey& other) const {
    return namespace_id == other.namespace_id && seq == other.seq && kind == other.kind &&
           stream_id == other.stream_id;
  }
};

struct CreditKeyHash {
  std::size_t operator()(const CreditKey& key) const noexcept {
    std::size_t h = std::hash<std::uint64_t>{}(key.namespace_id);
    const auto mix = [&h](std::size_t value) {
      h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    };
    mix(std::hash<std::string>{}(key.stream_id));
    mix(std::hash<std::int64_t>{}(key.frame_id));
    return h;
  }
};

struct CreditAliasKeyHash {
  std::size_t operator()(const CreditAliasKey& key) const noexcept {
    std::size_t h = std::hash<std::uint64_t>{}(key.namespace_id);
    const auto mix = [&h](std::size_t value) {
      h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    };
    mix(std::hash<std::string>{}(key.stream_id));
    mix(std::hash<std::int64_t>{}(key.seq));
    mix(std::hash<int>{}(static_cast<int>(key.kind)));
    return h;
  }
};

struct CreditEntry {
  RealtimeFrameCreditLanePtr lane;
  std::vector<RealtimeFrameCreditLanePtr> companion_lanes;
  std::uint64_t sequence = 0;
  std::atomic<bool> released{false};
  std::vector<CreditAliasKey> aliases;
};

std::mutex& credit_registry_mutex() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<CreditKey, std::shared_ptr<CreditEntry>, CreditKeyHash>& credit_registry() {
  static std::unordered_map<CreditKey, std::shared_ptr<CreditEntry>, CreditKeyHash> registry;
  return registry;
}

std::unordered_map<CreditAliasKey, std::shared_ptr<CreditEntry>, CreditAliasKeyHash>&
credit_alias_registry() {
  static std::unordered_map<CreditAliasKey, std::shared_ptr<CreditEntry>, CreditAliasKeyHash>
      registry;
  return registry;
}

std::atomic<std::uint64_t>& credit_sequence_counter() {
  static std::atomic<std::uint64_t> counter{1};
  return counter;
}

std::atomic<std::uint64_t>& credit_namespace_counter() {
  static std::atomic<std::uint64_t> counter{1};
  return counter;
}

bool credit_debug_enabled() {
  return env_bool("SIMA_GRAPH_REALTIME_CREDIT_DEBUG", false);
}

std::atomic<int>& credit_miss_debug_logs() {
  static std::atomic<int> count{0};
  return count;
}

const char* credit_seq_kind_name(CreditSeqKind kind) {
  switch (kind) {
  case CreditSeqKind::InputSeq:
    return "input_seq";
  case CreditSeqKind::OrigInputSeq:
    return "orig_input_seq";
  }
  return "seq";
}

void erase_aliases_for_entry_locked(const std::shared_ptr<CreditEntry>& entry) {
  if (!entry) {
    return;
  }
  auto& aliases = credit_alias_registry();
  for (const auto& alias : entry->aliases) {
    auto it = aliases.find(alias);
    if (it != aliases.end() && it->second == entry) {
      aliases.erase(it);
    }
  }
  entry->aliases.clear();
}

void erase_primary_keys_for_entry_locked(const std::shared_ptr<CreditEntry>& entry) {
  if (!entry) {
    return;
  }
  auto& registry = credit_registry();
  for (auto it = registry.begin(); it != registry.end();) {
    if (it->second == entry) {
      it = registry.erase(it);
    } else {
      ++it;
    }
  }
}

void release_credit_entry(const std::shared_ptr<CreditEntry>& entry, bool by_output) {
  if (!entry) {
    return;
  }
  bool expected = false;
  if (!entry->released.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
    return;
  }
  auto release_lane = [by_output](const RealtimeFrameCreditLanePtr& lane) {
    if (!lane || !lane->gate) {
      return;
    }
    lane->gate->release();
    if (by_output) {
      lane->released_by_output.fetch_add(1, std::memory_order_relaxed);
    } else {
      lane->released_without_output.fetch_add(1, std::memory_order_relaxed);
    }
    if (lane->wake) {
      lane->wake();
    }
  };
  release_lane(entry->lane);
  for (const auto& lane : entry->companion_lanes) {
    release_lane(lane);
  }
}

bool release_credit_for_exact_key(const CreditKey& key, const char* mode, bool by_output) {
  if (key.namespace_id == 0 || key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<CreditEntry> entry;
  {
    std::lock_guard<std::mutex> lock(credit_registry_mutex());
    auto& registry = credit_registry();
    auto it = registry.find(key);
    if (it == registry.end()) {
      return false;
    }
    entry = it->second;
    erase_aliases_for_entry_locked(entry);
    registry.erase(it);
  }
  if (credit_debug_enabled()) {
    std::fprintf(stderr,
                 "[graph][credit] release ns=%llu stream=%s frame=%lld mode=%s by_output=%d\n",
                 static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                 static_cast<long long>(key.frame_id), mode ? mode : "key", by_output ? 1 : 0);
  }
  release_credit_entry(entry, by_output);
  return true;
}

bool release_credit_for_unqualified_key(const CreditKey& key, const char* mode, bool by_output) {
  if (key.stream_id.empty() || key.frame_id < 0) {
    return false;
  }
  std::shared_ptr<CreditEntry> entry;
  CreditKey selected;
  bool found = false;
  bool ambiguous = false;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  {
    std::lock_guard<std::mutex> lock(credit_registry_mutex());
    auto& registry = credit_registry();
    for (const auto& item : registry) {
      if (item.first.stream_id != key.stream_id || item.first.frame_id != key.frame_id ||
          !item.second) {
        continue;
      }
      if (found) {
        ambiguous = true;
      }
      if (!found || item.second->sequence < best_sequence) {
        selected = item.first;
        entry = item.second;
        best_sequence = item.second->sequence;
        found = true;
      }
    }
    if (found) {
      erase_aliases_for_entry_locked(entry);
      registry.erase(selected);
    }
  }
  if (!found) {
    return false;
  }
  if (credit_debug_enabled()) {
    std::fprintf(stderr,
                 "[graph][credit] release ns=%llu stream=%s frame=%lld mode=%s "
                 "by_output=%d unqualified=%d ambiguous=%d\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 static_cast<long long>(selected.frame_id), mode ? mode : "key", by_output ? 1 : 0,
                 1, ambiguous ? 1 : 0);
  }
  release_credit_entry(entry, by_output);
  return true;
}

bool find_credit_entry_for_credit_locked(const RealtimeFrameCredit& credit,
                                         std::shared_ptr<CreditEntry>* out_entry,
                                         CreditKey* out_key) {
  auto& registry = credit_registry();
  const CreditKey requested{credit.namespace_id, credit.stream_id, credit.frame_id};
  if (requested.namespace_id != 0 && !requested.stream_id.empty() && requested.frame_id >= 0) {
    auto it = registry.find(requested);
    if (it != registry.end()) {
      if (out_entry) {
        *out_entry = it->second;
      }
      if (out_key) {
        *out_key = it->first;
      }
      return true;
    }
  }
  if (requested.stream_id.empty() || requested.frame_id < 0) {
    return false;
  }

  bool found = false;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  for (const auto& item : registry) {
    if (item.first.stream_id != requested.stream_id || item.first.frame_id != requested.frame_id ||
        !item.second) {
      continue;
    }
    if (!found || item.second->sequence < best_sequence) {
      if (out_entry) {
        *out_entry = item.second;
      }
      if (out_key) {
        *out_key = item.first;
      }
      best_sequence = item.second->sequence;
      found = true;
    }
  }
  return found;
}

bool alias_key_matches_unqualified(const CreditAliasKey& candidate, const CreditAliasKey& query) {
  if (candidate.seq != query.seq || candidate.kind != query.kind) {
    return false;
  }
  if (query.kind == CreditSeqKind::InputSeq && candidate.stream_id.empty()) {
    return true;
  }
  if (!query.stream_id.empty()) {
    return candidate.stream_id == query.stream_id;
  }
  return candidate.stream_id.empty();
}

bool release_credit_for_alias_key(const CreditAliasKey& key, const char* mode, bool by_output) {
  if (key.seq < 0) {
    return false;
  }

  std::shared_ptr<CreditEntry> entry;
  CreditAliasKey selected;
  bool found = false;
  bool ambiguous = false;
  std::uint64_t best_sequence = std::numeric_limits<std::uint64_t>::max();
  {
    std::lock_guard<std::mutex> lock(credit_registry_mutex());
    auto& aliases = credit_alias_registry();
    if (key.namespace_id != 0) {
      auto it = aliases.find(key);
      if (it != aliases.end()) {
        selected = it->first;
        entry = it->second;
        found = true;
      }
    }
    if (!found) {
      for (const auto& item : aliases) {
        if (!item.second || !alias_key_matches_unqualified(item.first, key)) {
          continue;
        }
        if (found) {
          ambiguous = true;
        }
        if (!found || item.second->sequence < best_sequence) {
          selected = item.first;
          entry = item.second;
          best_sequence = item.second->sequence;
          found = true;
        }
      }
    }
    if (found) {
      erase_aliases_for_entry_locked(entry);
      erase_primary_keys_for_entry_locked(entry);
    }
  }
  if (!found) {
    return false;
  }
  if (credit_debug_enabled()) {
    std::fprintf(stderr,
                 "[graph][credit] release-alias ns=%llu stream=%s %s=%lld mode=%s "
                 "by_output=%d ambiguous=%d\n",
                 static_cast<unsigned long long>(selected.namespace_id), selected.stream_id.c_str(),
                 credit_seq_kind_name(selected.kind), static_cast<long long>(selected.seq),
                 mode ? mode : "key", by_output ? 1 : 0, ambiguous ? 1 : 0);
  }
  release_credit_entry(entry, by_output);
  return true;
}

} // namespace

RealtimeFrameCreditLanePtr make_realtime_frame_credit_lane(int credit_limit,
                                                           RealtimeFrameCreditWakeFn wake_fn) {
  return std::make_shared<RealtimeFrameCreditLane>(credit_limit, std::move(wake_fn));
}

std::uint64_t next_realtime_frame_credit_namespace() {
  std::uint64_t ns = 0;
  do {
    ns = credit_namespace_counter().fetch_add(1, std::memory_order_relaxed);
  } while (ns == 0);
  return ns;
}

bool register_realtime_frame_credit(
    std::uint64_t namespace_id, const std::string& stream_id, std::int64_t frame_id,
    const RealtimeFrameCreditLanePtr& lane,
    const std::vector<RealtimeFrameCreditLanePtr>& companion_lanes) {
  if (!lane || !lane->gate || !lane->gate->enabled() || namespace_id == 0 || stream_id.empty() ||
      frame_id < 0) {
    if (lane) {
      lane->missing_key.fetch_add(1, std::memory_order_relaxed);
    }
    if (credit_debug_enabled()) {
      static std::atomic<int> invalid_logs{0};
      const int seen = invalid_logs.fetch_add(1, std::memory_order_relaxed);
      if (seen < 256) {
        std::fprintf(stderr,
                     "[graph][credit] register-invalid ns=%llu stream=%s frame=%lld "
                     "lane=%d gate=%d enabled=%d inflight=%d limit=%d\n",
                     static_cast<unsigned long long>(namespace_id),
                     stream_id.empty() ? "<empty>" : stream_id.c_str(),
                     static_cast<long long>(frame_id), lane ? 1 : 0, (lane && lane->gate) ? 1 : 0,
                     (lane && lane->gate && lane->gate->enabled()) ? 1 : 0,
                     (lane && lane->gate) ? lane->gate->inflight() : -1,
                     (lane && lane->gate) ? lane->gate->credit_limit() : -1);
      }
    }
    return false;
  }

  auto entry = std::make_shared<CreditEntry>();
  entry->lane = lane;
  entry->companion_lanes = companion_lanes;
  entry->sequence = credit_sequence_counter().fetch_add(1, std::memory_order_relaxed);

  std::shared_ptr<CreditEntry> replaced;
  {
    std::lock_guard<std::mutex> lock(credit_registry_mutex());
    CreditKey key{namespace_id, stream_id, frame_id};
    auto& registry = credit_registry();
    auto it = registry.find(key);
    if (it != registry.end()) {
      replaced = it->second;
      erase_aliases_for_entry_locked(replaced);
      it->second = entry;
    } else {
      registry.emplace(std::move(key), entry);
    }
  }
  lane->registered.fetch_add(1, std::memory_order_relaxed);
  if (replaced) {
    release_credit_entry(replaced, /*by_output=*/false);
  }
  if (credit_debug_enabled()) {
    std::fprintf(
        stderr, "[graph][credit] register ns=%llu stream=%s frame=%lld inflight=%d limit=%d\n",
        static_cast<unsigned long long>(namespace_id), stream_id.c_str(),
        static_cast<long long>(frame_id), lane->gate->inflight(), lane->gate->credit_limit());
  }
  return true;
}

bool alias_registered_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                             const Sample& sample, const char* mode) {
  struct SeqCandidate {
    CreditSeqKind kind = CreditSeqKind::InputSeq;
    std::int64_t value = -1;
  };
  std::vector<SeqCandidate> seqs;
  const auto add_seq = [&seqs](CreditSeqKind kind, std::int64_t value) {
    if (value < 0) {
      return;
    }
    const auto found = std::find_if(seqs.begin(), seqs.end(), [&](const SeqCandidate& existing) {
      return existing.kind == kind && existing.value == value;
    });
    if (found == seqs.end()) {
      seqs.push_back(SeqCandidate{kind, value});
    }
  };
  add_seq(CreditSeqKind::InputSeq, sample.input_seq);
  add_seq(CreditSeqKind::OrigInputSeq, sample.orig_input_seq);
  if (seqs.empty()) {
    if (credit_debug_enabled()) {
      static std::atomic<int> no_seq_logs{0};
      const int seen = no_seq_logs.fetch_add(1, std::memory_order_relaxed);
      if (seen < 128) {
        std::fprintf(
            stderr, "[graph][credit] alias-skip-no-seq mode=%s stream=%s frame=%lld credits=%zu\n",
            mode ? mode : "alias", sample.stream_id.empty() ? "<empty>" : sample.stream_id.c_str(),
            static_cast<long long>(sample.frame_id), credits.size());
      }
    }
    return false;
  }

  bool aliased_any = false;
  std::lock_guard<std::mutex> lock(credit_registry_mutex());
  auto& aliases = credit_alias_registry();
  for (const auto& credit : credits) {
    std::shared_ptr<CreditEntry> entry;
    CreditKey primary;
    if (!find_credit_entry_for_credit_locked(credit, &entry, &primary) || !entry) {
      if (credit_debug_enabled()) {
        static std::atomic<int> miss_logs{0};
        const int seen = miss_logs.fetch_add(1, std::memory_order_relaxed);
        if (seen < 128) {
          std::fprintf(
              stderr,
              "[graph][credit] alias-miss ns=%llu stream=%s frame=%lld input=%lld "
              "orig=%lld mode=%s sample_stream=%s sample_frame=%lld sample_input=%lld "
              "sample_orig=%lld\n",
              static_cast<unsigned long long>(credit.namespace_id),
              credit.stream_id.empty() ? "<empty>" : credit.stream_id.c_str(),
              static_cast<long long>(credit.frame_id), static_cast<long long>(credit.input_seq),
              static_cast<long long>(credit.orig_input_seq), mode ? mode : "alias",
              sample.stream_id.empty() ? "<empty>" : sample.stream_id.c_str(),
              static_cast<long long>(sample.frame_id), static_cast<long long>(sample.input_seq),
              static_cast<long long>(sample.orig_input_seq));
        }
      }
      continue;
    }
    const std::string stream_id = !sample.stream_id.empty() ? sample.stream_id : primary.stream_id;
    for (const auto& seq : seqs) {
      const CreditAliasKey stream_alias{primary.namespace_id, stream_id, seq.value, seq.kind};
      std::vector<CreditAliasKey> aliases_to_add{stream_alias};
      if (seq.kind == CreditSeqKind::InputSeq) {
        // The sanitized pipeline input_seq is segment-global, so it is safe as
        // a streamless fallback for outputs that lose public stream metadata.
        // orig_input_seq is stream-local in live fan-in graphs; a streamless
        // orig alias can release the wrong stream's credit when many streams
        // carry the same original frame number.
        aliases_to_add.push_back(
            CreditAliasKey{primary.namespace_id, std::string{}, seq.value, seq.kind});
      }
      for (const auto& alias : aliases_to_add) {
        auto it = aliases.find(alias);
        if (it != aliases.end() && it->second == entry) {
          continue;
        }
        aliases[alias] = entry;
        const auto found =
            std::find(entry->aliases.begin(), entry->aliases.end(), alias) != entry->aliases.end();
        if (!found) {
          entry->aliases.push_back(alias);
        }
        aliased_any = true;
        if (credit_debug_enabled()) {
          std::fprintf(stderr,
                       "[graph][credit] alias ns=%llu stream=%s frame=%lld %s=%lld mode=%s\n",
                       static_cast<unsigned long long>(primary.namespace_id),
                       alias.stream_id.empty() ? "<any>" : alias.stream_id.c_str(),
                       static_cast<long long>(primary.frame_id), credit_seq_kind_name(seq.kind),
                       static_cast<long long>(seq.value), mode ? mode : "alias");
        }
      }
    }
  }
  return aliased_any;
}

bool release_registered_realtime_frame_credit(const RealtimeFrameCredit& credit, const char* mode,
                                              bool by_output) {
  const CreditKey key{credit.namespace_id, credit.stream_id, credit.frame_id};
  bool released = false;
  if (key.namespace_id != 0) {
    released = release_credit_for_exact_key(key, mode, by_output);
  } else {
    released = release_credit_for_unqualified_key(key, mode, by_output);
  }
  if (!released && credit.input_seq >= 0) {
    released =
        release_credit_for_alias_key(CreditAliasKey{credit.namespace_id, credit.stream_id,
                                                    credit.input_seq, CreditSeqKind::InputSeq},
                                     mode, by_output);
  }
  if (!released && credit.orig_input_seq >= 0) {
    released = release_credit_for_alias_key(CreditAliasKey{credit.namespace_id, credit.stream_id,
                                                           credit.orig_input_seq,
                                                           CreditSeqKind::OrigInputSeq},
                                            mode, by_output);
  }
  if (!released && credit_debug_enabled()) {
    const int seen = credit_miss_debug_logs().fetch_add(1, std::memory_order_relaxed);
    if (seen < 128) {
      std::fprintf(stderr,
                   "[graph][credit] release-miss ns=%llu stream=%s frame=%lld mode=%s "
                   "by_output=%d\n",
                   static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                   static_cast<long long>(key.frame_id), mode ? mode : "key", by_output ? 1 : 0);
    }
  }
  return released;
}

void release_all_registered_realtime_frame_credits(std::uint64_t namespace_id, const char* mode) {
  if (namespace_id == 0) {
    return;
  }
  std::vector<std::shared_ptr<CreditEntry>> entries;
  std::vector<CreditKey> keys;
  {
    std::lock_guard<std::mutex> lock(credit_registry_mutex());
    auto& registry = credit_registry();
    for (auto it = registry.begin(); it != registry.end();) {
      if (it->first.namespace_id == namespace_id) {
        keys.push_back(it->first);
        entries.push_back(it->second);
        erase_aliases_for_entry_locked(it->second);
        it = registry.erase(it);
      } else {
        ++it;
      }
    }
  }
  const bool dump_release_all = env_bool("SIMA_GRAPH_REALTIME_CREDIT_DUMP_RELEASE_ALL", false);
  if ((credit_debug_enabled() || dump_release_all) && !entries.empty()) {
    std::fprintf(stderr, "[graph][credit] release-all ns=%llu count=%zu mode=%s\n",
                 static_cast<unsigned long long>(namespace_id), entries.size(),
                 mode ? mode : "release-all");
    if (dump_release_all) {
      for (const auto& key : keys) {
        std::fprintf(stderr,
                     "[graph][credit] release-all-key ns=%llu stream=%s frame=%lld mode=%s\n",
                     static_cast<unsigned long long>(key.namespace_id), key.stream_id.c_str(),
                     static_cast<long long>(key.frame_id), mode ? mode : "release-all");
      }
    }
  }
  for (const auto& entry : entries) {
    release_credit_entry(entry, /*by_output=*/false);
  }
}

} // namespace simaai::neat::pipeline_internal

#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace simaai::neat::pipeline_internal {

// Bounds retained cache ownership, not active callers. Construction and final
// destruction can start/close a Run, so neither may happen under the cache lock.
template <class Key, class Value, class Hash> class BoundedStageCache {
public:
  explicit BoundedStageCache(std::size_t capacity) : capacity_(capacity) {
    if (!capacity_)
      throw std::invalid_argument("stage cache capacity must be positive");
  }

  template <class Builder> std::shared_ptr<Value> get_or_build(const Key& key, Builder&& builder) {
    {
      std::lock_guard lock(mutex_);
      if (const auto hit = entries_.find(key); hit != entries_.end()) {
        recent_.splice(recent_.begin(), recent_, hit->second.position);
        return hit->second.value;
      }
    }
    auto value = builder();
    if (!value)
      throw std::invalid_argument("stage cache builder returned no value");
    std::shared_ptr<Value> retired;
    {
      std::lock_guard lock(mutex_);
      if (const auto hit = entries_.find(key); hit != entries_.end()) {
        recent_.splice(recent_.begin(), recent_, hit->second.position);
        return hit->second.value;
      }
      const auto position = recent_.insert(recent_.begin(), key);
      try {
        entries_.emplace(key, Entry{value, position});
      } catch (...) {
        recent_.erase(position);
        throw;
      }
      if (entries_.size() > capacity_) {
        const auto oldest = entries_.find(recent_.back());
        retired = std::move(oldest->second.value);
        entries_.erase(oldest);
        recent_.pop_back();
      }
    }
    return value;
  }

  std::size_t size() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
  }

private:
  struct Entry {
    std::shared_ptr<Value> value;
    typename std::list<Key>::iterator position;
  };
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::list<Key> recent_;
  std::unordered_map<Key, Entry, Hash> entries_;
};

} // namespace simaai::neat::pipeline_internal

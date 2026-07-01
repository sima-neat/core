#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <atomic>
#include <cstdint>
#include <memory>

namespace simaai::neat::pipeline_internal {

class HolderLoanGate {
public:
  explicit HolderLoanGate(int credit_limit = 0) : credit_limit_(credit_limit) {}

  void configure(int credit_limit) noexcept {
    credit_limit_.store(credit_limit, std::memory_order_relaxed);
  }

  bool enabled() const noexcept {
    return credit_limit_.load(std::memory_order_relaxed) > 0;
  }

  bool try_acquire() noexcept {
    const int limit = credit_limit_.load(std::memory_order_relaxed);
    if (limit <= 0) {
      return true;
    }
    int cur = inflight_.load(std::memory_order_relaxed);
    while (cur < limit) {
      if (inflight_.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) {
        return true;
      }
    }
    skipped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  void release() noexcept {
    int cur = inflight_.load(std::memory_order_relaxed);
    while (cur > 0) {
      if (inflight_.compare_exchange_weak(cur, cur - 1, std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) {
        released_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    overrelease_.fetch_add(1, std::memory_order_relaxed);
  }

  int credit_limit() const noexcept {
    return credit_limit_.load(std::memory_order_relaxed);
  }
  int inflight() const noexcept {
    return inflight_.load(std::memory_order_relaxed);
  }
  std::uint64_t skipped() const noexcept {
    return skipped_.load(std::memory_order_relaxed);
  }
  std::uint64_t released() const noexcept {
    return released_.load(std::memory_order_relaxed);
  }
  std::uint64_t overrelease() const noexcept {
    return overrelease_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> credit_limit_{0};
  std::atomic<int> inflight_{0};
  std::atomic<std::uint64_t> skipped_{0};
  std::atomic<std::uint64_t> released_{0};
  std::atomic<std::uint64_t> overrelease_{0};
};

using HolderLoanGatePtr = std::shared_ptr<HolderLoanGate>;

} // namespace simaai::neat::pipeline_internal

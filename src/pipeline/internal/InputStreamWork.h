#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace simaai::neat::pipeline_internal {

// Counts work before it can produce output. Admission and watchdog checks are
// serialized; terminal-pad observations use only atomics and never block GStreamer.
class InputStreamWork {
public:
  void begin() {
    std::lock_guard<std::mutex> lock(mu_);
    if (pending() == 0)
      last_progress_ns_.store(now_ns(), std::memory_order_release);
    pending_.fetch_add(1, std::memory_order_release);
  }

  void complete() noexcept {
    // Publication precedes retirement so a watchdog seeing the new count also
    // sees progress. CAS keeps time monotonic if output observers overlap.
    const auto now = now_ns();
    auto previous = last_progress_ns_.load(std::memory_order_acquire);
    while (previous < now &&
           !last_progress_ns_.compare_exchange_weak(previous, now, std::memory_order_release,
                                                    std::memory_order_relaxed)) {
    }
    retire();
  }

  void cancel() {
    std::lock_guard<std::mutex> lock(mu_);
    retire(); // Rejection must not extend older work's processing deadline.
  }

  std::int64_t pending() const noexcept {
    return pending_.load(std::memory_order_acquire);
  }

  bool timed_out(int timeout_ms) const {
    std::lock_guard<std::mutex> lock(mu_);
    return timeout_ms > 0 && pending() > 0 &&
           now_ns() - last_progress_ns_.load(std::memory_order_acquire) >
               static_cast<std::int64_t>(timeout_ms) * 1000000;
  }

private:
  static std::int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void retire() noexcept {
    auto count = pending();
    while (count > 0 && !pending_.compare_exchange_weak(count, count - 1, std::memory_order_release,
                                                        std::memory_order_relaxed)) {
    }
  }

  mutable std::mutex mu_;
  std::atomic<std::int64_t> pending_{0};
  std::atomic<std::int64_t> last_progress_ns_{0};
};

// RAII rollback also covers an exception after admission but before appsrc accepts.
template <class Work> class InputPushGuard {
public:
  explicit InputPushGuard(Work& work) : work_(work) {
    work_.begin();
  }
  InputPushGuard(const InputPushGuard&) = delete;
  InputPushGuard& operator=(const InputPushGuard&) = delete;
  ~InputPushGuard() {
    if (!accepted)
      work_.cancel();
  }
  bool accepted = false;

private:
  Work& work_;
};

} // namespace simaai::neat::pipeline_internal

/**
 * @file
 * @ingroup graph_runtime
 * @brief Simple bounded blocking queue for graph runtime.
 */
#pragma once

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <cstdint>
#include <mutex>
#include <utility>

namespace simaai::neat::graph::runtime {

enum class QueueTelemetryLevel : std::uint8_t {
  CountersOnly,
  Timing,
};

/**
 * @brief Thread-safe bounded blocking queue used by the runtime mailboxes.
 *
 * Supports blocking and non-blocking enqueue/dequeue with optional millisecond timeouts,
 * and a `close()` operation that wakes blocked waiters and refuses further pushes. A
 * capacity of `0` means unbounded.
 *
 * @tparam T Element type stored in the queue.
 *
 * @see StageMailbox
 * @ingroup graph
 */
template <class T> class BlockingQueue {
public:
  /**
   * @brief Lock-free snapshot of queue activity.
   *
   * `push_wait_ns` / `pop_wait_ns` include mutex acquisition plus condition-variable wait
   * time. For an uncontended queue this is close to enqueue/dequeue call overhead; when a
   * queue is full/empty it captures real backpressure/idle wait.
   */
  struct Stats {
    std::uint64_t push_count = 0;
    std::uint64_t pop_count = 0;
    std::uint64_t push_timeout_count = 0;
    std::uint64_t pop_timeout_count = 0;
    std::uint64_t push_closed_count = 0;
    std::uint64_t pop_closed_empty_count = 0;
    std::uint64_t push_wait_ns = 0;
    std::uint64_t pop_wait_ns = 0;
    std::uint64_t max_push_wait_ns = 0;
    std::uint64_t max_pop_wait_ns = 0;
    std::uint64_t residence_count = 0;
    std::uint64_t residence_ns = 0;
    std::uint64_t max_residence_ns = 0;
    std::size_t high_watermark = 0;
    std::size_t current_size = 0;
    std::size_t capacity = 0;
    bool closed = false;
    bool timing_enabled = false;
  };

  /// Construct a queue with the given capacity (0 = unbounded).
  explicit BlockingQueue(std::size_t capacity = 0) : capacity_(capacity) {}

  void set_telemetry_level(QueueTelemetryLevel level) noexcept {
    telemetry_level_.store(level, std::memory_order_release);
  }

  bool timing_enabled() const noexcept {
    return telemetry_level_.load(std::memory_order_acquire) == QueueTelemetryLevel::Timing;
  }

  /// Push `item` (copy). Blocks up to `timeout_ms` (or forever if -1). Returns false on
  /// close/timeout.
  bool push(const T& item, int timeout_ms = -1) {
    const bool timing = timing_enabled();
    const auto t0 =
        timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || has_capacity_locked(); });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || has_capacity_locked(); })) {
        record_push_wait(t0, timing);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (closed_) {
        record_push_wait(t0, timing);
        push_closed_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (!has_capacity_locked()) {
        record_push_wait(t0, timing);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
    record_push_wait(t0, timing);
    queue_.push_back(QueueEntry{item, timing ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{}});
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Push `item` (move). Blocks up to `timeout_ms` (or forever if -1). Returns false on
  /// close/timeout.
  bool push(T&& item, int timeout_ms = -1) {
    const bool timing = timing_enabled();
    const auto t0 =
        timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || has_capacity_locked(); });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || has_capacity_locked(); })) {
        record_push_wait(t0, timing);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (closed_) {
        record_push_wait(t0, timing);
        push_closed_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (!has_capacity_locked()) {
        record_push_wait(t0, timing);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
    record_push_wait(t0, timing);
    queue_.push_back(QueueEntry{std::move(item), timing ? std::chrono::steady_clock::now()
                                                        : std::chrono::steady_clock::time_point{}});
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking copy push; returns false if closed or full.
  bool try_push(const T& item) {
    const bool timing = timing_enabled();
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (!has_capacity_locked()) {
      push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    queue_.push_back(QueueEntry{item, timing ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{}});
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking move push; returns false if closed or full.
  bool try_push(T&& item) {
    const bool timing = timing_enabled();
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (!has_capacity_locked()) {
      push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    queue_.push_back(QueueEntry{std::move(item), timing ? std::chrono::steady_clock::now()
                                                        : std::chrono::steady_clock::time_point{}});
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Restore an item to the front of the queue. Intended for a consumer that popped an item but
  /// could not acquire an external resource needed to publish it. Returns false if closed or full.
  bool restore_front(T&& item) {
    const bool timing = timing_enabled();
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (!has_capacity_locked()) {
      push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    queue_.push_front(QueueEntry{std::move(item), timing
                                                      ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{}});
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Pop the next item while keeping its bounded-capacity slot reserved for either
  /// `restore_reserved_front()` or `release_restore_reservation()`. Use this for two-phase
  /// consumers that may have to put the exact item back after checking an external resource.
  bool pop_with_restore_reservation(T& out, int timeout_ms = -1) {
    return pop_impl(out, timeout_ms, true);
  }

  /// Restore an item popped by `pop_with_restore_reservation()` without exposing the reserved slot
  /// to producers in between. Returns false only if the queue was closed or no reservation exists.
  bool restore_reserved_front(T&& item) {
    const bool timing = timing_enabled();
    std::lock_guard<std::mutex> lock(mu_);
    if (restore_reservations_ == 0) {
      return false;
    }
    --restore_reservations_;
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      cv_not_full_.notify_one();
      return false;
    }
    queue_.push_front(QueueEntry{std::move(item), timing
                                                      ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{}});
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Release a slot reserved by `pop_with_restore_reservation()` after the popped item has been
  /// successfully consumed and will not be restored.
  bool release_restore_reservation() {
    std::lock_guard<std::mutex> lock(mu_);
    if (restore_reservations_ == 0) {
      return false;
    }
    --restore_reservations_;
    cv_not_full_.notify_one();
    return true;
  }

  /// Pop the next item into `out`. Blocks up to `timeout_ms` (or forever if -1). Returns false if
  /// closed and empty (or on timeout).
  bool pop(T& out, int timeout_ms = -1) {
    return pop_impl(out, timeout_ms, false);
  }

  /// Close the queue: wakes blocked threads and refuses further pushes.
  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

  /// Returns true iff the queue has been closed.
  bool closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

  /// Current queue size.
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
  }

  /// Return current queue telemetry.
  Stats stats() const {
    Stats s;
    s.push_count = push_count_.load(std::memory_order_relaxed);
    s.pop_count = pop_count_.load(std::memory_order_relaxed);
    s.push_timeout_count = push_timeout_count_.load(std::memory_order_relaxed);
    s.pop_timeout_count = pop_timeout_count_.load(std::memory_order_relaxed);
    s.push_closed_count = push_closed_count_.load(std::memory_order_relaxed);
    s.pop_closed_empty_count = pop_closed_empty_count_.load(std::memory_order_relaxed);
    s.push_wait_ns = push_wait_ns_.load(std::memory_order_relaxed);
    s.pop_wait_ns = pop_wait_ns_.load(std::memory_order_relaxed);
    s.max_push_wait_ns = max_push_wait_ns_.load(std::memory_order_relaxed);
    s.max_pop_wait_ns = max_pop_wait_ns_.load(std::memory_order_relaxed);
    s.residence_count = residence_count_.load(std::memory_order_relaxed);
    s.residence_ns = residence_ns_.load(std::memory_order_relaxed);
    s.max_residence_ns = max_residence_ns_.load(std::memory_order_relaxed);
    s.high_watermark = high_watermark_.load(std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mu_);
      s.current_size = effective_size_locked();
      s.capacity = capacity_;
      s.closed = closed_;
    }
    s.timing_enabled = timing_enabled();
    return s;
  }

private:
  static std::uint64_t duration_ns_since(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
  }

  static void atomic_max(std::atomic<std::uint64_t>& dst, std::uint64_t value) {
    std::uint64_t cur = dst.load(std::memory_order_relaxed);
    while (cur < value && !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
    }
  }

  static void atomic_max_size(std::atomic<std::size_t>& dst, std::size_t value) {
    std::size_t cur = dst.load(std::memory_order_relaxed);
    while (cur < value && !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
    }
  }

  bool has_capacity_locked() const {
    return capacity_ == 0 || effective_size_locked() < capacity_;
  }

  std::size_t effective_size_locked() const {
    return queue_.size() + restore_reservations_;
  }

  bool pop_impl(T& out, int timeout_ms, bool reserve_restore_slot) {
    const bool timing = timing_enabled();
    const auto t0 =
        timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::unique_lock<std::mutex> lock(mu_);
    if (timeout_ms < 0) {
      cv_not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    } else {
      cv_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return closed_ || !queue_.empty(); });
    }
    record_pop_wait(t0, timing);
    if (queue_.empty()) {
      if (closed_) {
        pop_closed_empty_count_.fetch_add(1, std::memory_order_relaxed);
      } else {
        pop_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    QueueEntry entry = std::move(queue_.front());
    queue_.pop_front();
    if (reserve_restore_slot) {
      ++restore_reservations_;
    }
    record_residence(entry.enqueue_time, timing);
    out = std::move(entry.value);
    pop_count_.fetch_add(1, std::memory_order_relaxed);
    if (!reserve_restore_slot) {
      cv_not_full_.notify_one();
    }
    return true;
  }

  void record_push_wait(std::chrono::steady_clock::time_point start, bool enabled) const {
    if (!enabled) {
      return;
    }
    const std::uint64_t ns = duration_ns_since(start);
    push_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_max(max_push_wait_ns_, ns);
  }

  void record_pop_wait(std::chrono::steady_clock::time_point start, bool enabled) const {
    if (!enabled) {
      return;
    }
    const std::uint64_t ns = duration_ns_since(start);
    pop_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_max(max_pop_wait_ns_, ns);
  }

  void record_residence(std::chrono::steady_clock::time_point enqueue_time, bool enabled) const {
    if (!enabled || enqueue_time.time_since_epoch().count() == 0) {
      return;
    }
    const std::uint64_t ns = duration_ns_since(enqueue_time);
    residence_count_.fetch_add(1, std::memory_order_relaxed);
    residence_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_max(max_residence_ns_, ns);
  }

  void update_high_watermark_locked() const {
    atomic_max_size(high_watermark_, queue_.size());
  }

  mutable std::mutex mu_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  struct QueueEntry {
    T value;
    std::chrono::steady_clock::time_point enqueue_time;
  };

  std::deque<QueueEntry> queue_;
  std::size_t capacity_ = 0;
  std::size_t restore_reservations_ = 0;
  bool closed_ = false;
  mutable std::atomic<std::uint64_t> push_count_{0};
  mutable std::atomic<std::uint64_t> pop_count_{0};
  mutable std::atomic<std::uint64_t> push_timeout_count_{0};
  mutable std::atomic<std::uint64_t> pop_timeout_count_{0};
  mutable std::atomic<std::uint64_t> push_closed_count_{0};
  mutable std::atomic<std::uint64_t> pop_closed_empty_count_{0};
  mutable std::atomic<std::uint64_t> push_wait_ns_{0};
  mutable std::atomic<std::uint64_t> pop_wait_ns_{0};
  mutable std::atomic<std::uint64_t> max_push_wait_ns_{0};
  mutable std::atomic<std::uint64_t> max_pop_wait_ns_{0};
  mutable std::atomic<std::uint64_t> residence_count_{0};
  mutable std::atomic<std::uint64_t> residence_ns_{0};
  mutable std::atomic<std::uint64_t> max_residence_ns_{0};
  mutable std::atomic<std::size_t> high_watermark_{0};
  mutable std::atomic<QueueTelemetryLevel> telemetry_level_{QueueTelemetryLevel::CountersOnly};
};

} // namespace simaai::neat::graph::runtime

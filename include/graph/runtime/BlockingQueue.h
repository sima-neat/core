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
    std::size_t high_watermark = 0;
    std::size_t current_size = 0;
    std::size_t capacity = 0;
    bool closed = false;
  };

  /// Construct a queue with the given capacity (0 = unbounded).
  explicit BlockingQueue(std::size_t capacity = 0) : capacity_(capacity) {}

  /// Push `item` (copy). Blocks up to `timeout_ms` (or forever if -1). Returns false on
  /// close/timeout.
  bool push(const T& item, int timeout_ms = -1) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || queue_.size() < capacity_; })) {
        record_push_wait(t0);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (closed_ || (capacity_ > 0 && queue_.size() >= capacity_)) {
        record_push_wait(t0);
        push_closed_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
    record_push_wait(t0);
    queue_.push_back(item);
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Push `item` (move). Blocks up to `timeout_ms` (or forever if -1). Returns false on
  /// close/timeout.
  bool push(T&& item, int timeout_ms = -1) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || queue_.size() < capacity_; })) {
        record_push_wait(t0);
        push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (closed_ || (capacity_ > 0 && queue_.size() >= capacity_)) {
        record_push_wait(t0);
        push_closed_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
    record_push_wait(t0);
    queue_.push_back(std::move(item));
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking copy push; returns false if closed or full.
  bool try_push(const T& item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0 && queue_.size() >= capacity_) {
      push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    queue_.push_back(item);
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking move push; returns false if closed or full.
  bool try_push(T&& item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      push_closed_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (capacity_ > 0 && queue_.size() >= capacity_) {
      push_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    queue_.push_back(std::move(item));
    push_count_.fetch_add(1, std::memory_order_relaxed);
    update_high_watermark_locked();
    cv_not_empty_.notify_one();
    return true;
  }

  /// Pop the next item into `out`. Blocks up to `timeout_ms` (or forever if -1). Returns false if
  /// closed and empty (or on timeout).
  bool pop(T& out, int timeout_ms = -1) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    if (timeout_ms < 0) {
      cv_not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    } else {
      cv_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return closed_ || !queue_.empty(); });
    }
    record_pop_wait(t0);
    if (queue_.empty()) {
      if (closed_) {
        pop_closed_empty_count_.fetch_add(1, std::memory_order_relaxed);
      } else {
        pop_timeout_count_.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    pop_count_.fetch_add(1, std::memory_order_relaxed);
    cv_not_full_.notify_one();
    return true;
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
    s.high_watermark = high_watermark_.load(std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mu_);
      s.current_size = queue_.size();
      s.capacity = capacity_;
      s.closed = closed_;
    }
    return s;
  }

private:
  static std::uint64_t duration_ns_since(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  }

  static void atomic_max(std::atomic<std::uint64_t>& dst, std::uint64_t value) {
    std::uint64_t cur = dst.load(std::memory_order_relaxed);
    while (cur < value &&
           !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
    }
  }

  static void atomic_max_size(std::atomic<std::size_t>& dst, std::size_t value) {
    std::size_t cur = dst.load(std::memory_order_relaxed);
    while (cur < value &&
           !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
    }
  }

  void record_push_wait(std::chrono::steady_clock::time_point start) const {
    const std::uint64_t ns = duration_ns_since(start);
    push_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_max(max_push_wait_ns_, ns);
  }

  void record_pop_wait(std::chrono::steady_clock::time_point start) const {
    const std::uint64_t ns = duration_ns_since(start);
    pop_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_max(max_pop_wait_ns_, ns);
  }

  void update_high_watermark_locked() const {
    atomic_max_size(high_watermark_, queue_.size());
  }

  mutable std::mutex mu_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::deque<T> queue_;
  std::size_t capacity_ = 0;
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
  mutable std::atomic<std::size_t> high_watermark_{0};
};

} // namespace simaai::neat::graph::runtime

/**
 * @file
 * @ingroup graph_runtime
 * @brief Simple bounded blocking queue for graph runtime.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
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
  /// Construct a queue with the given capacity (0 = unbounded).
  explicit BlockingQueue(std::size_t capacity = 0) : capacity_(capacity) {}

  /// Push `item` (copy). Blocks up to `timeout_ms` (or forever if -1). Returns false on close/timeout.
  bool push(const T& item, int timeout_ms = -1) {
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_)
      return false;
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || queue_.size() < capacity_; })) {
        return false;
      }
      if (closed_ || (capacity_ > 0 && queue_.size() >= capacity_))
        return false;
    }
    queue_.push_back(item);
    cv_not_empty_.notify_one();
    return true;
  }

  /// Push `item` (move). Blocks up to `timeout_ms` (or forever if -1). Returns false on close/timeout.
  bool push(T&& item, int timeout_ms = -1) {
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_)
      return false;
    if (capacity_ > 0) {
      if (timeout_ms < 0) {
        cv_not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
      } else if (!cv_not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                        [&] { return closed_ || queue_.size() < capacity_; })) {
        return false;
      }
      if (closed_ || (capacity_ > 0 && queue_.size() >= capacity_))
        return false;
    }
    queue_.push_back(std::move(item));
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking copy push; returns false if closed or full.
  bool try_push(const T& item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_)
      return false;
    if (capacity_ > 0 && queue_.size() >= capacity_)
      return false;
    queue_.push_back(item);
    cv_not_empty_.notify_one();
    return true;
  }

  /// Non-blocking move push; returns false if closed or full.
  bool try_push(T&& item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_)
      return false;
    if (capacity_ > 0 && queue_.size() >= capacity_)
      return false;
    queue_.push_back(std::move(item));
    cv_not_empty_.notify_one();
    return true;
  }

  /// Pop the next item into `out`. Blocks up to `timeout_ms` (or forever if -1). Returns false if closed and empty (or on timeout).
  bool pop(T& out, int timeout_ms = -1) {
    std::unique_lock<std::mutex> lock(mu_);
    if (timeout_ms < 0) {
      cv_not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    } else {
      cv_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return closed_ || !queue_.empty(); });
    }
    if (queue_.empty())
      return false;
    out = std::move(queue_.front());
    queue_.pop_front();
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

private:
  mutable std::mutex mu_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::deque<T> queue_;
  std::size_t capacity_ = 0;
  bool closed_ = false;
};

} // namespace simaai::neat::graph::runtime

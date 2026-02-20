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

template <class T> class BlockingQueue {
public:
  explicit BlockingQueue(std::size_t capacity = 0) : capacity_(capacity) {}

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

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

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

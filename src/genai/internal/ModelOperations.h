#pragma once

#include <condition_variable>
#include <mutex>
#include <stop_token>

namespace simaai::neat::genai::internal {

// Generation, image encoding and adapter changes share one model workspace.
class ModelOperations {
public:
  bool lock(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    if (!available_.wait(lock, stop, [this] { return !busy_; }) || stop.stop_requested())
      return false;
    busy_ = true;
    return true;
  }

  void lock() {
    lock(std::stop_token{});
  }

  bool try_lock() {
    std::lock_guard lock(mutex_);
    if (busy_)
      return false;
    busy_ = true;
    return true;
  }

  void unlock() {
    {
      std::lock_guard lock(mutex_);
      busy_ = false;
    }
    available_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable_any available_;
  bool busy_ = false;
};

} // namespace simaai::neat::genai::internal

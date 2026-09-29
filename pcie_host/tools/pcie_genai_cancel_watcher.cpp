#include "pcie_genai_cancel_watcher.h"

#include <utility>

namespace simaai::neat::pcie::genai::tools {

CancelWatcher::CancelWatcher(std::function<bool()> should_cancel, std::function<void()> cancel,
                             const std::chrono::milliseconds poll)
    : should_cancel_(std::move(should_cancel)), cancel_(std::move(cancel)), poll_(poll),
      thread_([this] { run(); }) {}

CancelWatcher::~CancelWatcher() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

bool CancelWatcher::fired() const {
  return fired_.load();
}

void CancelWatcher::run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    if (should_cancel_()) {
      fired_.store(true);
      lock.unlock(); // do not hold the lock the destructor needs while cancel runs
      try {
        cancel_();
      } catch (...) {
        // An exception leaving this thread would call std::terminate and kill
        // the CLI before it can stop the card backend. Cancel is best effort.
      }
      return; // fire once
    }
    cv_.wait_for(lock, poll_, [this] { return stop_; });
  }
}

} // namespace simaai::neat::pcie::genai::tools

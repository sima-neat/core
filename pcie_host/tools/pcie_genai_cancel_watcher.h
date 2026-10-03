#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace simaai::neat::pcie::genai::tools {

/**
 * @brief Calls @p cancel once, from its own thread, as soon as @p should_cancel
 * returns true.
 *
 * Why: GenerationStream::next() blocks until the card sends something. A
 * Ctrl-C check inside the token loop therefore never runs while the card is
 * silent (long prefill, or a stuck card). This watcher does not wait for a
 * token. Destroy it before the object that @p cancel refers to.
 */
class CancelWatcher {
public:
  CancelWatcher(std::function<bool()> should_cancel, std::function<void()> cancel,
                std::chrono::milliseconds poll = std::chrono::milliseconds(50));
  /// Stops the thread at once (it does not wait for the next poll).
  ~CancelWatcher();

  CancelWatcher(const CancelWatcher&) = delete;
  CancelWatcher& operator=(const CancelWatcher&) = delete;

  /// True once cancel has been called (or is being called).
  bool fired() const;

private:
  void run();

  std::function<bool()> should_cancel_;
  std::function<void()> cancel_;
  std::chrono::milliseconds poll_;
  std::atomic<bool> fired_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread thread_; ///< last member: it starts after all the others exist
};

} // namespace simaai::neat::pcie::genai::tools

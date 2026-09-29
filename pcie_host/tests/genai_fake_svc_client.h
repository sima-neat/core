#pragma once

#include "genai/SvcClient.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace simaai::neat::pcie::genai::internal::test {

/// Scripted stand-in for the simaai_svc daemon. on_notify lets a test play the
/// card: react to the prompt/cancel the transport sends by pushing notes back.
class FakeSvcClient final : public SvcClient {
public:
  using OnNotify =
      std::function<void(FakeSvcClient&, const std::string& tag, const std::string& payload)>;

  OnNotify on_notify;
  unsigned listeners = 1; ///< what notify() reports as the far-side subscriber count

  void subscribe(const std::string& tag) override {
    std::lock_guard<std::mutex> lock(mutex_);
    subscribed_.push_back(tag);
  }

  unsigned notify(const std::string& tag, const std::string& payload) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sent_.push_back({tag, payload});
    }
    if (on_notify) {
      on_notify(*this, tag, payload);
    }
    return listeners;
  }

  RecvStatus recv(SvcNote& out, int timeout_ms) override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                 [&] { return !incoming_.empty() || disconnected_; });
    if (!incoming_.empty()) {
      out = incoming_.front();
      incoming_.pop_front();
      return RecvStatus::Ok;
    }
    return disconnected_ ? RecvStatus::Disconnected : RecvStatus::Timeout;
  }

  void push(const std::string& tag, const std::string& payload) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      incoming_.push_back({tag, payload});
    }
    cv_.notify_all();
  }

  void disconnect() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      disconnected_ = true;
    }
    cv_.notify_all();
  }

  std::vector<SvcNote> sent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sent_;
  }

  std::vector<std::string> subscribed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subscribed_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<SvcNote> incoming_;
  std::vector<SvcNote> sent_;
  std::vector<std::string> subscribed_;
  bool disconnected_ = false;
};

} // namespace simaai::neat::pcie::genai::internal::test

#pragma once

#include <simaai/neat/pcie/Model.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

namespace simaai::neat::pcie::test {

namespace detail {
inline std::atomic<bool>* g_signal_requested = nullptr;

inline void signal_close_guard_handler(int) {
  if (g_signal_requested) {
    g_signal_requested->store(true);
  }
}
} // namespace detail

class SignalCloseGuard {
public:
  explicit SignalCloseGuard(Model& model) : model_(model) {
    requested_.store(false);
    done_.store(false);
    previous_int_ = std::signal(SIGINT, detail::signal_close_guard_handler);
    previous_term_ = std::signal(SIGTERM, detail::signal_close_guard_handler);
    detail::g_signal_requested = &requested_;
    worker_ = std::thread([this] {
      while (!done_.load()) {
        if (requested_.load()) {
          model_.close();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  ~SignalCloseGuard() {
    done_.store(true);
    if (worker_.joinable()) {
      worker_.join();
    }
    if (detail::g_signal_requested == &requested_) {
      detail::g_signal_requested = nullptr;
    }
    std::signal(SIGINT, previous_int_);
    std::signal(SIGTERM, previous_term_);
  }

  bool requested() const {
    return requested_.load();
  }

private:
  Model& model_;
  std::atomic<bool> requested_{false};
  std::atomic<bool> done_{false};
  std::thread worker_;
  void (*previous_int_)(int) = SIG_DFL;
  void (*previous_term_)(int) = SIG_DFL;
};

} // namespace simaai::neat::pcie::test

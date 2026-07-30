#pragma once

#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <utility>

namespace simaai::neat::pcie::test {

template <typename Cancel, typename Producer, typename Consumer>
void run_async_workers(Cancel&& cancel, Producer&& producer, Consumer&& consumer) {
  std::atomic_bool cancelled{false};
  std::mutex error_mutex;
  std::exception_ptr first_error;

  const auto fail = [&](std::exception_ptr error) noexcept {
    bool expected = false;
    if (!cancelled.compare_exchange_strong(expected, true)) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      first_error = std::move(error);
    }
    try {
      cancel();
    } catch (...) {
      // Preserve the worker failure that caused cancellation.
    }
  };

  const auto run = [&](auto& worker) {
    try {
      worker(cancelled);
    } catch (...) {
      fail(std::current_exception());
    }
  };

  auto producer_future =
      std::async(std::launch::async, [&] { run(producer); });
  auto consumer_future =
      std::async(std::launch::async, [&] { run(consumer); });

  producer_future.get();
  consumer_future.get();

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

} // namespace simaai::neat::pcie::test

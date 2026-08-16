#include "hardware/AsyncTestRunner.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace pcie_test = simaai::neat::pcie::test;

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_consumer_failure_cancels_blocked_producer() {
  std::mutex mutex;
  std::condition_variable cv;
  bool producer_waiting = false;
  bool stop_requested = false;
  std::atomic_int cancel_calls{0};

  try {
    pcie_test::run_async_workers(
        [&] {
          {
            std::lock_guard<std::mutex> lock(mutex);
            stop_requested = true;
          }
          ++cancel_calls;
          cv.notify_all();
        },
        [&](const std::atomic_bool&) {
          std::unique_lock<std::mutex> lock(mutex);
          producer_waiting = true;
          cv.notify_all();
          cv.wait(lock, [&] { return stop_requested; });
        },
        [&](const std::atomic_bool&) {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return producer_waiting; });
          throw std::runtime_error("original consumer failure");
        });
    throw std::runtime_error("async runner did not propagate the worker failure");
  } catch (const std::runtime_error& error) {
    expect(std::string(error.what()) == "original consumer failure",
           "async runner did not preserve the original worker failure");
  }

  expect(stop_requested, "async runner did not request cancellation");
  expect(cancel_calls.load() == 1, "async runner invoked cancellation more than once");
}

void test_producer_failure_cancels_blocked_consumer() {
  std::mutex mutex;
  std::condition_variable cv;
  bool consumer_waiting = false;
  bool stop_requested = false;
  std::atomic_int cancel_calls{0};

  try {
    pcie_test::run_async_workers(
        [&] {
          {
            std::lock_guard<std::mutex> lock(mutex);
            stop_requested = true;
          }
          ++cancel_calls;
          cv.notify_all();
        },
        [&](const std::atomic_bool&) {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return consumer_waiting; });
          throw std::runtime_error("original producer failure");
        },
        [&](const std::atomic_bool&) {
          std::unique_lock<std::mutex> lock(mutex);
          consumer_waiting = true;
          cv.notify_all();
          cv.wait(lock, [&] { return stop_requested; });
        });
    throw std::runtime_error("async runner did not propagate the worker failure");
  } catch (const std::runtime_error& error) {
    expect(std::string(error.what()) == "original producer failure",
           "async runner did not preserve the original worker failure");
  }

  expect(stop_requested, "async runner did not request cancellation");
  expect(cancel_calls.load() == 1, "async runner invoked cancellation more than once");
}

} // namespace

int main() {
  try {
    test_consumer_failure_cancels_blocked_producer();
    test_producer_failure_cancels_blocked_consumer();
    std::cout << "unit_pcie_host_async_test_runner_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "unit_pcie_host_async_test_runner_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}

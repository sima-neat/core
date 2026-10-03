#include "pcie_genai_cancel_watcher.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using simaai::neat::pcie::genai::tools::CancelWatcher;
using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Wait until calls > 0 or 1 s passed. Returns how long it took.
clock_type::duration wait_for_call(const std::atomic<int>& calls) {
  const auto start = clock_type::now();
  while (calls.load() == 0 && clock_type::now() - start < 1s) {
    std::this_thread::sleep_for(1ms);
  }
  return clock_type::now() - start;
}

} // namespace

int main() {
  try {
    // Fires once, soon after the flag turns true, and only once.
    {
      std::atomic<bool> flag{false};
      std::atomic<int> calls{0};
      CancelWatcher watcher([&] { return flag.load(); }, [&] { ++calls; }, 10ms);
      std::this_thread::sleep_for(50ms);
      require(calls.load() == 0 && !watcher.fired(), "must not fire while the flag is false");
      flag.store(true);
      require(wait_for_call(calls) < 200ms, "must fire within a few polls");
      require(calls.load() == 1 && watcher.fired(), "must fire once the flag is true");
      std::this_thread::sleep_for(50ms);
      require(calls.load() == 1, "must fire only once, even if the flag stays true");
    }

    // Review focus 5: an idle watcher must not slow down every prompt.
    {
      std::atomic<int> calls{0};
      const auto start = clock_type::now();
      {
        CancelWatcher watcher([] { return false; }, [&] { ++calls; }, 10s);
      }
      require(clock_type::now() - start < 500ms, "destructor must not wait for the poll");
      require(calls.load() == 0, "a false flag must never fire");
    }

    // A flag that is already true when the watcher starts fires at once.
    {
      std::atomic<int> calls{0};
      CancelWatcher watcher([] { return true; }, [&] { ++calls; }, 10s);
      require(wait_for_call(calls) < 200ms, "an already-true flag must fire at once");
    }

    // Review focus 4: a throwing cancel must not kill the process
    // (an exception escaping a std::thread calls std::terminate).
    {
      CancelWatcher watcher([] { return true; }, [] { throw std::runtime_error("boom"); }, 10ms);
      std::this_thread::sleep_for(50ms);
      require(watcher.fired(), "a throwing cancel still counts as fired");
    }

    std::cout << "[PASS] pcie-genai CancelWatcher\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

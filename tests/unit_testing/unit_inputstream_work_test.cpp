#include "pipeline/internal/InputStreamWork.h"

#include <barrier>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using simaai::neat::pipeline_internal::InputPushGuard;
using simaai::neat::pipeline_internal::InputStreamWork;
using namespace std::chrono_literals;

void check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

void test_idle_and_rejection() {
  InputStreamWork work;
  std::this_thread::sleep_for(30ms);
  check(!work.timed_out(10), "An unused stream timed out");
  {
    InputPushGuard rejected(work);
    check(work.pending() == 1, "Input must be counted before the push");
  }
  check(work.pending() == 0, "Rejected input was not rolled back");
  std::this_thread::sleep_for(30ms);
  check(!work.timed_out(10), "Rejected work armed an idle watchdog");
  work.complete();
  check(work.pending() == 0, "Unsolicited output underflowed the count");
}

void test_stall_and_progress() {
  InputStreamWork work;
  {
    InputPushGuard first(work);
    first.accepted = true;
  }
  std::this_thread::sleep_for(30ms);
  {
    InputPushGuard next(work);
    next.accepted = true;
  }
  check(work.timed_out(10), "Further input postponed a genuine stall");
  { InputPushGuard rejected(work); }
  check(work.timed_out(10), "Rejected input postponed a genuine stall");
  check(work.pending() == 2, "Incorrect outstanding count after rejection");
  check(!work.timed_out(0), "Disabled internal watchdog unexpectedly timed out");
  work.complete();
  // A large interval avoids a scheduler-dependent assertion on the clock boundary.
  check(!work.timed_out(1000), "Output progress did not refresh the deadline");
  work.complete();
  std::this_thread::sleep_for(30ms);
  check(!work.timed_out(10), "A drained stream timed out");
  {
    InputPushGuard resumed(work);
    resumed.accepted = true;
  }
  check(!work.timed_out(1000), "Resume reused the old idle deadline");
  work.complete();
}

void test_output_before_push_returns() {
  InputStreamWork work;
  std::barrier handoff(2);
  std::jthread consumer([&] {
    for (int i = 0; i < 1000; ++i) {
      handoff.arrive_and_wait();
      work.complete();
      handoff.arrive_and_wait();
    }
  });
  for (int i = 0; i < 1000; ++i) {
    InputPushGuard push(work);
    handoff.arrive_and_wait();
    handoff.arrive_and_wait();
    push.accepted = true;
  }
  consumer.join();
  check(work.pending() == 0, "Fast outputs left phantom outstanding work");
  std::this_thread::sleep_for(30ms);
  check(!work.timed_out(10), "Fast-output stream timed out after draining");
}
} // namespace

int main() {
  try {
    test_idle_and_rejection();
    test_stall_and_progress();
    test_output_before_push_returns();
    std::cout << "[OK] unit_inputstream_work_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}

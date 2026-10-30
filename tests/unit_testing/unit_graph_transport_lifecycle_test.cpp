#define SIMA_NEAT_INTERNAL 1
#include "pipeline/runtime/GraphTransportLifecycle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

// This observes the exact production publication helpers without depending on whether an OS
// happens to schedule a notifier in the false-predicate-to-wait window. Dropping the production
// mutex makes store() fail deterministically; the real-CV checks below additionally exercise
// wakeup.
struct LockState {
  bool held = false;
  int acquisitions = 0;
  int notifications = 0;
  int destroys = 0;
  bool stop = false;
  std::vector<std::string> events;
};

struct CheckedMutex {
  LockState& state;
  void lock() {
    check(!state.held, "construction or an external callback reentered the transport mutex");
    state.held = true;
    ++state.acquisitions;
    state.events.emplace_back("lock");
  }
  void unlock() {
    check(state.held, "transport mutex was not held");
    state.events.emplace_back("unlock");
    state.held = false;
  }
};

struct CheckedFlag {
  LockState& state;
  bool value = false;
  void store(bool next, std::memory_order order) {
    check(state.held, "lifecycle publication bypassed the waiter mutex");
    check(order == std::memory_order_release, "fast-path readers lost release publication");
    value = next;
    state.events.emplace_back("publish");
  }
};

struct CheckedCondition {
  LockState& state;
  void notify_all() {
    check(!state.held, "notification unnecessarily retains the waiter mutex");
    ++state.notifications;
    state.events.emplace_back("notify");
  }
};

struct CheckedTransport {
  explicit CheckedTransport(LockState& state)
      : mu{state}, completion_forwarded{state}, built{state}, cv{state} {}
  CheckedMutex mu;
  CheckedFlag completion_forwarded;
  CheckedFlag built;
  CheckedCondition cv;
};

struct ChildOwner {
  explicit ChildOwner(LockState& state) : state(state) {}
  ~ChildOwner() {
    // A fatal check is deliberate: a child destructor must not run in a lifecycle critical
    // section, including the exception/stop paths where callbacks may reenter the graph.
    if (state.held) {
      std::terminate();
    }
    ++state.destroys;
  }
  LockState& state;
};

struct CheckedPipeline {
  explicit CheckedPipeline(LockState& state) : transport(state) {}
  CheckedTransport transport;
  std::shared_ptr<ChildOwner> run_core;
};

void publication_uses_waiter_mutex() {
  LockState state;
  CheckedTransport transport(state);
  simaai::neat::runtime::publish_graph_transport_completion(transport);
  check(transport.completion_forwarded.value, "completion was not published");
  check(state.events == std::vector<std::string>{"lock", "publish", "unlock", "notify"},
        "completion publication/notification ordering changed");
}

void external_state_notification_synchronizes_with_waiter() {
  LockState state;
  CheckedTransport transport(state);
  // Stop and public ingress close are already published by their own authority. The transport
  // must handshake after that publication, not invent a second stop/closed flag.
  state.stop = true;
  simaai::neat::runtime::notify_graph_transport_external_state(transport);
  check(state.stop, "notification changed the externally owned predicate");
  check(state.events == std::vector<std::string>{"lock", "unlock", "notify"},
        "external-state notification bypassed the waiter mutex handshake");
}

void stop_during_construction_retains_child_owner() {
  LockState state;
  CheckedPipeline pipeline(state);
  std::weak_ptr<ChildOwner> observed;
  simaai::neat::runtime::construct_graph_source_pipeline(pipeline, [&] {
    check(!state.held, "source construction retained the transport mutex");
    // Model a construction-time streaming callback requesting stop. The real stop path uses
    // this same handshake; retaining the construction lock would fail on reentry here.
    state.stop = true;
    simaai::neat::runtime::notify_graph_transport_external_state(pipeline.transport);
    auto child = std::make_shared<ChildOwner>(state);
    observed = child;
    return child;
  });
  check(state.stop, "construction-time stop was lost");
  check(pipeline.transport.built.value, "constructed child was not published for teardown");
  check(pipeline.run_core && !observed.expired() && state.destroys == 0,
        "stop discarded an owned child instead of handing it to graph teardown");
  check(state.events == std::vector<std::string>{"lock", "unlock", "notify", "lock", "publish",
                                                 "unlock", "notify"},
        "construction callback/publication lock ordering changed");
  pipeline.run_core.reset();
  check(observed.expired() && state.destroys == 1,
        "normal teardown did not release the constructed child exactly once");
}

void failed_construction_does_not_publish_ready() {
  LockState state;
  CheckedPipeline pipeline(state);
  bool caught = false;
  try {
    simaai::neat::runtime::construct_graph_source_pipeline(
        pipeline, [&]() -> std::shared_ptr<ChildOwner> {
          auto child = std::make_shared<ChildOwner>(state);
          throw std::runtime_error("original construction failure");
        });
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "original construction failure";
  }
  check(caught, "construction exception was replaced");
  check(!pipeline.run_core && !pipeline.transport.built.value && state.notifications == 0,
        "failed construction published an unowned child as ready");
  check(state.destroys == 1, "failed construction lost its partial child ownership");
  // Construction failure is not a new sticky lifecycle state. In particular, an unsuccessful
  // speculative prebuild must not prevent a later independent real-input construction attempt.
  simaai::neat::runtime::construct_graph_source_pipeline(
      pipeline, [&] { return std::make_shared<ChildOwner>(state); });
  check(pipeline.run_core && pipeline.transport.built.value && state.notifications == 1,
        "a failed construction prevented a later successful independent attempt");
  pipeline.run_core.reset();
  check(state.destroys == 2, "retry construction did not preserve single ownership");
}

struct RealTransport {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> completion_forwarded{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> abort_test{false};
};

void waiting_threads_observe_completion_and_stop(bool stop) {
  RealTransport transport;
  constexpr int kWaiters = 4;
  std::latch checked_false(kWaiters);
  std::mutex finished_mu;
  std::condition_variable finished_cv;
  int finished = 0;
  std::vector<std::thread> waiters;
  for (int i = 0; i < kWaiters; ++i) {
    waiters.emplace_back([&] {
      bool first = true;
      {
        std::unique_lock lock(transport.mu);
        // No timed predicate recheck: a missing notification must not turn into success when
        // a waiter reaches its own timeout. Only the separate observer owns the watchdog.
        transport.cv.wait(lock, [&] {
          const bool terminal = transport.completion_forwarded.load(std::memory_order_acquire) ||
                                transport.stop.load(std::memory_order_acquire) ||
                                transport.abort_test.load(std::memory_order_acquire);
          if (first) {
            first = false;
            checked_false.count_down();
          }
          return terminal;
        });
      }
      if (!transport.abort_test.load(std::memory_order_acquire)) {
        {
          std::lock_guard lock(finished_mu);
          ++finished;
        }
        finished_cv.notify_all();
      }
    });
  }
  checked_false.wait();
  if (stop) {
    transport.stop.store(true, std::memory_order_release);
    simaai::neat::runtime::notify_graph_transport_external_state(transport);
  } else {
    simaai::neat::runtime::publish_graph_transport_completion(transport);
  }
  bool completed = false;
  {
    std::unique_lock lock(finished_mu);
    completed =
        finished_cv.wait_for(lock, std::chrono::seconds(2), [&] { return finished == kWaiters; });
  }
  if (!completed) {
    // Test-only cleanup cannot satisfy the recorded completion assertion. Publish abort under
    // the actual waiter mutex, so a deliberately broken helper cannot strand the test threads.
    {
      std::lock_guard lock(transport.mu);
      transport.abort_test.store(true, std::memory_order_release);
    }
    transport.cv.notify_all();
  }
  for (auto& waiter : waiters) {
    waiter.join();
  }
  check(completed, "terminal publication failed to wake every transport waiter");
}

void completion_before_wait_remains_observable() {
  RealTransport transport;
  simaai::neat::runtime::publish_graph_transport_completion(transport);
  std::unique_lock lock(transport.mu);
  check(transport.cv.wait_for(
            lock, std::chrono::milliseconds(0),
            [&] { return transport.completion_forwarded.load(std::memory_order_acquire); }),
        "an earlier notification lost the persistent completion predicate");
}

} // namespace

int main() {
  try {
    publication_uses_waiter_mutex();
    external_state_notification_synchronizes_with_waiter();
    stop_during_construction_retains_child_owner();
    failed_construction_does_not_publish_ready();
    waiting_threads_observe_completion_and_stop(false);
    waiting_threads_observe_completion_and_stop(true);
    completion_before_wait_remains_observable();
    std::cout << "[OK] unit_graph_transport_lifecycle_test passed (7 contracts)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

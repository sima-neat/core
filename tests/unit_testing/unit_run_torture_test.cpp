// H0's torture list (H0-input-output.md 2.6 and plan C4) on Core's public API, on the backend
// SIMA_NEAT_BACKEND names (dataflow: RunBackend::Dataflow, requested explicitly; otherwise
// RunBackend::GStreamer). The same executable is registered once per backend, and it is the
// subject of the sanitizer runs (ASan+UBSan, TSan) of part C4.
//
// Cases, each repeated SIMA_TORTURE_REPEAT times (default 3) for more interleavings:
// 1. stop during a blocked push: the push returns (false, or a "stopped"/"closed" error), stop is
//    bounded, nothing is accepted afterwards;
// 2. stop during a pending pull: the pull returns within 1 s (Dataflow: Closed);
// 3. close_input with queued messages: every queued message is pulled, in order and intact, then
//    the output is Closed;
// 4. two pushers: two threads push the same Run, serialized by an application mutex, because
//    Core's Run contract forbids unsynchronized concurrent pushes (Run.h, "Thread safety"); a
//    third thread pulls; every frame arrives exactly once, each pusher's frames in order;
// 5. a lease outliving the Run: tensors pulled through the old and the new verbs stay readable
//    after the Run is closed and destroyed, also from another thread, and are released after.
// A wait that does not end is reported and the process exits at once (std::_Exit), so a hang in
// one backend shows as a failure, not a ctest timeout.

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

bool on_dataflow() {
  const char* env = std::getenv("SIMA_NEAT_BACKEND");
  return env != nullptr && std::string(env) == "dataflow";
}

const char* backend_name() {
  return on_dataflow() ? "dataflow" : "gstreamer";
}

int repeats() {
  const char* env = std::getenv("SIMA_TORTURE_REPEAT");
  const int n = env ? std::atoi(env) : 3;
  return n > 0 ? n : 3;
}

std::int64_t ms_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
}

// An RGB tensor whose bytes encode `tag` and their position.
Tensor rgb(std::uint8_t tag, int w = 32, int h = 24) {
  Tensor t = make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return t;
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// The linear Input -> Output graph, seeded (it builds on both backends), RGB 32x24 declared by
// the seed, Output max_buffers = `depth`, RunOptions::queue_depth = `depth`, Block. `max_bytes`
// and `block` give GStreamer's appsrc a bound so a blocking push can actually block there (the
// Dataflow backend ignores both with a one-time warning: its link bounds the queue, at
// queue_depth + max_buffers = 2 x `depth`).
Run make_run(int depth, bool bounded_appsrc = false) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  if (bounded_appsrc) {
    in.max_bytes = 32U * 24U * 3U;
    in.block = true;
  }
  OutputOptions out;
  out.max_buffers = depth;
  out.drop = false;
  out.sync = false;
  Graph g;
  g.add(nodes::Input(in));
  g.add(nodes::Output(out));
  RunOptions opt;
  opt.backend = on_dataflow() ? RunBackend::Dataflow : RunBackend::GStreamer;
  opt.queue_depth = depth;
  opt.overflow_policy = OverflowPolicy::Block;
  opt.startup_preflight = false;
  return g.build(TensorList{rgb(0)}, opt);
}

// Run `fn` on its own thread and wait at most `budget`; past it, report and exit at once.
template <class T>
T bounded(const std::string& what, std::chrono::milliseconds budget, std::function<T()> fn) {
  struct State {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::optional<T> value;
    std::exception_ptr error;
  };
  auto state = std::make_shared<State>();
  std::thread([state, fn = std::move(fn)] {
    std::optional<T> value;
    std::exception_ptr error;
    try {
      value.emplace(fn());
    } catch (...) {
      error = std::current_exception();
    }
    std::lock_guard<std::mutex> lock(state->mu);
    state->value = std::move(value);
    state->error = error;
    state->done = true;
    state->cv.notify_all();
  }).detach();
  std::unique_lock<std::mutex> lock(state->mu);
  if (!state->cv.wait_for(lock, budget, [&] { return state->done; })) {
    std::cerr << "[FAIL] unit_run_torture_test (" << backend_name() << "): " << what
              << " did not return within " << budget.count() << " ms\n";
    std::_Exit(1);
  }
  if (state->error) {
    std::rethrow_exception(state->error);
  }
  return std::move(*state->value);
}

// A push that a stop or close_input ended: false, or an error that says so.
struct PushOutcome {
  bool accepted = false;
  std::string error;
};

PushOutcome push_outcome(Run& run, const Tensor& t) {
  PushOutcome out;
  try {
    out.accepted = run.push(TensorList{t});
  } catch (const std::exception& e) {
    out.error = e.what();
  }
  return out;
}

bool ended_by_teardown(const PushOutcome& o) {
  if (o.accepted) {
    return false;
  }
  return o.error.empty() || o.error.find("stop") != std::string::npos ||
         o.error.find("closed") != std::string::npos || o.error.find("EOS") != std::string::npos;
}

// 1. stop during a blocked push.
void stop_during_blocked_push(int round) {
  Run run = make_run(1, /*bounded_appsrc=*/true);
  // Push until one push does not return within 300 ms: that push is blocked on a full queue.
  struct Blocked {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    PushOutcome outcome;
  };
  std::shared_ptr<Blocked> blocked;
  constexpr int kMaxFrames = 64;
  int frames = 0;
  for (; frames < kMaxFrames && !blocked; ++frames) {
    auto attempt = std::make_shared<Blocked>();
    const Tensor t = rgb(static_cast<std::uint8_t>(frames + 1));
    std::thread([attempt, &run, t] {
      PushOutcome o = push_outcome(run, t);
      std::lock_guard<std::mutex> lock(attempt->mu);
      attempt->outcome = std::move(o);
      attempt->done = true;
      attempt->cv.notify_all();
    }).detach();
    std::unique_lock<std::mutex> lock(attempt->mu);
    if (!attempt->cv.wait_for(lock, 300ms, [&] { return attempt->done; })) {
      blocked = attempt;
      break;
    }
    require(attempt->outcome.accepted,
            "blocked push: a push before the queue filled was refused: " + attempt->outcome.error);
  }
  require(blocked != nullptr, std::string("blocked push: no push blocked within ") +
                                  std::to_string(kMaxFrames) + " frames");
  if (round == 0) {
    std::cout << "[INFO] " << backend_name() << ": a push blocked after " << frames
              << " accepted frames\n";
  }
  const auto t0 = Clock::now();
  run.stop();
  const auto stop_ms = ms_since(t0);
  const auto budget = on_dataflow() ? 1000ms : 5000ms;
  {
    std::unique_lock<std::mutex> lock(blocked->mu);
    if (!blocked->cv.wait_for(lock, budget, [&] { return blocked->done; })) {
      std::cerr << "[FAIL] unit_run_torture_test (" << backend_name()
                << "): the blocked push did not return within " << budget.count()
                << " ms of stop\n";
      std::_Exit(1);
    }
    require(ended_by_teardown(blocked->outcome),
            "blocked push: after stop the push must not be accepted; error='" +
                blocked->outcome.error + "'");
  }
  require(stop_ms < (on_dataflow() ? 1000 : 5000),
          "blocked push: stop took " + std::to_string(stop_ms) + " ms");
  const PushOutcome after = push_outcome(run, rgb(0x77));
  require(ended_by_teardown(after), "blocked push: a push after stop was accepted");
  bounded<int>("close after a stop that ended a blocked push", 5000ms, [&] {
    run.close();
    return 0;
  });
}

// 2. stop during a pending pull.
void stop_during_pending_pull(int) {
  Run run = make_run(4);
  // One round trip first, so the GStreamer pipeline is live (its pre-push pull is slow, c2.md
  // section 4 row 7).
  require(run.push(TensorList{rgb(1)}), "pending pull: warm-up push");
  require(!run.pull_tensors(2000).empty(), "pending pull: warm-up pull");
  struct Pending {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    PullStatus status = PullStatus::Ok;
    std::string message;
  };
  auto pending = std::make_shared<Pending>();
  std::thread([pending, &run] {
    Sample s;
    PullError err;
    PullStatus st = PullStatus::Error;
    std::string msg;
    try {
      st = run.pull(5000, s, &err);
      msg = err.message;
    } catch (const std::exception& e) {
      msg = e.what();
    }
    std::lock_guard<std::mutex> lock(pending->mu);
    pending->status = st;
    pending->message = msg;
    pending->done = true;
    pending->cv.notify_all();
  }).detach();
  std::this_thread::sleep_for(100ms);
  {
    std::lock_guard<std::mutex> lock(pending->mu);
    require(!pending->done, "pending pull: the pull returned before stop");
  }
  const auto t0 = Clock::now();
  run.stop();
  std::unique_lock<std::mutex> lock(pending->mu);
  if (!pending->cv.wait_for(lock, 1000ms, [&] { return pending->done; })) {
    std::cerr << "[FAIL] unit_run_torture_test (" << backend_name()
              << "): a pending pull did not return within 1 s of stop (stop took " << ms_since(t0)
              << " ms)\n";
    std::_Exit(1);
  }
  require(pending->status != PullStatus::Ok, "pending pull: no output was pushed");
  if (on_dataflow()) {
    require(pending->status == PullStatus::Closed,
            "pending pull: Dataflow reports Closed after stop; got " +
                std::to_string(static_cast<int>(pending->status)) + " " + pending->message);
  }
  lock.unlock();
  run.close();
}

// 3. close_input with queued messages.
void close_input_with_queued_messages(int) {
  constexpr int kQueued = 4;
  Run run = make_run(kQueued);
  std::vector<std::vector<std::uint8_t>> expected;
  for (int i = 0; i < kQueued; ++i) {
    Sample s;
    s.kind = SampleKind::Tensor;
    s.tensor = rgb(static_cast<std::uint8_t>(0x20 + i));
    s.frame_id = 100 + i;
    expected.push_back(bytes_of(*s.tensor));
    require(run.push(Sample{s}), "queued: push " + std::to_string(i));
  }
  run.close_input();
  for (int i = 0; i < kQueued; ++i) {
    Sample s;
    PullError err;
    const PullStatus st = run.pull(3000, s, &err);
    require(st == PullStatus::Ok, "queued: message " + std::to_string(i) +
                                      " pulled after close_input; status " +
                                      std::to_string(static_cast<int>(st)) + " " + err.message);
    require(s.frame_id == 100 + i, "queued: in order at " + std::to_string(i));
    const TensorList t = tensors_from_sample(s);
    require(t.size() == 1U && bytes_of(t.front()) == expected[static_cast<std::size_t>(i)],
            "queued: intact at " + std::to_string(i));
  }
  Sample s;
  PullError err;
  const PullStatus st = run.pull(3000, s, &err);
  require(st == PullStatus::Closed, "queued: Closed after the queue drained; got " +
                                        std::to_string(static_cast<int>(st)) + " " + err.message);
  run.close();
}

// 4. two pushers, serialized by the application (Core's contract), one puller.
void two_pushers(int) {
  constexpr int kPerPusher = 48;
  Run run = make_run(4);
  std::mutex push_mu;
  std::atomic<bool> failed{false};
  std::string failure;
  std::mutex failure_mu;
  auto pusher = [&](int id) {
    try {
      for (int i = 0; i < kPerPusher; ++i) {
        Sample s;
        s.kind = SampleKind::Tensor;
        s.tensor = rgb(static_cast<std::uint8_t>(id * 64 + i));
        s.frame_id = id * 1000 + i;
        s.stream_id = id == 0 ? "left" : "right";
        bool ok = false;
        {
          std::lock_guard<std::mutex> serialize(push_mu);
          ok = run.push(Sample{s});
        }
        if (!ok) {
          throw std::runtime_error("push refused at " + std::to_string(s.frame_id));
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(failure_mu);
      failure = e.what();
      failed = true;
    }
  };
  std::map<std::string, std::vector<std::int64_t>> seen;
  std::thread a(pusher, 0);
  std::thread b(pusher, 1);
  try {
    for (int n = 0; n < 2 * kPerPusher && !failed.load(); ++n) {
      Sample s;
      PullError err;
      const PullStatus st = run.pull(5000, s, &err);
      require(st == PullStatus::Ok, "two pushers: pull " + std::to_string(n) + " status " +
                                        std::to_string(static_cast<int>(st)) + " " + err.message);
      const TensorList t = tensors_from_sample(s);
      const int id = s.stream_id == "left" ? 0 : 1;
      const auto index = s.frame_id - id * 1000;
      require(t.size() == 1U &&
                  bytes_of(t.front()) == bytes_of(rgb(static_cast<std::uint8_t>(id * 64 + index))),
              "two pushers: frame " + std::to_string(s.frame_id) + " intact");
      seen[s.stream_id].push_back(s.frame_id);
    }
  } catch (...) {
    run.stop(); // releases a pusher parked on the full queue, so the joins below end
    a.join();
    b.join();
    throw;
  }
  if (failed.load()) {
    run.stop(); // the other pusher may be parked with nobody pulling
  }
  a.join();
  b.join();
  require(!failed.load(), "two pushers: " + failure);
  for (const char* stream : {"left", "right"}) {
    const auto& ids = seen[stream];
    require(ids.size() == static_cast<std::size_t>(kPerPusher),
            std::string("two pushers: every frame of ") + stream + " arrives exactly once");
    for (std::size_t i = 1; i < ids.size(); ++i) {
      require(ids[i] == ids[i - 1] + 1, std::string("two pushers: ") + stream + " in order");
    }
  }
  run.close();
}

// 5. a lease (pulled tensor) outliving the Run.
void lease_outlives_run(int) {
  std::vector<Tensor> old_verb;
  std::vector<Tensor> new_verb;
  std::vector<std::vector<std::uint8_t>> expected;
  {
    Run run = make_run(8);
    for (int i = 0; i < 6; ++i) {
      const Tensor t = rgb(static_cast<std::uint8_t>(0x40 + i));
      expected.push_back(bytes_of(t));
      require(run.push(TensorList{t}), "lease: push " + std::to_string(i));
      if (i % 2 == 0) {
        const TensorList out = run.pull_tensors(3000);
        require(out.size() == 1U, "lease: old pull " + std::to_string(i));
        old_verb.push_back(out.front());
      } else {
        Result<TensorList> r = run.pull(3000ms);
        require(r.ok() && r->size() == 1U,
                "lease: new pull " + std::to_string(i) + ": " + r.error().message);
        new_verb.push_back(r->front());
      }
    }
    run.close();
  } // the Run is destroyed here; the tensors are still held
  auto check = [&] {
    for (std::size_t i = 0; i < old_verb.size(); ++i) {
      require(bytes_of(old_verb[i]) == expected[2 * i], "lease: old-verb tensor readable");
    }
    for (std::size_t i = 0; i < new_verb.size(); ++i) {
      require(bytes_of(new_verb[i]) == expected[2 * i + 1], "lease: new-verb tensor readable");
    }
  };
  check();
  bounded<int>("reading leases from another thread", 5000ms, [&] {
    check();
    return 0;
  });
  // Release from another thread than the one that pulled.
  bounded<int>("releasing leases from another thread", 5000ms, [&] {
    old_verb.clear();
    new_verb.clear();
    return 0;
  });
}

} // namespace

RUN_TEST("unit_run_torture_test", ([] {
           const int n = repeats();
           std::cout << "[INFO] backend=" << backend_name() << " repeats=" << n << "\n";
           for (int round = 0; round < n; ++round) {
             stop_during_blocked_push(round);
             stop_during_pending_pull(round);
             close_input_with_queued_messages(round);
             two_pushers(round);
             lease_outlives_run(round);
           }
         }));

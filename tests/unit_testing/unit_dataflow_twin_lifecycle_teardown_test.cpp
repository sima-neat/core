// Dataflow twin of unit_run_lifecycle_teardown_test (H0 C4, plan D12), on the public API only:
// every Run here requests RunBackend::Dataflow explicitly; no internal header, no
// pipeline_handle(), no appsink pad probe, no RunCore field. It also carries the Dataflow twin
// of unit_public_output_appsink_contract_test's close_input contract (that test drives
// GStreamer's InputStream and GraphSinkQueue directly and never builds a Run).
//
// The GStreamer test's blocks after its first one need GStreamer internals: a blocking pad probe
// on the appsink to wedge stream teardown, RunCore::stream_close_state, the input thread parked on
// latency_mu, child pipelines of a composite graph, RunCore::start with injected failures. What
// they protect, stated for the public API, and kept here:
// - concurrent try_push and pull, then close_input, stop and a second stop, bounded (< 5 s), and
//   try_push refused after stop (the GStreamer test's first block, unchanged);
// - close_input cancels a parked push (< 1 s, false), keeps what was queued pullable, then the
//   output is Closed and the Run stops itself;
// - stop unblocks a pending pull within 1 s (Closed) and a parked push (false);
// - several threads calling stop at once: one owner, every call returns, bounded;
// - teardown with application-held leases (pulled tensors): close is bounded and the leases stay
//   readable after the Run is gone (the GStreamer test's "retained until the blocked work exits");
// - state and measurement reads racing stop: MeasureScope::stop and the Run's state queries from
//   other threads while stop runs (the GStreamer test's measurement-reads-safe block).

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using Clock = std::chrono::steady_clock;

std::int64_t ms_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
}

Tensor rgb(std::uint8_t tag, int w = 64, int h = 48) {
  return make_color_tensor(w, h, ImageSpec::PixelFormat::RGB, tag);
}

// sima_test::make_async_rgb_run's graph (the GStreamer test's), on Dataflow.
Run make_rgb_run(int queue_depth, int max_buffers = 128) {
  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Image;
  src_opt.format = FormatTag::RGB;
  src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(max_buffers)));
  RunOptions run_opt;
  run_opt.backend = RunBackend::Dataflow;
  run_opt.queue_depth = queue_depth;
  run_opt.overflow_policy = OverflowPolicy::Block;
  return graph.build(TensorList{rgb(0x3A)}, run_opt);
}

// make_rgb_run(1, 1)'s link holds queue_depth 1 + max_buffers 1 = 2 messages: two blocking
// pushes fill it, the next one parks.
constexpr int kSmallLinkDepth = 2;

bool fill_small_link(Run& run) {
  bool ok = true;
  for (int i = 0; i < kSmallLinkDepth; ++i) {
    ok = run.push(TensorList{rgb(static_cast<std::uint8_t>(1 + i))}) && ok;
  }
  return ok;
}

bool push_refused(Run& run, std::uint8_t tag) {
  try {
    return !run.try_push(TensorList{rgb(tag)});
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    return msg.find("stopped") != std::string::npos || msg.find("closed") != std::string::npos;
  }
}

// The GStreamer test's first block, on Dataflow.
void concurrent_traffic_then_bounded_teardown() {
  Run run = make_rgb_run(128);
  std::atomic<bool> keep_running{true};
  std::atomic<int> pushes{0};
  std::atomic<int> pulls{0};
  std::mutex err_mu;
  std::string producer_err;
  std::string consumer_err;

  std::thread producer([&] {
    try {
      while (keep_running.load(std::memory_order_relaxed)) {
        if (run.try_push(TensorList{rgb(0x3A)})) {
          pushes.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(err_mu);
      producer_err = e.what();
    }
  });
  std::thread consumer([&] {
    try {
      while (keep_running.load(std::memory_order_relaxed)) {
        if (run.pull(20).has_value()) {
          pulls.fetch_add(1, std::memory_order_relaxed);
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(err_mu);
      consumer_err = e.what();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const auto t0 = Clock::now();
  run.close_input();
  run.stop();
  run.stop(); // idempotent
  const auto stop_ms = ms_since(t0);
  keep_running.store(false, std::memory_order_relaxed);
  producer.join();
  consumer.join();
  {
    std::lock_guard<std::mutex> lock(err_mu);
    require(producer_err.empty(), "producer failed: " + producer_err);
    require(consumer_err.empty(), "consumer failed: " + consumer_err);
  }
  require(stop_ms < 5000, "close_input + stop + stop exceeded 5 s: " + std::to_string(stop_ms));
  require(pushes.load() > 0 && pulls.load() > 0, "traffic flowed before teardown");
  require(push_refused(run, 1), "try_push must fail after stop()");
  run.close();
}

// close_input while a push is parked on the full link (unit_public_output_appsink_contract_test's
// close_public_input contract, on the public Run).
void close_input_cancels_a_parked_push() {
  Run run = make_rgb_run(1, 1);
  require(fill_small_link(run), "two pushes fill the link");
  auto parked = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(9)}); });
  require(parked.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
          "the third push parks on the full link");
  const auto t0 = Clock::now();
  run.close_input();
  require(parked.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "close_input wakes the parked push within 1 s");
  require(!parked.get(), "the cancelled push returns false");
  require(ms_since(t0) < 1000, "close_input does not wait behind the parked push");
  for (int i = 0; i < kSmallLinkDepth; ++i) {
    const TensorList queued = run.pull_tensors(1000);
    require(queued.size() == 1U &&
                queued.front().copy_dense_bytes_tight() ==
                    rgb(static_cast<std::uint8_t>(1 + i)).copy_dense_bytes_tight(),
            "what was queued before close_input stays pullable");
  }
  Sample s;
  PullError err;
  require(run.pull(1000, s, &err) == PullStatus::Closed, "then the output is Closed");
  require(!run.running(), "the drained, closed Run stops itself");
  require(push_refused(run, 3), "a push after close_input is refused");
  run.close();
}

// stop releases both kinds of waiter within 1 s.
void stop_unblocks_pending_pull_and_parked_push() {
  {
    Run run = make_rgb_run(4);
    auto pending = std::async(std::launch::async, [&] {
      Sample s;
      PullError err;
      return run.pull(5000, s, &err);
    });
    require(pending.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout,
            "the pull waits on an empty link");
    const auto t0 = Clock::now();
    run.stop();
    require(pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
            "a pending pull(5000) unblocks within 1 s of stop");
    require(pending.get() == PullStatus::Closed, "the pending pull reports Closed");
    require(ms_since(t0) < 1000, "stop returned within 1 s");
    require(!run.running(), "not running after stop");
    run.close();
  }
  {
    Run run = make_rgb_run(1, 1);
    require(fill_small_link(run), "fill the link");
    auto parked = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(9)}); });
    require(parked.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
            "the push parks on the full link");
    const auto t0 = Clock::now();
    run.stop();
    require(parked.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
            "a parked push unblocks within 1 s of stop");
    require(!parked.get(), "the parked push returns false after stop");
    require(ms_since(t0) < 1000, "stop returned within 1 s");
    run.close();
  }
}

// Several threads stop the same Run at once while a push is parked: one owner does the work,
// every call returns, the parked push is released.
void concurrent_stop_has_one_owner() {
  Run run = make_rgb_run(1, 1);
  require(fill_small_link(run), "fill the link");
  auto parked = std::async(std::launch::async, [&] { return run.push(TensorList{rgb(9)}); });
  require(parked.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
          "the push parks on the full link");
  std::atomic<bool> go{false};
  std::vector<std::thread> stoppers;
  const auto t0 = Clock::now();
  for (int i = 0; i < 4; ++i) {
    stoppers.emplace_back([&] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      run.stop();
    });
  }
  go.store(true, std::memory_order_release);
  for (auto& t : stoppers) {
    t.join();
  }
  require(ms_since(t0) < 3000, "four concurrent stops return within 3 s");
  require(parked.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !parked.get(),
          "the parked push is released with false");
  require(!run.running(), "stopped");
  run.close();
  run.close(); // idempotent
}

// Pulled tensors are leases on pool slots: teardown does not wait for them, and they stay
// readable after the Run is gone.
void teardown_with_held_leases() {
  std::vector<Tensor> kept;
  std::vector<std::vector<std::uint8_t>> expected;
  {
    Run run = make_rgb_run(8, 8);
    for (std::uint8_t i = 0; i < 3; ++i) {
      const Tensor t = rgb(static_cast<std::uint8_t>(0x50 + i));
      expected.push_back(t.copy_dense_bytes_tight());
      require(run.push(TensorList{t}), "push");
      const TensorList out = run.pull_tensors(2000);
      require(out.size() == 1U, "pull");
      kept.push_back(out.front());
    }
    const auto t0 = Clock::now();
    run.close();
    require(ms_since(t0) < 1000, "close is bounded while the application holds leases");
  }
  for (std::size_t i = 0; i < kept.size(); ++i) {
    require(kept[i].copy_dense_bytes_tight() == expected[i],
            "a lease stays readable after its Run is gone (" + std::to_string(i) + ")");
  }
  kept.clear(); // the slots return to the retired pool, which goes with its last lease
}

// State and measurement reads from other threads while stop runs.
void reads_race_stop() {
  Run run = make_rgb_run(8, 8);
  for (std::uint8_t i = 0; i < 4; ++i) {
    require(run.push(TensorList{rgb(i)}), "push");
  }
  MeasureScope measurement = run.start_measurement();
  std::atomic<bool> start{false};
  std::atomic<bool> keep_reading{true};
  std::atomic<std::uint64_t> reads{0};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (keep_reading.load(std::memory_order_acquire)) {
        (void)run.running();
        (void)run.can_push();
        (void)run.can_pull();
        (void)run.last_error();
        (void)run.input_names();
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  std::exception_ptr measurement_error;
  MeasureReport report;
  std::thread stop_measurement([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    try {
      report = measurement.stop();
    } catch (...) {
      measurement_error = std::current_exception();
    }
  });
  start.store(true, std::memory_order_release);
  run.stop();
  stop_measurement.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  keep_reading.store(false, std::memory_order_release);
  for (auto& r : readers) {
    r.join();
  }
  if (measurement_error) {
    std::rethrow_exception(measurement_error);
  }
  require(reads.load() > 0, "the readers ran");
  require(!run.running(), "stopped");
  run.close();
}

} // namespace

RUN_TEST("unit_dataflow_twin_lifecycle_teardown_test", ([] {
           concurrent_traffic_then_bounded_teardown();
           close_input_cancels_a_parked_push();
           stop_unblocks_pending_pull_and_parked_push();
           concurrent_stop_has_one_owner();
           teardown_with_held_leases();
           reads_race_stop();
         }));

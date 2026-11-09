// RunPreset and an explicit overflow policy, on the backend SIMA_NEAT_BACKEND names (dataflow:
// RunBackend::Dataflow, requested explicitly; otherwise RunBackend::GStreamer). The same
// executable is registered once per backend (H0 C4b).
//
// Core resolves RunOptions by the path the graph takes. A plain linear Run is one pipeline segment
// whose options are resolve_build_opt's: the preset fills queue_depth and overflow_policy where
// they equal their RunOptions{} defaults, so under RunPreset::Realtime an explicit Block is
// KeepLatest and try_push never refuses. A connected or named graph is a graph Run that keeps the
// RunOptions as given: under Realtime an explicit Block refuses a try_push when the queue is full,
// and a blocking push waits for room (pyneat's test_blocking_push_gil builds this graph). Both
// backends must agree on every case below.
//
// Cases: connected Realtime + explicit Block (try_push refuses, no drop; a blocking push waits for
// a pull); connected Realtime alone (the same: the preset does not reach a graph Run's ingress);
// linear Realtime + explicit Block and linear Realtime alone (KeepLatest: try_push never refuses);
// linear Balanced + Block (the control: try_push refuses when full).

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

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

Tensor rgb(std::uint8_t tag) {
  return make_color_tensor(32, 24, ImageSpec::PixelFormat::RGB, tag);
}

InputOptions declared_rgb() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.width = 32;
  in.height = 24;
  in.depth = 3;
  in.max_width = 32;
  in.max_height = 24;
  in.max_depth = 3;
  in.memory_policy = InputMemoryPolicy::SystemMemory;
  return in;
}

// RunOptions with the preset, queue_depth 1, and the overflow policy when one is given (left at
// its RunOptions{} default otherwise).
RunOptions options(RunPreset preset, std::optional<OverflowPolicy> policy) {
  RunOptions opt;
  opt.backend = on_dataflow() ? RunBackend::Dataflow : RunBackend::GStreamer;
  opt.preset = preset;
  opt.queue_depth = 1;
  if (policy) {
    opt.overflow_policy = *policy;
  }
  opt.output_memory = OutputMemory::Owned;
  opt.startup_preflight = false;
  return opt;
}

// pyneat's test_blocking_push_gil graph: Graph::connect of an Input("input") fragment and an
// Output("output", EveryFrame(1)) fragment, built without a seed.
Run connected_run(const RunOptions& opt) {
  Graph source("input");
  source.add(nodes::Input("input", declared_rgb()));
  Graph sink("output");
  sink.add(nodes::Output("output", OutputOptions::EveryFrame(1)));
  Graph g("preset_overflow");
  g.connect(source, sink);
  return g.build(opt);
}

Run linear_run(const RunOptions& opt) {
  Graph g;
  g.add(nodes::Input(declared_rgb()));
  g.add(nodes::Output(OutputOptions::EveryFrame(1)));
  return g.build(TensorList{rgb(1)}, opt);
}

struct Saturation {
  int accepted = 0;
  bool refused = false; // 10 consecutive refusals (10 ms apart) within 5 s, as pyneat's test
  std::uint64_t inputs_dropped = 0;
};

Saturation saturate(Run& run, bool connected) {
  Saturation s;
  MeasureScope scope = run.start_measurement();
  int consecutive = 0;
  const auto deadline = Clock::now() + 5s;
  while (Clock::now() < deadline) {
    const Tensor t = rgb(static_cast<std::uint8_t>(s.accepted));
    const bool ok = connected ? run.try_push("input", TensorList{t}) : run.try_push(TensorList{t});
    if (ok) {
      ++s.accepted;
      consecutive = 0;
      continue;
    }
    if (++consecutive == 10) {
      s.refused = true;
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  s.inputs_dropped = scope.stop().counters.inputs_dropped;
  return s;
}

// On stderr: a GStreamer process's stdout can lose its tail at exit (Core's stdio filter thread).
void report(const std::string& what, const Saturation& s) {
  std::cerr << "[INFO] " << backend_name() << " " << what << ": accepted " << s.accepted
            << (s.refused ? ", then refused" : ", never refused") << ", inputs_dropped "
            << s.inputs_dropped << std::endl;
}

// Connected graph, Realtime + explicit Block: try_push refuses when the queue is full and drops
// nothing; a blocking push then waits until a pull makes room.
void connected_realtime_explicit_block() {
  Run run = connected_run(options(RunPreset::Realtime, OverflowPolicy::Block));
  const Saturation s = saturate(run, true);
  report("connected Realtime + explicit Block", s);
  require(s.refused, "connected Realtime + explicit Block: try_push must refuse when full");
  require(s.accepted > 0, "connected Realtime + explicit Block: nothing accepted");
  require(s.inputs_dropped == 0U, "connected Realtime + explicit Block: try_push dropped " +
                                      std::to_string(s.inputs_dropped));
  auto consumer = std::async(std::launch::async, [&] {
    std::this_thread::sleep_for(200ms);
    return run.pull("output", 5000);
  });
  const auto t0 = Clock::now();
  const bool pushed = run.push("input", TensorList{rgb(0xEE)});
  const auto waited =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
  require(consumer.wait_for(10s) == std::future_status::ready, "the consumer's pull returned");
  require(consumer.get().has_value(), "the consumer pulled a frame");
  require(pushed, "the blocking push is accepted once a pull makes room");
  require(waited >= 100, "the blocking push waited for room (" + std::to_string(waited) + " ms)");
  run.close();
}

// Connected graph, Realtime alone: the same, the preset does not reach a graph Run's ingress.
void connected_realtime_default_policy() {
  Run run = connected_run(options(RunPreset::Realtime, std::nullopt));
  const Saturation s = saturate(run, true);
  report("connected Realtime (policy left at its default)", s);
  require(s.refused, "connected Realtime: try_push must refuse when full");
  require(s.inputs_dropped == 0U,
          "connected Realtime: try_push dropped " + std::to_string(s.inputs_dropped));
  run.close();
}

// Linear Run, Realtime with Block explicit or left at its default: resolved to KeepLatest, so
// try_push never refuses.
void linear_realtime(std::optional<OverflowPolicy> policy, const std::string& what) {
  Run run = linear_run(options(RunPreset::Realtime, policy));
  const Saturation s = saturate(run, false);
  report(what, s);
  require(!s.refused, what + ": KeepLatest, try_push must never refuse");
  if (on_dataflow()) {
    require(s.inputs_dropped > 0U, what + ": KeepLatest evicts the oldest past the link depth");
  }
  run.close();
}

// The control: a linear Run on Balanced with Block refuses a try_push when full.
void linear_balanced_block() {
  Run run = linear_run(options(RunPreset::Balanced, OverflowPolicy::Block));
  const Saturation s = saturate(run, false);
  report("linear Balanced + Block", s);
  require(s.refused, "linear Balanced + Block: try_push must refuse when full");
  require(s.inputs_dropped == 0U,
          "linear Balanced + Block: try_push dropped " + std::to_string(s.inputs_dropped));
  run.close();
}

} // namespace

RUN_TEST("unit_run_preset_overflow_test", ([] {
           std::cerr << "[INFO] backend=" << backend_name() << std::endl;
           connected_realtime_explicit_block();
           connected_realtime_default_policy();
           linear_realtime(OverflowPolicy::Block, "linear Realtime + explicit Block");
           linear_realtime(std::nullopt, "linear Realtime (policy left at its default)");
           linear_balanced_block();
         }));

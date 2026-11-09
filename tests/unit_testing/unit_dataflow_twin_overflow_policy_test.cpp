// Dataflow twin of unit_run_overflow_policy_parity_test (H0 C4, plan D12), on the public API
// only: every Run here requests RunBackend::Dataflow explicitly, and no internal header or
// runtime type is used (counters come from MeasureScope, not run_internal::stats).
//
// The GStreamer test stays GStreamer-only: its PushPath::Holder probe pushes a GStreamer buffer
// holder (an error on Dataflow), and its Runner block parks RunCore's input worker on latency_mu
// and inspects pipeline.in_queue. Kept here, per contract:
// - try_push policy parity across the Mat, TensorList and Sample paths of the same RGB graph
//   (Block and DropIncoming report false when the queue is full, KeepLatest never does; drops
//   only for the dropping policies), plus the exact Dataflow numbers: the queue is the link,
//   whose depth is RunOptions::queue_depth + the Output's max_buffers (GStreamer's input queue
//   plus its appsink).
// - push vs try_push accounting: try_push applies the policy, a blocking push never drops.
// - the holder path's Dataflow outcome: a structured kPipelineShape error, nothing pushed.
// - the Runner contract on the tensor-pair (record) graph: warm-up, incompatible media rejected
//   with kInputShape before anything is queued, valid input accepted afterwards, try_push never
//   waits for room, the configured admission policy decides which pair survives, pairs and their
//   identity stay together, and nothing is accepted after close_input. Where the GStreamer test
//   checks that a queued input retains the caller's storage, the Dataflow contract is the
//   opposite and is checked as such: push copies into a pool slot, so the caller's storage is
//   free when push returns (c3.md section 4 row 3).

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;

// The link holds RunOptions::queue_depth + OutputOptions::max_buffers messages.
constexpr int kRgbQueueDepth = 1;   // RunOptions::queue_depth of the RGB graph's Runs below
constexpr int kRgbMaxBuffers = 128; // OutputOptions::EveryFrame(128) below, as the GStreamer test
constexpr int kRgbLinkDepth = kRgbQueueDepth + kRgbMaxBuffers;

enum class PushPath { Mat, Tensor, Sample };

const char* path_name(PushPath path) {
  switch (path) {
  case PushPath::Mat:
    return "Mat";
  case PushPath::Tensor:
    return "TensorList";
  case PushPath::Sample:
    return "Sample";
  }
  return "?";
}

const char* policy_name(OverflowPolicy policy) {
  switch (policy) {
  case OverflowPolicy::Block:
    return "Block";
  case OverflowPolicy::DropIncoming:
    return "DropIncoming";
  case OverflowPolicy::KeepLatest:
    return "KeepLatest";
  }
  return "?";
}

struct PolicyProbeResult {
  int attempts = 0;
  int accepted = 0;
  bool saw_false = false;
  std::uint64_t inputs_dropped = 0;
};

// The GStreamer test's graph, on Dataflow.
Run make_async_rgb_run_with_policy(const Tensor& seed, OverflowPolicy overflow_policy,
                                   int queue_depth = kRgbQueueDepth) {
  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Image;
  src_opt.format = FormatTag::RGB;
  src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(kRgbMaxBuffers)));

  RunOptions run_opt;
  run_opt.backend = RunBackend::Dataflow;
  run_opt.queue_depth = queue_depth;
  run_opt.overflow_policy = overflow_policy;
  run_opt.advanced.copy_input = true;
  return graph.build(TensorList{seed}, run_opt);
}

Sample tensor_to_sample(const Tensor& tensor) {
  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = tensor;
  sample.payload_tag = "RGB";
  sample.owned = true;
  return sample;
}

PolicyProbeResult probe_policy(OverflowPolicy policy, PushPath path) {
  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4A);
  Run run = make_async_rgb_run_with_policy(seed, policy, kRgbQueueDepth);
  const cv::Mat mat_seed(48, 64, CV_8UC3, cv::Scalar(40, 100, 180));
  const Sample sample_seed = tensor_to_sample(seed);

  PolicyProbeResult result;
  MeasureScope scope = run.start_measurement();
  constexpr int kMaxAttempts = 16384;
  for (int i = 0; i < kMaxAttempts; ++i) {
    ++result.attempts;
    bool ok = false;
    switch (path) {
    case PushPath::Mat:
      ok = run.try_push(std::vector<cv::Mat>{mat_seed});
      break;
    case PushPath::Tensor:
      ok = run.try_push(TensorList{seed});
      break;
    case PushPath::Sample:
      ok = run.try_push(Sample{sample_seed});
      break;
    }
    if (ok) {
      ++result.accepted;
      continue;
    }
    result.saw_false = true;
    if (policy != OverflowPolicy::KeepLatest) {
      break;
    }
  }
  result.inputs_dropped = scope.stop().counters.inputs_dropped;
  run.stop();
  return result;
}

void require_policy_parity(OverflowPolicy policy, bool expect_backpressure_false,
                           bool expect_drops) {
  for (const PushPath path : {PushPath::Mat, PushPath::Tensor, PushPath::Sample}) {
    const PolicyProbeResult r = probe_policy(policy, path);
    const std::string what = std::string(policy_name(policy)) + "/" + path_name(path) + ": ";
    require(r.saw_false == expect_backpressure_false,
            what + (expect_backpressure_false ? "did not report policy backpressure"
                                              : "unexpectedly reported false under KeepLatest"));
    require(r.accepted > 0, what + "accepted no input");
    require(expect_drops ? r.inputs_dropped > 0 : r.inputs_dropped == 0,
            what + (expect_drops ? "expected dropped inputs" : "unexpectedly dropped inputs") +
                " (inputs_dropped=" + std::to_string(r.inputs_dropped) + ")");
    // Dataflow's exact numbers: the link holds kRgbLinkDepth messages; try_push past that is
    // refused (Block: no drop; DropIncoming: one drop, then the probe stops) or evicts the oldest
    // (KeepLatest: every attempt past the depth is one drop).
    if (policy == OverflowPolicy::KeepLatest) {
      require(r.accepted == r.attempts, what + "KeepLatest accepts every try_push");
      require(r.inputs_dropped == static_cast<std::uint64_t>(r.attempts - kRgbLinkDepth),
              what + "KeepLatest drops one per try_push past the link depth, got " +
                  std::to_string(r.inputs_dropped));
    } else {
      require(r.accepted == kRgbLinkDepth, what + "accepted " + std::to_string(r.accepted) +
                                               ", the link depth is " +
                                               std::to_string(kRgbLinkDepth));
      require(r.inputs_dropped == (policy == OverflowPolicy::DropIncoming ? 1U : 0U),
              what + "drop count at the first refusal");
    }
  }
}

// The holder path of the GStreamer test is an error on Dataflow: structured, nothing pushed.
void holder_path_is_a_structured_error() {
  const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4A);
  Run run = make_async_rgb_run_with_policy(seed, OverflowPolicy::Block, kRgbQueueDepth);
  const auto holder = std::make_shared<int>(7);
  MeasureScope scope = run.start_measurement();
  require_neat_error([&] { (void)run.try_push_holder(holder); }, error_codes::kPipelineShape,
                     "holders");
  require_neat_error([&] { (void)run.push_holder(holder); }, error_codes::kPipelineShape,
                     "holders");
  const MeasureReport report = scope.stop();
  require(report.counters.inputs_pushed == 0 && report.counters.inputs_dropped == 0,
          "a refused holder push neither publishes nor drops");
  Sample s;
  PullError err;
  require(run.pull(0, s, &err) == PullStatus::Timeout, "nothing was published by a holder push");
  run.stop();
}

// try_push applies the policy; a blocking push waits and never drops, whatever the policy.
Tensor tagged_rgb(std::uint8_t tag) {
  return make_color_tensor(8, 8, ImageSpec::PixelFormat::RGB, tag);
}

// The smallest link: queue_depth 1 + max_buffers 1 = 2 (neither term can be below 1).
constexpr int kSmallLinkDepth = 2;

Run smallest_link_run(OverflowPolicy policy) {
  Graph graph;
  graph.add(nodes::Input());
  OutputOptions out;
  out.max_buffers = 1;
  graph.add(nodes::Output(out));
  RunOptions opt;
  opt.backend = RunBackend::Dataflow;
  opt.queue_depth = 1;
  opt.overflow_policy = policy;
  return graph.build(TensorList{tagged_rgb(1)}, opt);
}

void push_vs_try_push_accounting() {
  for (const OverflowPolicy policy :
       {OverflowPolicy::Block, OverflowPolicy::DropIncoming, OverflowPolicy::KeepLatest}) {
    const std::string what = std::string(policy_name(policy)) + ": ";
    // try_push at the smallest depth: five attempts into a link that holds two.
    {
      Run run = smallest_link_run(policy);
      std::vector<bool> accepted;
      MeasureScope scope = run.start_measurement();
      for (int i = 0; i < 5; ++i) {
        accepted.push_back(run.try_push(TensorList{tagged_rgb(static_cast<std::uint8_t>(i))}));
      }
      const MeasureCounters counters = scope.stop().counters;
      const bool keep_latest = policy == OverflowPolicy::KeepLatest;
      const std::vector<bool> expected = keep_latest
                                             ? std::vector<bool>(5, true)
                                             : std::vector<bool>{true, true, false, false, false};
      require(accepted == expected, what + "try_push outcomes at depth 2");
      const std::uint64_t expected_drops = policy == OverflowPolicy::Block ? 0U : 3U;
      require(counters.inputs_dropped == expected_drops,
              what + "try_push drops: expected " + std::to_string(expected_drops) + ", got " +
                  std::to_string(counters.inputs_dropped));
      for (int i = 0; i < kSmallLinkDepth; ++i) {
        const auto kept = run.pull_tensors(1000);
        require(kept.size() == 1U, what + "two frames kept");
        const int tag = keep_latest ? 5 - kSmallLinkDepth + i : i;
        require(kept.front().copy_dense_bytes_tight() ==
                    tagged_rgb(static_cast<std::uint8_t>(tag)).copy_dense_bytes_tight(),
                what + (keep_latest ? "KeepLatest keeps the newest" : "the oldest are kept"));
      }
      run.stop();
    }
    // A blocking push at the smallest depth with a paced puller: every frame arrives, in order,
    // no drop.
    {
      Run run = smallest_link_run(policy);
      constexpr int kFrames = 24;
      std::atomic<bool> push_failed{false};
      MeasureScope scope = run.start_measurement();
      std::thread pusher([&] {
        for (int i = 0; i < kFrames; ++i) {
          Sample s = tensor_to_sample(tagged_rgb(static_cast<std::uint8_t>(i)));
          s.frame_id = i;
          if (!run.push(Sample{s})) {
            push_failed = true;
          }
        }
      });
      for (int i = 0; i < kFrames; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(i % 3 == 0 ? 4 : 0));
        Sample s;
        PullError err;
        require(run.pull(5000, s, &err) == PullStatus::Ok, what + "blocking push: frame arrives");
        require(s.frame_id == i, what + "blocking push: in order at " + std::to_string(i));
      }
      pusher.join();
      const MeasureCounters counters = scope.stop().counters;
      require(!push_failed, what + "a blocking push never returns false on a live run");
      require(counters.inputs_dropped == 0, what + "a blocking push never drops");
      run.stop();
    }
  }
}

// The Runner block of the GStreamer test, on the tensor-pair graph (a positional record on
// Dataflow since C2).
Sample make_tensor_pair(int id) {
  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.payload_type = PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  sample.frame_id = id;
  sample.stream_id = "stream0";
  sample.pts_ns = id * 1000;
  for (int i = 0; i < 2; ++i) {
    Tensor tensor =
        Tensor::from_vector(std::vector<float>{float(id), float(i)}, {2}, TensorMemory::CPU);
    tensor.route.physical_index = 0;
    tensor.route.memory_index = i;
    sample.tensors.push_back(std::move(tensor));
  }
  return sample;
}

constexpr int kPairQueueDepth = 1; // RunOptions::queue_depth below
constexpr int kPairMaxBuffers = 8; // OutputOptions::EveryFrame(8) below
constexpr int kPairLinkDepth = kPairQueueDepth + kPairMaxBuffers;

Run make_tensor_pair_run(const Sample& seed, OverflowPolicy policy) {
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::FP32;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::Output(OutputOptions::EveryFrame(kPairMaxBuffers)));
  RunOptions options;
  options.backend = RunBackend::Dataflow;
  options.queue_depth = kPairQueueDepth;
  options.overflow_policy = policy;
  options.advanced.copy_input = true;
  return graph.build(Sample{seed}, options);
}

void require_tensor_pair(const TensorList& output, int id, const std::string& what) {
  require(output.size() == 2, what + ": the graph must forward both members");
  for (std::size_t i = 0; i < output.size(); ++i) {
    require(output[i].dtype == TensorDType::Float32 && output[i].shape == std::vector<int64_t>{2},
            what + ": member type or shape changed");
    auto mapping = output[i].map_read();
    float values[2]{};
    require(mapping.data && mapping.size_bytes >= sizeof(values), what + ": output truncated");
    std::memcpy(values, mapping.data, sizeof(values));
    require(values[0] == float(id) && values[1] == float(i),
            what + ": member contents or order changed (expected pair " + std::to_string(id) +
                ", got " + std::to_string(values[0]) + ")");
  }
}

void runner_try_push_preserves_policy(OverflowPolicy policy, bool tensor_list) {
  const std::string what =
      std::string("Runner/") + policy_name(policy) + (tensor_list ? "/TensorList" : "/Sample");
  const Sample seed = make_tensor_pair(7);
  Run run = make_tensor_pair_run(seed, policy);
  MeasureScope scope = run.start_measurement();
  Model::Runner runner(std::move(run));
  auto submit = [&](const Sample& sample) {
    return tensor_list ? runner.try_push(sample.tensors) : runner.try_push(sample);
  };
  require(submit(seed), what + ": warm-up must be accepted");
  require_tensor_pair(runner.pull(2000).tensors, 7, what + " warm-up");

  // Incompatible media: a structured shape rejection, before anything is queued or dropped.
  Sample incompatible = make_tensor_pair(8);
  incompatible.tensors.front() =
      Tensor::from_vector(std::vector<std::uint8_t>{8, 0}, {2}, TensorMemory::CPU);
  require_neat_error([&] { (void)submit(incompatible); }, error_codes::kInputShape);
  require(submit(make_tensor_pair(8)), what + ": valid input after a media rejection");
  require_tensor_pair(runner.pull(2000).tensors, 8, what + " after rejection");

  // Fill the link (no puller), then one more try_push: it must not wait for room, and the policy
  // decides. The caller's storage is free as soon as push returns (Dataflow copies).
  for (int id = 1; id <= kPairLinkDepth; ++id) {
    Sample pair = make_tensor_pair(id);
    const std::weak_ptr<TensorBuffer> storage = pair.tensors[0].storage;
    require(submit(pair), what + ": pair " + std::to_string(id) + " fits the link");
    pair = {};
    require(storage.expired(), what + ": push copied; the caller's storage is released");
  }
  const Sample latest = make_tensor_pair(kPairLinkDepth + 1);
  auto pending = std::async(std::launch::async, [&] { return submit(latest); });
  const bool returned = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
  if (!returned) {
    runner.close_input();
  }
  const bool accepted = pending.get();
  require(returned, what + ": try_push must not wait for queue space");
  const bool replaces = policy == OverflowPolicy::KeepLatest;
  require(accepted == replaces, what + ": the configured admission policy decides");

  // Drain: the surviving pairs, in order, each with its identity.
  const int first = replaces ? 2 : 1;
  for (int id = first; id < first + kPairLinkDepth; ++id) {
    const Sample out = runner.pull(2000);
    require_tensor_pair(out.tensors, id, what + " drain");
    if (!tensor_list) {
      require(out.frame_id == id && out.stream_id == "stream0" && out.pts_ns == id * 1000,
              what + ": Sample identity must stay attached to its pair " + std::to_string(id));
    }
  }
  const MeasureCounters counters = scope.stop().counters;
  require(counters.inputs_dropped == (policy == OverflowPolicy::Block ? 0U : 1U),
          what + ": drop accounting, got " + std::to_string(counters.inputs_dropped));
  runner.close_input();
  require(!submit(latest), what + ": no input after close_input");
  runner.close();
}

} // namespace

RUN_TEST("unit_dataflow_twin_overflow_policy_test", ([] {
           require_policy_parity(OverflowPolicy::Block, true, false);
           require_policy_parity(OverflowPolicy::DropIncoming, true, true);
           require_policy_parity(OverflowPolicy::KeepLatest, false, true);
           holder_path_is_a_structured_error();
           push_vs_try_push_accounting();
           for (auto policy :
                {OverflowPolicy::Block, OverflowPolicy::DropIncoming, OverflowPolicy::KeepLatest}) {
             for (bool tensor_list : {false, true}) {
               runner_try_push_preserves_policy(policy, tensor_list);
             }
           }
         }));

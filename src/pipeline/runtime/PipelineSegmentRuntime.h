#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "internal/InputStream.h"
#include "ExecutionGraphPlan.h"
#include "pipeline/internal/InputRouteProcessor.h"
#include "pipeline/Run.h"

#include <opencv2/core/mat.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace simaai::neat {

namespace graph::runtime {
template <typename T> class BlockingQueue;
}

enum class QueuedInputKind {
  Mat,
  Tensor,
  Holder,
  Message,
};

struct InputItem {
  QueuedInputKind kind = QueuedInputKind::Mat;
  cv::Mat mat;
  simaai::neat::Tensor tensor;
  std::shared_ptr<void> holder;
  Sample msg;
};

namespace runtime {

struct RunCore;

// Runtime state for one compiled pipeline segment.
//
// Phase 3B keeps the existing public Run behavior intact while separating the
// single linear segment state from RunCore. Later Phase 3 slices will move the
// segment-specific algorithms currently implemented in Run facade files behind
// this type and allow RunCore to own more than one segment.
struct PipelineSegmentRuntime {
  // Graph-only transport state attached to a pipeline segment while the graph
  // runtime is being collapsed onto the same segment abstraction as linear
  // Run. This intentionally contains only queues/threads/identity maps around
  // a segment; the compiled segment description stays in the graph compiler
  // plan and aggregate lifecycle/error/metrics stay in RunCore/GraphRun state.
  struct GraphTransport {
    std::atomic<bool> built{false};
    bool building = false;
    bool has_input = false;
    bool has_output = false;
    std::vector<std::string> expected_buffer_names;

    std::shared_ptr<simaai::neat::graph::runtime::BlockingQueue<Sample>> input_queue;
    std::thread push_thread;
    std::thread pull_thread;
    std::atomic<bool> push_done{false};
    std::atomic<bool> pull_done{false};

    std::mutex stream_mu;
    std::atomic<int64_t> next_input_seq{0};
    struct SampleIdentity {
      int64_t frame_id = -1;
      int64_t pts_ns = -1;
      int64_t dts_ns = -1;
      int64_t duration_ns = -1;
      int64_t input_seq = -1;
      int64_t orig_input_seq = -1;
      std::string stream_id;
      std::string stream_label;
      std::string port_name;
    };

    std::unordered_map<int64_t, SampleIdentity> identity_by_input_seq;
    std::unordered_map<int64_t, std::deque<SampleIdentity>> identity_by_frame;
    std::deque<int64_t> input_seq_order;
    std::deque<int64_t> stream_order;
    std::deque<SampleIdentity> pending_identities;
    std::atomic<int64_t> identity_rewrite_count{0};
    std::atomic<int64_t> identity_map_miss_count{0};

    std::mutex mu;
    std::condition_variable cv;
  };

  std::thread input_thread;

  // Compiled segment materialization state used by graph/runtime execution.
  // Linear public Run uses the InputStream/queue fields below directly; graph
  // execution uses the same segment runtime plus GraphTransport for edge
  // queues and lazy segment build.
  PipelineSegmentPlan seg;
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  GraphOptions route_options;
  RunOptions run_options;
  std::string last_pipeline;
  std::shared_ptr<RunCore> run_core;
  GraphTransport transport;

  InputStream stream;
  InputStreamOptions stream_opt;
  std::optional<InputOptions> tensor_input_opt_for_cv;
  pipeline_internal::InputRouteProcessorPtr input_route_processor;

  std::mutex in_mu;
  std::condition_variable in_cv;
  std::mutex out_mu;
  std::condition_variable out_cv;
  std::deque<InputItem> in_queue;
  std::deque<Sample> out_queue;
  std::deque<std::chrono::steady_clock::time_point> pending_times;

  bool supports_push = false;
  bool supports_pull = false;
  bool zero_copy_fallback_enabled = false;
  std::atomic<bool> copy_output_latched{true};
  std::atomic<bool> zero_copy_warned{false};
  bool input_closed = false;
  std::atomic<bool> input_thread_done{false};
  std::atomic<bool> pull_timeout_logged{false};
};

} // namespace runtime

} // namespace simaai::neat

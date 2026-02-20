/**
 * @file
 * @ingroup pipeline
 * @brief Run push/pull runtime handle and diagnostics stats.
 */
#pragma once

#include "pipeline/SessionOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {

class InputStream;
struct InputStreamOptions;

enum class OverflowPolicy {
  Block = 0,
  KeepLatest,
  DropIncoming,
};

enum class RunPreset {
  Realtime,
  Balanced,
  Reliable,
};

enum class OutputMemory {
  Auto = 0,
  ZeroCopy,
  Owned,
};

struct RunAdvancedOptions {
  bool copy_input = false;
  std::size_t max_input_bytes = 0;
};

struct RunOptions {
  RunPreset preset = RunPreset::Balanced;
  int queue_depth = 4;
  OverflowPolicy overflow_policy = OverflowPolicy::Block;
  OutputMemory output_memory = OutputMemory::Auto;
  bool enable_metrics = false;
  RunAdvancedOptions advanced{};
  std::function<void(const struct InputDropInfo&)> on_input_drop;
};

struct InputDropInfo {
  SampleKind kind = SampleKind::Unknown;
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
  int64_t frame_id = -1;
  std::string stream_id;
  std::string port_name;
  std::string reason;
};

struct InputStreamStats {
  std::uint64_t push_count = 0;
  std::uint64_t push_failures = 0;
  std::uint64_t pull_count = 0;
  std::uint64_t poll_count = 0;
  std::uint64_t dropped_frames = 0;
  std::uint64_t renegotiations = 0;
  std::uint64_t alloc_grows = 0;
  std::uint64_t growth_blocked = 0;
  std::uint64_t renegotiation_blocked = 0;
  double avg_alloc_us = 0.0;
  double avg_map_us = 0.0;
  double avg_copy_us = 0.0;
  double avg_push_us = 0.0;
  double avg_pull_wait_us = 0.0;
  double avg_decode_us = 0.0;
};

struct RunStats {
  std::uint64_t inputs_enqueued = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_ready = 0;
  std::uint64_t outputs_pulled = 0;
  std::uint64_t outputs_dropped = 0;
  double avg_latency_ms = 0.0;
  double min_latency_ms = 0.0;
  double max_latency_ms = 0.0;
};

struct RunStageStats {
  std::string stage_name;
  std::uint64_t samples = 0;
  std::uint64_t total_us = 0;
  std::uint64_t max_us = 0;
};

struct RunElementTimingStats {
  std::string element_name;
  std::uint64_t samples = 0;
  std::uint64_t total_us = 0;
  std::uint64_t max_us = 0;
  std::uint64_t min_us = 0;
  std::uint64_t missed_in = 0;
  std::uint64_t missed_out = 0;
};

struct RunElementFlowStats {
  std::string element_name;
  std::uint64_t in_buffers = 0;
  std::uint64_t out_buffers = 0;
  std::uint64_t in_bytes = 0;
  std::uint64_t out_bytes = 0;
  std::uint64_t caps_changes = 0;
};

struct RunDiagSnapshot {
  std::vector<RunStageStats> stages;
  std::vector<BoundaryFlowStats> boundaries;
  std::vector<RunElementTimingStats> element_timings;
  std::vector<RunElementFlowStats> element_flows;
};

struct RunReportOptions {
  bool include_pipeline = true;
  bool include_stage_timings = true;
  bool include_element_timings = true;
  bool include_boundaries = true;
  bool include_flow_stats = true;
  bool include_node_reports = false;
  bool include_next_cpu = false;
  bool include_queue_depth = true;
  bool include_num_buffers = true;
  bool include_run_stats = true;
  bool include_input_stats = true;
  bool include_system_info = false;
};

/**
 * @brief Runtime handle for pushing inputs and pulling outputs.
 */
class Run {
public:
  Run() = default;
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;

  Run(Run&&) noexcept;
  Run& operator=(Run&&) noexcept;
  ~Run();

  explicit operator bool() const noexcept;
  bool can_push() const;
  bool can_pull() const;
  bool running() const;

  bool push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  bool push(const simaai::neat::Tensor& input);
  bool try_push(const simaai::neat::Tensor& input);
  bool push(const Sample& msg);
  bool try_push(const Sample& msg);
  // Internal: pushes a GstBuffer held by a tensor ref to preserve plugin metadata.
  bool push_holder(const std::shared_ptr<void>& holder);
  bool try_push_holder(const std::shared_ptr<void>& holder);
  void close_input();
  PullStatus pull(int timeout_ms, Sample& out, PullError* err = nullptr);
  std::optional<Sample> pull(int timeout_ms = -1);
  std::optional<simaai::neat::Tensor> pull_tensor(int timeout_ms = -1);
  simaai::neat::Tensor pull_tensor_or_throw(int timeout_ms = -1);
  std::optional<simaai::neat::Tensor> pull_tensor_matching(const std::string& payload_tag,
                                                           int timeout_ms = -1);
  Sample push_and_pull(const cv::Mat& input, int timeout_ms = -1);
  Sample push_and_pull(const simaai::neat::Tensor& input, int timeout_ms = -1);
  Sample push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms = -1);
  Sample run(const cv::Mat& input, int timeout_ms = -1);
  Sample run(const simaai::neat::Tensor& input, int timeout_ms = -1);
  Sample run(const Sample& input, int timeout_ms = -1);
  int warmup(const cv::Mat& input, int warm = -1, int timeout_ms = -1);

  RunStats stats() const;
  InputStreamStats input_stats() const;
  RunDiagSnapshot diag_snapshot() const;
  std::string report(const RunReportOptions& opt = {}) const;
  std::string last_error() const;
  std::string diagnostics_summary() const;

  void stop();
  void close();

private:
  struct State;
  std::shared_ptr<State> state_;
  bool owns_ref_ = false;

  explicit Run(std::shared_ptr<State> state);
  bool push_impl(const cv::Mat& input, bool block);
  bool push_impl(const simaai::neat::Tensor& input, bool block);
  bool push_holder_impl(const std::shared_ptr<void>& holder, bool block);
  bool push_message_impl(const Sample& msg, bool block);
  static Run create(InputStream stream, const RunOptions& opt,
                    const struct InputStreamOptions& stream_opt);
  friend class Session;
};

} // namespace simaai::neat

#pragma once

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Run.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>

namespace sima_test {

inline int elapsed_ms(std::chrono::steady_clock::time_point start,
                      std::chrono::steady_clock::time_point end) {
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

inline void require_timeout_window(const char* op_name, int measured_ms, int timeout_ms,
                                   int early_slack_ms = 40, int late_slack_ms = 700) {
  const int lower = timeout_ms - early_slack_ms;
  const int upper = timeout_ms + late_slack_ms;
  if (measured_ms < lower || measured_ms > upper) {
    throw std::runtime_error(std::string(op_name) + " timeout outside expected window: measured=" +
                             std::to_string(measured_ms) + "ms expected=[" + std::to_string(lower) +
                             "," + std::to_string(upper) +
                             "] timeout=" + std::to_string(timeout_ms) + "ms");
  }
}

inline bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

inline simaai::neat::Run make_async_rgb_run(const simaai::neat::Tensor& seed,
                                            int producer_queue_depth = 16,
                                            int consumer_queue_depth = 16) {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions run_opt;
  run_opt.queue_depth = std::max(producer_queue_depth, consumer_queue_depth);
  run_opt.overflow_policy = OverflowPolicy::Block;

  return graph.build(simaai::neat::TensorList{seed}, run_opt);
}

struct TryPushFillResult {
  int attempts = 0;
  bool saw_backpressure = false;
  int max_call_ms = 0;
};

inline TryPushFillResult fill_try_push_queue_non_blocking(simaai::neat::Run& run,
                                                          const simaai::neat::Tensor& sample,
                                                          int max_attempts = 1024) {
  TryPushFillResult out;
  for (int i = 0; i < max_attempts; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = run.try_push(simaai::neat::TensorList{sample});
    const auto t1 = std::chrono::steady_clock::now();

    ++out.attempts;
    out.max_call_ms = std::max(out.max_call_ms, elapsed_ms(t0, t1));
    if (!ok) {
      out.saw_backpressure = true;
      break;
    }
  }
  return out;
}

inline bool likely_runtime_missing(const std::string& msg) {
  return is_dispatcher_unavailable(msg) || msg.find("No such element") != std::string::npos ||
         msg.find("missing element") != std::string::npos ||
         msg.find("not found") != std::string::npos;
}

inline int rtsp_port_base_from_env(int fallback = 8554) {
  const char* raw = std::getenv("SIMA_RTSP_PORT_BASE");
  if (!raw || !*raw)
    return fallback;
  const int v = std::atoi(raw);
  return (v > 0) ? v : fallback;
}

inline int allocate_local_rtsp_port(int block_size = 25) {
  static std::atomic<int> seq{0};
  const int base = rtsp_port_base_from_env();
  const int pid_bucket = static_cast<int>(::getpid() % 200);
  return base + pid_bucket * block_size + seq.fetch_add(1);
}

} // namespace sima_test

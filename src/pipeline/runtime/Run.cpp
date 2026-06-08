#include "pipeline/Run.h"
#include "RunInternal.h"

#include "internal/InputStream.h"
#include "pipeline/RunExport.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat {
using pipeline_internal::env_bool;

namespace {
using run_internal::force_copy_sample_if_zero_copy;
using run_internal::queue_full;
using run_internal::sample_has_zero_copy_tensor;

std::string read_first_line(const char* path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};
  std::string line;
  std::getline(in, line);
  return line;
}

bool run_input_thread_timing_enabled() {
  return pipeline_internal::env_bool("SIMA_RUN_INPUT_THREAD_TIMING", false);
}

int run_input_thread_timing_limit() {
  static const int limit =
      std::max(0, pipeline_internal::env_int("SIMA_RUN_INPUT_THREAD_TIMING_LIMIT", 32));
  return limit;
}

const char* queued_input_kind_name(QueuedInputKind kind) {
  switch (kind) {
  case QueuedInputKind::Tensor:
    return "Tensor";
  case QueuedInputKind::Holder:
    return "Holder";
  case QueuedInputKind::Message:
    return "Message";
  case QueuedInputKind::Mat:
    return "Mat";
  }
  return "Unknown";
}

std::vector<std::string> tail_lines(const std::string& path, size_t max_lines) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};
  std::deque<std::string> buf;
  std::string line;
  while (std::getline(in, line)) {
    if (buf.size() == max_lines)
      buf.pop_front();
    buf.push_back(line);
  }
  return std::vector<std::string>(buf.begin(), buf.end());
}

bool contains_case_insensitive(const std::string& haystack, const std::string& needle) {
  if (needle.empty())
    return true;
  const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                              [](char a, char b) {
                                return std::tolower(static_cast<unsigned char>(a)) ==
                                       std::tolower(static_cast<unsigned char>(b));
                              });
  return it != haystack.end();
}

std::string collect_system_info() {
  std::ostringstream oss;
  const std::string governor =
      read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  const std::string loadavg = read_first_line("/proc/loadavg");
  if (!governor.empty()) {
    oss << "cpu_governor=" << governor;
  }
  if (!loadavg.empty()) {
    if (!oss.str().empty())
      oss << " ";
    oss << "loadavg=" << loadavg;
  }

  std::vector<std::string> rpmsg_lines;
  const std::vector<std::string> kern_tail = tail_lines("/var/log/kern.log", 200);
  const auto maybe_add_rpmsg = [&](const std::vector<std::string>& lines) {
    for (const auto& line : lines) {
      if (!contains_case_insensitive(line, "rpmsg"))
        continue;
      if (contains_case_insensitive(line, "error") || contains_case_insensitive(line, "err") ||
          contains_case_insensitive(line, "fail")) {
        rpmsg_lines.push_back(line);
      }
    }
  };

  if (!kern_tail.empty()) {
    maybe_add_rpmsg(kern_tail);
  } else {
    const std::vector<std::string> syslog_tail = tail_lines("/var/log/syslog", 200);
    maybe_add_rpmsg(syslog_tail);
  }

  if (!rpmsg_lines.empty()) {
    if (!oss.str().empty())
      oss << " ";
    oss << "rpmsg_errors=" << rpmsg_lines.size();
    const size_t show = std::min<size_t>(rpmsg_lines.size(), 3);
    for (size_t i = rpmsg_lines.size() - show; i < rpmsg_lines.size(); ++i) {
      oss << "\n  rpmsg: " << rpmsg_lines[i];
    }
  }

  return oss.str();
}

} // namespace

Run::Run(std::shared_ptr<runtime::RunCore> core) : core_(std::move(core)) {
  if (!core_) {
    return;
  }
  const RunAutoExportOptions& export_options =
      !core_->opt.run_export.path.empty() ? core_->opt.run_export : core_->opt.graph_run_export;
  if (export_options.path.empty()) {
    return;
  }

  RunExportOptions opt;
  opt.label = export_options.label;
  opt.include_metrics = export_options.include_metrics;
  opt.include_power = export_options.include_power;
  opt.indent = export_options.indent;

  std::string err;
  if (!save_run_json(*this, export_options.path, opt, &err)) {
    throw std::runtime_error(err.empty() ? "Run graph export failed" : err);
  }
}

std::shared_ptr<runtime::RunCore> run_internal::release_core(Run& run) {
  return std::move(run.core_);
}

std::shared_ptr<const runtime::RunCore> run_internal::core(const Run& run) {
  return run.core_;
}

Run::~Run() {
  close();
}

std::shared_ptr<runtime::RunCore> runtime::RunCore::start_single_pipeline(
    InputStream stream, const RunOptions& opt, const InputStreamOptions& stream_opt, RunMode mode,
    const std::optional<InputOptions>& tensor_input_opt_for_cv,
    pipeline_internal::InputRouteProcessorPtr input_route_processor) {
  auto st = std::make_shared<runtime::RunCore>();
  runtime::initialize_run_identity(*st);
  st->pipeline.stream = std::move(stream);
  st->opt = opt;
  st->pipeline.stream_opt = stream_opt;
  st->mode = mode;
  st->pipeline.tensor_input_opt_for_cv = tensor_input_opt_for_cv;
  st->pipeline.input_route_processor = std::move(input_route_processor);
  st->pipeline.supports_push = st->pipeline.stream.can_push();
  st->pipeline.supports_pull = st->pipeline.stream.can_pull();
  st->pipeline.copy_output_latched.store(stream_opt.copy_output, std::memory_order_relaxed);
  st->pipeline.zero_copy_fallback_enabled =
      (opt.preset == RunPreset::Balanced) && !stream_opt.copy_output;
  st->diag_enabled = env_bool("SIMA_ASYNC_TPUT_DIAG", false);
  if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
      pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
    std::fprintf(stderr, "[PIPELINE] create supports_push=%s supports_pull=%s\n",
                 st->pipeline.supports_push ? "true" : "false",
                 st->pipeline.supports_pull ? "true" : "false");
  }
  if (st->diag_enabled) {
    st->diag_sysinfo = collect_system_info();
  }
  if (opt.power_monitor.enabled) {
    st->power_monitor = std::make_unique<PowerMonitor>(opt.power_monitor);
    st->power_monitor->start();
  }

  auto on_output = [st](Sample out) {
    if (st->stop_requested.load()) {
      if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
          pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
        const char* kind = "Unknown";
        if (out.kind == SampleKind::TensorSet)
          kind = "TensorSet";
        else if (sample_is_multi_output(out))
          kind = "Bundle";
        std::fprintf(
            stderr,
            "[PIPELINE] on_output_drop kind=%s frame_id=%lld stream_id=%s reason=stop_requested\n",
            kind, static_cast<long long>(out.frame_id), out.stream_id.c_str());
      }
      return;
    }

    std::chrono::steady_clock::time_point t0;
    bool have_ts = false;
    {
      std::lock_guard<std::mutex> lock(st->latency_mu);
      if (!st->pipeline.pending_times.empty()) {
        t0 = st->pipeline.pending_times.front();
        st->pipeline.pending_times.pop_front();
        have_ts = true;
      }
    }
    if (have_ts) {
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::lock_guard<std::mutex> lock(st->latency_mu);
      st->latency_count += 1;
      if (!st->latency_init) {
        st->latency_mean_ms = ms;
        st->latency_min_ms = ms;
        st->latency_max_ms = ms;
        st->latency_init = true;
      } else {
        const double n = static_cast<double>(st->latency_count);
        st->latency_mean_ms += (ms - st->latency_mean_ms) / n;
        st->latency_min_ms = std::min(st->latency_min_ms, ms);
        st->latency_max_ms = std::max(st->latency_max_ms, ms);
      }
      if (st->measurement_active) {
        st->measurement_latencies_ms.push_back(ms);
      }
    }
    {
      const auto now = std::chrono::steady_clock::now();
      std::lock_guard<std::mutex> lock(st->latency_mu);
      if (!st->output_timing_init) {
        st->first_output_at = now;
        st->output_timing_init = true;
      }
      st->last_output_at = now;
      if (st->measurement_active) {
        if (st->measurement_output_timing_init) {
          const double gap_ms =
              std::chrono::duration<double, std::milli>(now - st->measurement_last_output_at)
                  .count();
          st->measurement_frame_gaps_ms.push_back(gap_ms);
        }
        st->measurement_last_output_at = now;
        st->measurement_output_timing_init = true;
      }
    }

    {
      std::unique_lock<std::mutex> lock(st->pipeline.out_mu);
      const int max = st->opt.queue_depth;
      const bool has_zero_copy_output = sample_has_zero_copy_tensor(out);
      if (!st->pipeline.copy_output_latched.load(std::memory_order_relaxed) &&
          st->pipeline.zero_copy_fallback_enabled && has_zero_copy_output && max > 0 &&
          static_cast<int>(st->pipeline.out_queue.size()) >= max) {
        st->pipeline.copy_output_latched.store(true, std::memory_order_relaxed);
        if (!st->pipeline.zero_copy_warned.exchange(true, std::memory_order_relaxed)) {
          std::fprintf(stderr, "[WARN] Balanced preset: zero-copy output reliability trip; "
                               "switching to copy-output mode.\n");
        }
      }
      const bool copy_output = st->pipeline.copy_output_latched.load(std::memory_order_relaxed);
      if (copy_output && has_zero_copy_output) {
        force_copy_sample_if_zero_copy(out);
      }
      if (max > 0) {
        OverflowPolicy output_drop = st->opt.overflow_policy;
        if (output_drop == OverflowPolicy::Block && !copy_output &&
            pipeline_internal::env_bool("SIMA_PIPELINE_OUTPUT_DROP_ON_ZERO_COPY", true)) {
          output_drop = OverflowPolicy::KeepLatest;
        }
        if (output_drop == OverflowPolicy::Block) {
          st->pipeline.out_cv.wait(lock, [&]() {
            return st->stop_requested.load() ||
                   static_cast<int>(st->pipeline.out_queue.size()) < max;
          });
        } else if (queue_full(st->pipeline.out_queue, max)) {
          if (output_drop == OverflowPolicy::DropIncoming) {
            st->outputs_dropped.fetch_add(1, std::memory_order_relaxed);
            if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
                pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
              const char* kind = "Unknown";
              if (out.kind == SampleKind::TensorSet)
                kind = "TensorSet";
              else if (sample_is_multi_output(out))
                kind = "Bundle";
              std::fprintf(stderr,
                           "[PIPELINE] on_output_drop kind=%s frame_id=%lld stream_id=%s "
                           "reason=queue_full\n",
                           kind, static_cast<long long>(out.frame_id), out.stream_id.c_str());
            }
            return;
          }
          // Drop oldest to make room for the new output.
          if (!st->pipeline.out_queue.empty()) {
            st->pipeline.out_queue.pop_front();
            st->outputs_dropped.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      if (st->stop_requested.load())
        return;
      st->pipeline.out_queue.push_back(std::move(out));
      st->outputs_ready.fetch_add(1, std::memory_order_relaxed);
      if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
          pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
        const Sample& latest = st->pipeline.out_queue.back();
        const char* kind = "Unknown";
        if (latest.kind == SampleKind::TensorSet)
          kind = "TensorSet";
        else if (sample_is_multi_output(latest))
          kind = "Bundle";
        std::fprintf(stderr, "[PIPELINE] on_output kind=%s frame_id=%lld stream_id=%s queue=%zu\n",
                     kind, static_cast<long long>(latest.frame_id), latest.stream_id.c_str(),
                     st->pipeline.out_queue.size());
      }
    }
    st->pipeline.out_cv.notify_one();
  };

  if (st->pipeline.supports_pull) {
    st->pipeline.stream.start(on_output);
  }

  if (!st->pipeline.supports_push) {
    st->pipeline.input_thread_done.store(true);
    return st;
  }

  st->pipeline.input_thread = std::thread([st]() {
    const bool input_thread_timing = run_input_thread_timing_enabled();
    int input_thread_timing_count = 0;
    while (true) {
      InputItem item;
      std::size_t q_after_pop = 0;
      {
        std::unique_lock<std::mutex> lock(st->pipeline.in_mu);
        st->pipeline.in_cv.wait(lock, [&]() {
          return st->stop_requested.load() || st->pipeline.input_closed ||
                 !st->pipeline.in_queue.empty();
        });
        if (st->stop_requested.load())
          break;
        if (st->pipeline.in_queue.empty()) {
          if (st->pipeline.input_closed)
            break;
          continue;
        }
        item = std::move(st->pipeline.in_queue.front());
        st->pipeline.in_queue.pop_front();
        q_after_pop = st->pipeline.in_queue.size();
      }
      st->pipeline.in_cv.notify_one();

      const auto t0 = std::chrono::steady_clock::now();
      if (st->pipeline.supports_pull) {
        std::lock_guard<std::mutex> lock(st->latency_mu);
        st->pipeline.pending_times.push_back(t0);
      }
      try {
        switch (item.kind) {
        case QueuedInputKind::Tensor:
          st->pipeline.stream.push(item.tensor);
          break;
        case QueuedInputKind::Holder:
          st->pipeline.stream.push_holder(item.holder);
          break;
        case QueuedInputKind::Message:
          st->pipeline.stream.push_message(item.msg);
          break;
        case QueuedInputKind::Mat:
        default:
          st->pipeline.stream.push(item.mat);
          break;
        }
        st->inputs_pushed.fetch_add(1, std::memory_order_relaxed);
        if (input_thread_timing) {
          const int idx = input_thread_timing_count++;
          const int limit = run_input_thread_timing_limit();
          if (idx < limit || (limit > 0 && (idx % limit) == 0)) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto push_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            std::fprintf(
                stderr,
                "[RUN_INPUT_THREAD_TIMING] Run::input_thread idx=%d kind=%s q_after_pop=%zu "
                "stream_push_ns=%lld\n",
                idx, queued_input_kind_name(item.kind), q_after_pop,
                static_cast<long long>(push_ns));
          }
        }
      } catch (const std::exception& e) {
        if (st->pipeline.supports_pull) {
          std::lock_guard<std::mutex> lock(st->latency_mu);
          if (!st->pipeline.pending_times.empty()) {
            st->pipeline.pending_times.pop_back();
          }
        }
        if (pipeline_internal::env_bool("SIMA_PIPELINE_DEBUG", false) ||
            pipeline_internal::env_bool("SIMA_GRAPH_DEBUG", false)) {
          std::fprintf(stderr, "[PIPELINE] input_thread_error: %s\n", e.what());
        }
        std::lock_guard<std::mutex> lock(st->error_mu);
        st->error = e.what();
        st->stop_requested.store(true);
        st->pipeline.out_cv.notify_all();
        break;
      }
    }

    if (!st->stop_requested.load() && st->pipeline.input_closed) {
      try {
        st->pipeline.stream.signal_eos();
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(st->error_mu);
        st->error = e.what();
        st->stop_requested.store(true);
        st->pipeline.out_cv.notify_all();
      }
    }
    st->pipeline.input_thread_done.store(true);
    st->pipeline.out_cv.notify_all();
  });

  return st;
}

Run Run::create(InputStream stream, const RunOptions& opt, const InputStreamOptions& stream_opt,
                RunMode mode, const std::optional<InputOptions>& tensor_input_opt_for_cv,
                pipeline_internal::InputRouteProcessorPtr input_route_processor) {
  return Run(runtime::RunCore::start_single_pipeline(std::move(stream), opt, stream_opt, mode,
                                                     tensor_input_opt_for_cv,
                                                     std::move(input_route_processor)));
}

bool runtime::RunCore::valid() const noexcept {
  return graph_execution_ || static_cast<bool>(pipeline.stream);
}

bool runtime::RunCore::can_push() const {
  if (graph_execution_) {
    return graph_execution_->plan.default_input.has_value();
  }
  return pipeline.supports_push;
}

bool runtime::RunCore::can_pull() const {
  if (graph_execution_) {
    return graph_execution_->plan.default_output.has_value();
  }
  return pipeline.supports_pull;
}

bool runtime::RunCore::running() const {
  if (graph_execution_) {
    return !graph_stop_requested();
  }
  if (pipeline.supports_pull) {
    return pipeline.stream.running();
  }
  return !stop_requested.load();
}

std::vector<std::string> runtime::RunCore::input_names() const {
  std::vector<std::string> names;
  if (!graph_execution_) {
    return names;
  }
  names.reserve(graph_execution_->plan.named_inputs.size());
  for (const auto& kv : graph_execution_->plan.named_inputs) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> runtime::RunCore::output_names() const {
  std::vector<std::string> names;
  if (!graph_execution_) {
    return names;
  }
  names.reserve(graph_execution_->plan.named_outputs.size());
  for (const auto& kv : graph_execution_->plan.named_outputs) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::string runtime::RunCore::last_error() const {
  std::lock_guard<std::mutex> lock(error_mu);
  if (!error.empty())
    return error;
  if (graph_execution_)
    return {};
  return pipeline.stream.last_error();
}

std::string runtime::RunCore::diagnostics_summary() const {
  if (graph_execution_)
    return last_error();
  return pipeline.stream.diagnostics_summary();
}

Run::operator bool() const noexcept {
  return core_ && core_->valid();
}

bool Run::can_push() const {
  return core_ && core_->can_push();
}

bool Run::can_pull() const {
  return core_ && core_->can_pull();
}

std::vector<std::string> Run::input_names() const {
  return core_ ? core_->input_names() : std::vector<std::string>{};
}

std::vector<std::string> Run::output_names() const {
  return core_ ? core_->output_names() : std::vector<std::string>{};
}

bool Run::running() const {
  return core_ && core_->running();
}

std::string Run::last_error() const {
  return core_ ? core_->last_error() : std::string{};
}

std::string Run::diagnostics_summary() const {
  return core_ ? core_->diagnostics_summary() : std::string{};
}

} // namespace simaai::neat

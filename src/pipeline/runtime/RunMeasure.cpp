#include "pipeline/Run.h"
#include "RunInternal.h"
#include "pipeline/LatencyProfiler.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

using Clock = std::chrono::steady_clock;

void append_plugin_latency_from_profiler(const ProfilerReport& profiler,
                                         std::vector<MeasurePluginLatency>* out) {
  if (!out) return;
  out->clear();
  out->reserve(profiler.kernel_aggregates.size());
  for (const auto& k : profiler.kernel_aggregates) {
    MeasurePluginLatency p;
    p.name = k.backend + ":" + (k.stage_name.empty() ? k.kernel_name : k.stage_name);
    p.calls = k.count;
    p.avg_ms = k.avg_ms();
    p.min_ms = k.min_ms;
    p.max_ms = k.max_ms;
    out->push_back(std::move(p));
  }
}

RunStats delta_counters(const RunStats& before, const RunStats& after) {
  RunStats out;
  out.inputs_enqueued = after.inputs_enqueued >= before.inputs_enqueued
                            ? after.inputs_enqueued - before.inputs_enqueued
                            : 0;
  out.inputs_dropped = after.inputs_dropped >= before.inputs_dropped
                           ? after.inputs_dropped - before.inputs_dropped
                           : 0;
  out.inputs_pushed = after.inputs_pushed >= before.inputs_pushed
                          ? after.inputs_pushed - before.inputs_pushed
                          : 0;
  out.outputs_ready = after.outputs_ready >= before.outputs_ready
                          ? after.outputs_ready - before.outputs_ready
                          : 0;
  out.outputs_pulled = after.outputs_pulled >= before.outputs_pulled
                           ? after.outputs_pulled - before.outputs_pulled
                           : 0;
  out.outputs_dropped = after.outputs_dropped >= before.outputs_dropped
                            ? after.outputs_dropped - before.outputs_dropped
                            : 0;
  return out;
}

MeasureLatencyStats summarize_samples(std::vector<double> samples) {
  MeasureLatencyStats out;
  out.count = samples.size();
  if (samples.empty()) return out;

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  out.avg_ms = sum / static_cast<double>(samples.size());
  std::sort(samples.begin(), samples.end());

  auto percentile = [&](double p) {
    if (samples.empty()) return 0.0;
    const double index = p * static_cast<double>(samples.size() - 1U);
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo + 1U, samples.size() - 1U);
    const double frac = index - static_cast<double>(lo);
    return samples[lo] + ((samples[hi] - samples[lo]) * frac);
  };

  out.p50_ms = percentile(0.50);
  out.p90_ms = percentile(0.90);
  out.p95_ms = percentile(0.95);
  out.p99_ms = percentile(0.99);
  out.max_ms = samples.back();
  return out;
}

void validate_options(const MeasureOptions& opt) {
  if (opt.duration_ms <= 0) throw std::runtime_error("MeasureOptions::duration_ms must be > 0");
  if (opt.warmup_ms < 0) throw std::runtime_error("MeasureOptions::warmup_ms must be >= 0");
  if (opt.timeout_ms <= 0) throw std::runtime_error("MeasureOptions::timeout_ms must be > 0");
}

void print_latency_row(std::ostream& os, const char* name, const MeasureLatencyStats& s) {
  os << std::left << std::setw(28) << name
     << std::right << std::setw(9) << s.count
     << std::setw(11) << s.avg_ms
     << std::setw(11) << s.p50_ms
     << std::setw(11) << s.p90_ms
     << std::setw(11) << s.p95_ms
     << std::setw(11) << s.p99_ms
     << std::setw(11) << s.max_ms << "\n";
}

} // namespace

struct MeasureScope::Impl {
  Run* run = nullptr;
  MeasureOptions options;
  RunStats before{};
  Clock::time_point start{};
  std::unique_ptr<LatencyProfiler> profiler;
  bool stopped = false;
  MeasureReport cached{};
};

MeasureScope::MeasureScope(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MeasureScope::MeasureScope(MeasureScope&&) noexcept = default;
MeasureScope& MeasureScope::operator=(MeasureScope&& other) noexcept {
  if (this != &other) {
    if (impl_ && !impl_->stopped && impl_->run && impl_->run->state_) {
      auto st = impl_->run->state_;
      std::lock_guard<std::mutex> lock(st->latency_mu);
      st->measurement_active = false;
      st->measurement_output_timing_init = false;
      st->measurement_latencies_ms.clear();
      st->measurement_frame_gaps_ms.clear();
    }
    impl_ = std::move(other.impl_);
  }
  return *this;
}
MeasureScope::~MeasureScope() {
  if (impl_ && !impl_->stopped && impl_->run && impl_->run->state_) {
    auto st = impl_->run->state_;
    std::lock_guard<std::mutex> lock(st->latency_mu);
    st->measurement_active = false;
    st->measurement_output_timing_init = false;
    st->measurement_latencies_ms.clear();
    st->measurement_frame_gaps_ms.clear();
  }
}

bool MeasureScope::stopped() const { return !impl_ || impl_->stopped; }

MeasureReport MeasureScope::stop() {
  if (!impl_) throw std::runtime_error("MeasureScope::stop: invalid moved-from scope");
  if (impl_->stopped) return impl_->cached;

  const auto end = Clock::now();
  const RunStats after = impl_->run->stats();
  const RunStats measured = delta_counters(impl_->before, after);

  MeasureReport report;
  report.options = impl_->options;
  if (report.options.logical_batch_size <= 0) report.options.logical_batch_size = 1;
  report.final_run_stats = after;
  report.inputs_pushed = measured.inputs_pushed;
  report.outputs_pulled = measured.outputs_pulled;
  report.inputs_dropped = measured.inputs_dropped;
  report.outputs_dropped = measured.outputs_dropped;
  report.outputs = static_cast<std::size_t>(measured.outputs_pulled);
  report.elapsed_s = std::chrono::duration<double>(end - impl_->start).count();
  report.throughput_batches_per_s = report.elapsed_s > 0.0
                                        ? static_cast<double>(report.outputs) / report.elapsed_s
                                        : 0.0;
  report.throughput_inferences_per_s =
      report.throughput_batches_per_s * static_cast<double>(report.options.logical_batch_size);
  std::vector<double> latency_samples;
  std::vector<double> frame_gap_samples;
  if (impl_->run->state_) {
    auto st = impl_->run->state_;
    std::lock_guard<std::mutex> lock(st->latency_mu);
    st->measurement_active = false;
    latency_samples = st->measurement_latencies_ms;
    frame_gap_samples = st->measurement_frame_gaps_ms;
    st->measurement_latencies_ms.clear();
    st->measurement_frame_gaps_ms.clear();
    st->measurement_output_timing_init = false;
  }
  report.end_to_end = summarize_samples(std::move(latency_samples));
  report.frame_gap = summarize_samples(std::move(frame_gap_samples));
  report.latency_samples_collected =
      report.end_to_end.count > 0 || report.frame_gap.count > 0;
  if (impl_->options.include_power) report.power = impl_->run->power_summary();
  if (impl_->profiler) {
    append_plugin_latency_from_profiler(impl_->profiler->finalize(), &report.plugin_latency);
    impl_->profiler.reset();
  }

  impl_->cached = report;
  impl_->stopped = true;
  return impl_->cached;
}

MeasureScope Run::start_measurement(const MeasureOptions& opt) {
  validate_options(opt);
  auto impl = std::unique_ptr<MeasureScope::Impl>(new MeasureScope::Impl());
  impl->run = this;
  impl->options = opt;
  impl->before = stats();
  impl->start = Clock::now();
  if (state_) {
    std::lock_guard<std::mutex> lock(state_->latency_mu);
    state_->measurement_latencies_ms.clear();
    state_->measurement_frame_gaps_ms.clear();
    state_->measurement_output_timing_init = false;
    state_->measurement_active = true;
  }
  if (opt.include_plugin_latency) {
    impl->profiler = std::make_unique<LatencyProfiler>();
    impl->profiler->attach(*this);
    impl->profiler->mark_warmup_done();
  }
  return MeasureScope(std::move(impl));
}

std::string MeasureReport::to_text() const {
  std::ostringstream os;
  os << std::fixed << std::setprecision(3);
  os << "\n============================================================\n"
     << " " << (options.title.empty() ? "NEAT measurement" : options.title) << "\n"
     << "============================================================\n";
  if (!options.model.empty()) os << "Model                 : " << options.model << "\n";
  if (options.logical_batch_size > 0) {
    os << "Compiled batch size   : " << options.logical_batch_size << "\n";
  }
  if (!options.input.empty()) os << "Input                 : " << options.input << "\n";
  if (!options.placement.empty()) os << "Placement             : " << options.placement << "\n";
  os << "Measure mode          : observer\n"
     << "Warmup window         : " << options.warmup_ms << " ms\n"
     << "Measured outputs      : " << outputs << "\n"
     << "Elapsed               : " << elapsed_s << " s\n"
     << "Throughput            : " << throughput_batches_per_s << " batches/s\n"
     << "Logical throughput    : " << throughput_inferences_per_s << " inferences/s\n";

  os << "\nLatency distribution (ms):\n";
  if (latency_samples_collected) {
    os << std::left << std::setw(28) << "metric"
       << std::right << std::setw(9) << "count"
       << std::setw(11) << "avg"
       << std::setw(11) << "p50"
       << std::setw(11) << "p90"
       << std::setw(11) << "p95"
       << std::setw(11) << "p99"
       << std::setw(11) << "max" << "\n";
    print_latency_row(os, "end-to-end push->output", end_to_end);
    print_latency_row(os, "between output frames", frame_gap);
  } else {
    os << "  no app-visible latency samples were produced during this observer window\n";
  }

  os << "\nPer-plugin / kernel latency during measured window (ms):\n";
  os << std::left << std::setw(34) << "plugin/kernel"
     << std::right << std::setw(9) << "calls"
     << std::setw(11) << "avg"
     << std::setw(11) << "min"
     << std::setw(11) << "max" << "\n";
  if (plugin_latency.empty()) {
    os << "  no plugin/kernel timing samples were reported\n";
  } else {
    for (const auto& p : plugin_latency) {
      os << std::left << std::setw(34) << p.name
         << std::right << std::setw(9) << p.calls
         << std::setw(11) << p.avg_ms
         << std::setw(11) << p.min_ms
         << std::setw(11) << p.max_ms << "\n";
    }
  }

  os << "\nNEAT counters during measured window:\n"
     << "  inputs_pushed        : " << inputs_pushed << "\n"
     << "  outputs_pulled       : " << outputs_pulled << "\n"
     << "  inputs_dropped       : " << inputs_dropped << "\n"
     << "  outputs_dropped      : " << outputs_dropped << "\n";

  if (!power.rails.empty()) {
    os << "\nPower telemetry:\n" << format_power_summary(power);
  }
  return os.str();
}

} // namespace simaai::neat

#include "pipeline/PowerTelemetry.h"

#include "pipeline/internal/PowerTelemetryInternal.h"
#include "pipeline/internal/SentinelClient.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

namespace simaai::neat {

namespace {

using nlohmann::json;

/// Sentinel samples every 2 s; polling twice as often adds each sample within 1 s.
constexpr std::chrono::seconds kSentinelPollInterval{1};

std::string hex_byte(std::uint8_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::nouppercase << static_cast<int>(value);
  return oss.str();
}

std::string escape_json(const std::string& input) {
  std::ostringstream oss;
  for (const char ch : input) {
    switch (ch) {
    case '\\':
      oss << "\\\\";
      break;
    case '"':
      oss << "\\\"";
      break;
    case '\n':
      oss << "\\n";
      break;
    case '\r':
      oss << "\\r";
      break;
    case '\t':
      oss << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20) {
        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(static_cast<unsigned char>(ch));
      } else {
        oss << ch;
      }
    }
  }
  return oss.str();
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::optional<std::string> read_device_tree_model() {
  constexpr const char* kModelPath = "/proc/device-tree/model";
  FILE* f = std::fopen(kModelPath, "rb");
  if (!f) {
    return std::nullopt;
  }
  std::string data;
  char buf[256];
  while (const auto n = std::fread(buf, 1, sizeof(buf), f)) {
    data.append(buf, buf + n);
  }
  std::fclose(f);
  data.erase(std::remove(data.begin(), data.end(), '\0'), data.end());
  if (data.empty()) {
    return std::nullopt;
  }
  return data;
}

/// Rail name of a Sentinel metric key such as `power_rail_mla_watts`; empty for other keys.
std::string sentinel_rail_name(std::string_view key) {
  constexpr std::string_view kPrefix = "power_rail_";
  constexpr std::string_view kSuffix = "_watts";
  if (key.size() <= kPrefix.size() + kSuffix.size() || key.substr(0, kPrefix.size()) != kPrefix ||
      key.substr(key.size() - kSuffix.size()) != kSuffix) {
    return {};
  }
  return std::string(key.substr(kPrefix.size(), key.size() - kPrefix.size() - kSuffix.size()));
}

/// Rail power readings of one sample from Sentinel's `GET /v1/samples/latest`.
PowerSnapshot power_snapshot_from_sentinel_sample(const json& sample) {
  PowerSnapshot snapshot;
  snapshot.timestamp = std::chrono::steady_clock::now();
  const json values = sample.value("values", json::object());
  for (const auto& [key, value] : values.items()) {
    PowerRailReading reading;
    reading.config.name = sentinel_rail_name(key);
    if (reading.config.name.empty()) {
      continue;
    }
    if (value.is_number()) {
      reading.power_w.available = true;
      reading.power_w.value = value.get<double>();
      snapshot.total_watts += reading.power_w.value;
      ++snapshot.rails_with_power;
    } else {
      reading.power_w.error = "Sentinel has no reading";
    }
    snapshot.rails.push_back(std::move(reading));
  }
  return snapshot;
}

struct FieldAgg {
  std::uint64_t samples = 0;
  std::uint64_t errors = 0;
  double sum = 0.0;
  double min = 0.0;
  double max = 0.0;

  void add(const PowerFieldReading& reading) {
    if (!reading.available) {
      if (!reading.error.empty()) {
        ++errors;
      }
      return;
    }
    if (samples == 0) {
      min = reading.value;
      max = reading.value;
    } else {
      min = std::min(min, reading.value);
      max = std::max(max, reading.value);
    }
    sum += reading.value;
    ++samples;
  }

  PowerFieldSummary summary() const {
    PowerFieldSummary out;
    out.samples = samples;
    out.errors = errors;
    if (samples > 0) {
      out.avg = sum / static_cast<double>(samples);
      out.min = min;
      out.max = max;
    }
    return out;
  }
};

struct RailAgg {
  PowerRailConfig config;
  FieldAgg power_w;
};

} // namespace

std::vector<PowerRailConfig> default_modalix_som_power_rails() {
  return {
      {"SoC and DDR_VDD 0.765V", 3, 0x4f, 0x00, -8, -3, -2},
      {"HDMI 1.2V", 3, 0x4f, 0x01, -8, -6, -5},
      {"SoC and DDR VDDQ 0.5V", 3, 0x4f, 0x02, -8, -6, -5},
      {"SoC and DDR VDD2H 1.05V", 3, 0x4f, 0x03, -8, -6, -5},
      {"MLA 0.68V", 3, 0x4d, 0x00, -8, -3, -2},
      {"PICe and ETH VP 0.9V", 3, 0x4d, 0x01, -8, -6, -5},
      {"SOM Platform Rail 1.8V", 3, 0x4d, 0x02, -8, -6, -5},
      {"SOM Platform Rail 3.3V", 3, 0x4d, 0x03, -8, -6, -5},
  };
}

std::vector<PowerRailConfig> default_modalix_dvt_power_rails() {
  return {{"DVT PMIC 0x4d page0", 4, 0x4d, 0x00, -8, -3, -2}};
}

PowerMonitorProfile detect_default_power_monitor_profile() {
  const auto model = read_device_tree_model();
  if (model && lowercase_ascii(*model).find("modalix dvt") != std::string::npos) {
    return PowerMonitorProfile::ModalixDvt;
  }
  return PowerMonitorProfile::ModalixSom;
}

std::string power_monitor_profile_name(PowerMonitorProfile profile) {
  switch (profile) {
  case PowerMonitorProfile::Auto:
    return "auto";
  case PowerMonitorProfile::ModalixSom:
    return "modalix_som";
  case PowerMonitorProfile::ModalixDvt:
    return "modalix_dvt";
  case PowerMonitorProfile::Custom:
    return "custom";
  }
  return "unknown";
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
std::vector<PowerRailConfig> power_rails_for_profile(PowerMonitorProfile profile) {
  switch (profile) {
  case PowerMonitorProfile::Auto:
    return power_rails_for_profile(detect_default_power_monitor_profile());
  case PowerMonitorProfile::ModalixSom:
    return default_modalix_som_power_rails();
  case PowerMonitorProfile::ModalixDvt:
    return default_modalix_dvt_power_rails();
  case PowerMonitorProfile::Custom:
    return {};
  }
  return {};
}
#pragma GCC diagnostic pop

PowerMonitorOptions board_power_monitor_options(int sample_interval_ms,
                                                PowerMonitorProfile profile) {
  PowerMonitorOptions options;
  options.enabled = true;
  options.sample_interval_ms = std::max(1, sample_interval_ms);
  options.profile = profile;
  return options;
}

PowerMonitorOptions modalix_som_power_monitor_options(int sample_interval_ms) {
  return board_power_monitor_options(sample_interval_ms, PowerMonitorProfile::ModalixSom);
}

PowerMonitorOptions modalix_dvt_power_monitor_options(int sample_interval_ms) {
  return board_power_monitor_options(sample_interval_ms, PowerMonitorProfile::ModalixDvt);
}

PowerSnapshot read_power_snapshot(const PowerMonitorOptions& /*options*/) {
  return power_snapshot_from_sentinel_sample(pipeline_internal::SentinelClient().latest_sample());
}

class PowerMonitor::Impl {
public:
  Impl(PowerMonitorOptions options, pipeline_internal::SentinelClient sentinel)
      : options_(std::move(options)), sentinel_(std::move(sentinel)) {}

  ~Impl() {
    stop();
  }

  void start() {
    if (!options_.enabled)
      return;
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
      return;
    // Sentinel's current sample predates the interval; only later samples are added.
    const std::optional<json> baseline = read_latest_sample();
    {
      std::lock_guard<std::mutex> lock(mu_);
      start_time_ = std::chrono::steady_clock::now();
      end_time_ = {};
      if (baseline) {
        last_timestamp_ = baseline->value("timestamp", "");
      }
    }
    worker_ = std::thread([this]() { run_loop(); });
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    sample_once();
    std::lock_guard<std::mutex> lock(mu_);
    end_time_ = std::chrono::steady_clock::now();
  }

  void sample_once() {
    if (!options_.enabled)
      return;
    const std::optional<json> sample = read_latest_sample();
    if (!sample) {
      return;
    }
    const std::string timestamp = sample->value("timestamp", "");
    std::lock_guard<std::mutex> lock(mu_);
    if (timestamp == last_timestamp_) {
      return;
    }
    last_timestamp_ = timestamp;
    const PowerSnapshot snapshot = power_snapshot_from_sentinel_sample(*sample);
    if (start_time_ == std::chrono::steady_clock::time_point{}) {
      start_time_ = snapshot.timestamp;
    }
    if (!running_.load()) {
      end_time_ = snapshot.timestamp;
    }
    add_snapshot_locked(snapshot);
  }

  PowerSummary summary() const {
    std::lock_guard<std::mutex> lock(mu_);
    PowerSummary out;
    out.enabled = options_.enabled;
    out.samples = samples_;
    if (start_time_ != std::chrono::steady_clock::time_point{}) {
      const auto end = end_time_ == std::chrono::steady_clock::time_point{}
                           ? std::chrono::steady_clock::now()
                           : end_time_;
      out.duration_seconds = std::chrono::duration<double>(end - start_time_).count();
    }
    if (total_samples_ > 0) {
      out.total_avg_watts = total_sum_watts_ / static_cast<double>(total_samples_);
      out.total_min_watts = total_min_watts_;
      out.total_max_watts = total_max_watts_;
      out.energy_joules = out.total_avg_watts * std::max(0.0, out.duration_seconds);
    }
    out.rails.reserve(rail_aggs_.size());
    for (const auto& agg : rail_aggs_) {
      PowerRailSummary rail;
      rail.config = agg.config;
      rail.power_w = agg.power_w.summary();
      out.rails.push_back(std::move(rail));
    }
    return out;
  }

  bool running() const {
    return running_.load();
  }

private:
  /// Sentinel's latest sample; warns once and returns nothing when Sentinel cannot be read.
  std::optional<json> read_latest_sample() {
    try {
      return sentinel_.latest_sample();
    } catch (const std::exception& e) {
      if (!warned_.exchange(true)) {
        std::cerr << "[WARN] Power telemetry is unavailable: " << e.what() << "\n";
      }
      return std::nullopt;
    }
  }

  void run_loop() {
    while (running_.load()) {
      {
        std::unique_lock<std::mutex> lock(wait_mu_);
        cv_.wait_for(lock, kSentinelPollInterval, [this]() { return !running_.load(); });
      }
      if (!running_.load())
        break;
      sample_once();
    }
  }

  void add_snapshot_locked(const PowerSnapshot& snapshot) {
    ++samples_;
    if (snapshot.rails_with_power > 0) {
      if (total_samples_ == 0) {
        total_min_watts_ = snapshot.total_watts;
        total_max_watts_ = snapshot.total_watts;
      } else {
        total_min_watts_ = std::min(total_min_watts_, snapshot.total_watts);
        total_max_watts_ = std::max(total_max_watts_, snapshot.total_watts);
      }
      total_sum_watts_ += snapshot.total_watts;
      ++total_samples_;
    }
    for (const PowerRailReading& reading : snapshot.rails) {
      auto agg = std::find_if(rail_aggs_.begin(), rail_aggs_.end(), [&](const RailAgg& rail) {
        return rail.config.name == reading.config.name;
      });
      if (agg == rail_aggs_.end()) {
        agg = rail_aggs_.insert(rail_aggs_.end(), RailAgg{reading.config, {}});
      }
      agg->power_w.add(reading.power_w);
    }
  }

  PowerMonitorOptions options_;
  pipeline_internal::SentinelClient sentinel_;
  mutable std::mutex mu_;
  std::mutex wait_mu_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<bool> warned_{false};
  std::chrono::steady_clock::time_point start_time_{};
  std::chrono::steady_clock::time_point end_time_{};
  std::string last_timestamp_;
  std::uint64_t samples_ = 0;
  std::uint64_t total_samples_ = 0;
  double total_sum_watts_ = 0.0;
  double total_min_watts_ = 0.0;
  double total_max_watts_ = 0.0;
  std::vector<RailAgg> rail_aggs_;
};

PowerMonitor::PowerMonitor(PowerMonitorOptions options)
    : impl_(std::make_unique<Impl>(std::move(options), pipeline_internal::SentinelClient())) {}

PowerMonitor::~PowerMonitor() = default;
PowerMonitor::PowerMonitor(PowerMonitor&&) noexcept = default;
PowerMonitor& PowerMonitor::operator=(PowerMonitor&&) noexcept = default;

PowerMonitor internal::PowerMonitorAccess::make(PowerMonitorOptions options,
                                                std::string sentinel_socket_path) {
  PowerMonitor monitor;
  monitor.impl_ = std::make_unique<PowerMonitor::Impl>(
      std::move(options), pipeline_internal::SentinelClient(std::move(sentinel_socket_path)));
  return monitor;
}

void PowerMonitor::start() {
  if (impl_)
    impl_->start();
}

void PowerMonitor::stop() {
  if (impl_)
    impl_->stop();
}

void PowerMonitor::sample_once() {
  if (impl_)
    impl_->sample_once();
}

PowerSummary PowerMonitor::summary() const {
  return impl_ ? impl_->summary() : PowerSummary{};
}

bool PowerMonitor::running() const {
  return impl_ && impl_->running();
}

std::string format_power_summary(const PowerSummary& summary) {
  if (!summary.enabled) {
    return {};
  }
  std::ostringstream oss;
  oss << "PowerStats: samples=" << summary.samples << " duration_s=" << summary.duration_seconds
      << " total_avg_w=" << summary.total_avg_watts << " total_min_w=" << summary.total_min_watts
      << " total_max_w=" << summary.total_max_watts << " energy_j=" << summary.energy_joules
      << "\n";
  for (const auto& rail : summary.rails) {
    oss << "  RailPower: name=\"" << rail.config.name
        << "\" addr=" << hex_byte(rail.config.i2c_addr) << " page=" << hex_byte(rail.config.page)
        << " v_avg=" << rail.voltage_v.avg << " i_avg=" << rail.current_a.avg
        << " p_avg_w=" << rail.power_w.avg << " p_min_w=" << rail.power_w.min
        << " p_max_w=" << rail.power_w.max << " p_samples=" << rail.power_w.samples
        << " errors=" << (rail.voltage_v.errors + rail.current_a.errors + rail.power_w.errors)
        << "\n";
  }
  return oss.str();
}

std::string power_summary_to_json(const PowerSummary& summary, int indent) {
  const std::string base(std::max(0, indent), ' ');
  const std::string inner(std::max(0, indent + 2), ' ');
  const std::string rail_indent(std::max(0, indent + 4), ' ');
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{\n";
  oss << inner << "\"enabled\": " << (summary.enabled ? "true" : "false") << ",\n";
  oss << inner << "\"samples\": " << summary.samples << ",\n";
  oss << inner << "\"duration_seconds\": " << summary.duration_seconds << ",\n";
  oss << inner << "\"total_avg_watts\": " << summary.total_avg_watts << ",\n";
  oss << inner << "\"total_min_watts\": " << summary.total_min_watts << ",\n";
  oss << inner << "\"total_max_watts\": " << summary.total_max_watts << ",\n";
  oss << inner << "\"energy_joules\": " << summary.energy_joules << ",\n";
  oss << inner << "\"rails\": [";
  if (!summary.rails.empty())
    oss << "\n";
  for (std::size_t i = 0; i < summary.rails.size(); ++i) {
    const auto& rail = summary.rails[i];
    oss << rail_indent << "{" << "\"name\": \"" << escape_json(rail.config.name) << "\", "
        << "\"i2c_addr\": \"" << hex_byte(rail.config.i2c_addr) << "\", " << "\"page\": \""
        << hex_byte(rail.config.page) << "\", " << "\"voltage_avg_v\": " << rail.voltage_v.avg
        << ", " << "\"current_avg_a\": " << rail.current_a.avg << ", "
        << "\"power_avg_watts\": " << rail.power_w.avg << ", "
        << "\"power_min_watts\": " << rail.power_w.min << ", "
        << "\"power_max_watts\": " << rail.power_w.max << ", "
        << "\"power_samples\": " << rail.power_w.samples << ", "
        << "\"errors\": " << (rail.voltage_v.errors + rail.current_a.errors + rail.power_w.errors)
        << "}";
    if (i + 1 != summary.rails.size())
      oss << ",";
    oss << "\n";
  }
  if (!summary.rails.empty())
    oss << inner;
  oss << "]\n" << base << "}";
  return oss.str();
}

} // namespace simaai::neat

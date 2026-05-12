#include "pipeline/PowerTelemetry.h"

#include "pipeline/internal/PowerTelemetryInternal.h"

#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

namespace simaai::neat {

namespace {

constexpr std::uint8_t kPageReg = 0x00;
constexpr std::uint8_t kPoutReg = 0x96;

double scale_raw(std::uint8_t raw, int exponent) {
  return static_cast<double>(raw) * std::ldexp(1.0, exponent);
}

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

PowerMonitorProfile detect_profile_from_model_text(const std::string& model) {
  const std::string normalized = lowercase_ascii(model);
  if (normalized.find("modalix dvt") != std::string::npos) {
    return PowerMonitorProfile::ModalixDvt;
  }
  if (normalized.find("modalix som") != std::string::npos) {
    return PowerMonitorProfile::ModalixSom;
  }
  return PowerMonitorProfile::ModalixSom;
}

std::vector<PowerRailConfig> modalix_dvt_power_rails_impl() {
  // Modalix DVT PMBus access has proven fragile when probing the full PMIC set
  // or when reading multiple telemetry registers per page.  The stable board
  // path verified on DVT is the single page-0 POUT byte read from PMIC 0x4d on
  // mux bus 4: raw register 0x96 scaled by 2^-2 watts.
  return {{"DVT PMIC 0x4d page0", 4, 0x4d, 0x00, -8, -3, -2}};
}

std::vector<PowerRailConfig> resolved_power_rails(const PowerMonitorOptions& options) {
  if (!options.rails.empty()) {
    return options.rails;
  }
  return power_rails_for_profile(options.profile);
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
  FieldAgg voltage_v;
  FieldAgg current_a;
  FieldAgg power_w;

  explicit RailAgg(PowerRailConfig c) : config(std::move(c)) {}
};

} // namespace

namespace pipeline_internal {

namespace {

class NativePowerI2cBackend final : public PowerI2cBackend {
public:
  bool set_page(int bus, std::uint8_t addr, std::uint8_t page, std::string& error) override {
    const int fd = open_device(bus, addr, error);
    if (fd < 0)
      return false;
    const bool ok = smbus_write_byte_data(fd, kPageReg, page, error);
    if (!ok) {
      error = "i2c write page " + hex_byte(page) + " failed: " + error;
    }
    ::close(fd);
    return ok;
  }

  bool read_register(int bus, std::uint8_t addr, std::uint8_t reg, std::uint8_t& value,
                     std::string& error) override {
    const int fd = open_device(bus, addr, error);
    if (fd < 0)
      return false;
    const bool read_ok = smbus_read_byte_data(fd, reg, value, error);
    if (!read_ok) {
      error = "i2c read register " + hex_byte(reg) + " failed: " + error;
    }
    ::close(fd);
    return read_ok;
  }

private:
  union SmbusData {
    __u8 byte;
    __u16 word;
    __u8 block[I2C_SMBUS_BLOCK_MAX + 2];
  };

  static int smbus_access(int fd, char read_write, std::uint8_t command, int size,
                          SmbusData* data) {
    struct i2c_smbus_ioctl_data args {};
    args.read_write = read_write;
    args.command = command;
    args.size = size;
    args.data = reinterpret_cast<union i2c_smbus_data*>(data);
    return ::ioctl(fd, I2C_SMBUS, &args);
  }

  static bool smbus_write_byte_data(int fd, std::uint8_t reg, std::uint8_t value,
                                    std::string& error) {
    SmbusData data{};
    data.byte = value;
    if (smbus_access(fd, I2C_SMBUS_WRITE, reg, I2C_SMBUS_BYTE_DATA, &data) < 0) {
      error = std::strerror(errno);
      return false;
    }
    return true;
  }

  static bool smbus_read_byte_data(int fd, std::uint8_t reg, std::uint8_t& value,
                                   std::string& error) {
    SmbusData data{};
    if (smbus_access(fd, I2C_SMBUS_READ, reg, I2C_SMBUS_BYTE_DATA, &data) < 0) {
      error = std::strerror(errno);
      return false;
    }
    value = data.byte;
    return true;
  }

  static int open_device(int bus, std::uint8_t addr, std::string& error) {
    const std::string path = "/dev/i2c-" + std::to_string(bus);
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      error = "open " + path + " failed: " + std::strerror(errno);
      return -1;
    }
    if (::ioctl(fd, I2C_SLAVE, static_cast<int>(addr)) < 0) {
      error = "I2C_SLAVE " + hex_byte(addr) + " failed: " + std::strerror(errno);
      ::close(fd);
      return -1;
    }
    return fd;
  }
};

} // namespace

std::shared_ptr<PowerI2cBackend> make_native_power_i2c_backend() {
  return std::make_shared<NativePowerI2cBackend>();
}

PowerSnapshot read_power_snapshot_with_backend(const PowerMonitorOptions& options,
                                               const std::shared_ptr<PowerI2cBackend>& backend) {
  PowerSnapshot snapshot;
  snapshot.timestamp = std::chrono::steady_clock::now();
  if (!backend)
    return snapshot;

  std::vector<PowerRailConfig> rails = options.rails;
  if (rails.empty()) {
    rails = resolved_power_rails(options);
  }
  snapshot.rails.reserve(rails.size());

  for (const auto& rail : rails) {
    PowerRailReading reading;
    reading.config = rail;

    std::string error;
    if (!backend->set_page(rail.i2c_bus, rail.i2c_addr, rail.page, error)) {
      const std::string detail = "set page failed for " + rail.name + ": " + error;
      reading.voltage_v.error = detail;
      reading.current_a.error = detail;
      reading.power_w.error = detail;
      snapshot.rails.push_back(std::move(reading));
      continue;
    }

    std::uint8_t raw = 0;
    std::string reg_error;
    if (!backend->read_register(rail.i2c_bus, rail.i2c_addr, kPoutReg, raw, reg_error)) {
      reading.power_w.error = reg_error;
      snapshot.rails.push_back(std::move(reading));
      continue;
    }
    reading.power_w.available = true;
    reading.power_w.raw = raw;
    reading.power_w.value = scale_raw(raw, rail.pout_exponent);
    if (reading.power_w.available) {
      snapshot.total_watts += reading.power_w.value;
      ++snapshot.rails_with_power;
    }
    snapshot.rails.push_back(std::move(reading));
  }

  return snapshot;
}

} // namespace pipeline_internal

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
  return modalix_dvt_power_rails_impl();
}

PowerMonitorProfile detect_default_power_monitor_profile() {
  const auto model = read_device_tree_model();
  if (!model) {
    return PowerMonitorProfile::ModalixSom;
  }
  return detect_profile_from_model_text(*model);
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

PowerMonitorOptions board_power_monitor_options(int sample_interval_ms,
                                                PowerMonitorProfile profile) {
  PowerMonitorOptions options;
  options.enabled = true;
  options.sample_interval_ms = std::max(1, sample_interval_ms);
  const PowerMonitorProfile resolved =
      (profile == PowerMonitorProfile::Auto) ? detect_default_power_monitor_profile() : profile;
  options.profile = resolved;
  options.rails = power_rails_for_profile(resolved);
  return options;
}

PowerMonitorOptions modalix_som_power_monitor_options(int sample_interval_ms) {
  return board_power_monitor_options(sample_interval_ms, PowerMonitorProfile::ModalixSom);
}

PowerMonitorOptions modalix_dvt_power_monitor_options(int sample_interval_ms) {
  return board_power_monitor_options(sample_interval_ms, PowerMonitorProfile::ModalixDvt);
}

PowerSnapshot read_power_snapshot(const PowerMonitorOptions& options) {
  return pipeline_internal::read_power_snapshot_with_backend(
      options, pipeline_internal::make_native_power_i2c_backend());
}

class PowerMonitor::Impl {
public:
  explicit Impl(PowerMonitorOptions options)
      : options_(std::move(options)), backend_(pipeline_internal::make_native_power_i2c_backend()) {
    if (options_.rails.empty()) {
      options_.rails = resolved_power_rails(options_);
    }
    rail_aggs_.reserve(options_.rails.size());
    for (const auto& rail : options_.rails) {
      rail_aggs_.emplace_back(rail);
    }
  }

  ~Impl() {
    stop();
  }

  void start() {
    if (!options_.enabled)
      return;
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
      return;
    {
      std::lock_guard<std::mutex> lock(mu_);
      start_time_ = std::chrono::steady_clock::now();
      end_time_ = {};
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
    std::lock_guard<std::mutex> lock(mu_);
    end_time_ = std::chrono::steady_clock::now();
  }

  void sample_once() {
    if (!options_.enabled)
      return;
    const PowerSnapshot snapshot =
        pipeline_internal::read_power_snapshot_with_backend(options_, backend_);
    std::lock_guard<std::mutex> lock(mu_);
    if (samples_ == 0 && start_time_ == std::chrono::steady_clock::time_point{}) {
      start_time_ = snapshot.timestamp;
    }
    add_snapshot_locked(snapshot);
    end_time_ = snapshot.timestamp;
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
      rail.voltage_v = agg.voltage_v.summary();
      rail.current_a = agg.current_a.summary();
      rail.power_w = agg.power_w.summary();
      out.rails.push_back(std::move(rail));
    }
    return out;
  }

  bool running() const {
    return running_.load();
  }

private:
  void run_loop() {
    sample_once();
    const int interval = std::max(1, options_.sample_interval_ms);
    while (running_.load()) {
      std::unique_lock<std::mutex> lock(wait_mu_);
      cv_.wait_for(lock, std::chrono::milliseconds(interval),
                   [this]() { return !running_.load(); });
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
    const std::size_t n = std::min(rail_aggs_.size(), snapshot.rails.size());
    for (std::size_t i = 0; i < n; ++i) {
      rail_aggs_[i].voltage_v.add(snapshot.rails[i].voltage_v);
      rail_aggs_[i].current_a.add(snapshot.rails[i].current_a);
      rail_aggs_[i].power_w.add(snapshot.rails[i].power_w);
    }
  }

  PowerMonitorOptions options_;
  std::shared_ptr<pipeline_internal::PowerI2cBackend> backend_;
  mutable std::mutex mu_;
  std::mutex wait_mu_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  std::chrono::steady_clock::time_point start_time_{};
  std::chrono::steady_clock::time_point end_time_{};
  std::uint64_t samples_ = 0;
  std::uint64_t total_samples_ = 0;
  double total_sum_watts_ = 0.0;
  double total_min_watts_ = 0.0;
  double total_max_watts_ = 0.0;
  std::vector<RailAgg> rail_aggs_;
};

PowerMonitor::PowerMonitor(PowerMonitorOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

PowerMonitor::~PowerMonitor() = default;
PowerMonitor::PowerMonitor(PowerMonitor&&) noexcept = default;
PowerMonitor& PowerMonitor::operator=(PowerMonitor&&) noexcept = default;

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
    oss << rail_indent << "{"
        << "\"name\": \"" << escape_json(rail.config.name) << "\", "
        << "\"i2c_addr\": \"" << hex_byte(rail.config.i2c_addr) << "\", "
        << "\"page\": \"" << hex_byte(rail.config.page) << "\", "
        << "\"voltage_avg_v\": " << rail.voltage_v.avg << ", "
        << "\"current_avg_a\": " << rail.current_a.avg << ", "
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

#include "model/internal/BenchmarkTelemetry.h"

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace simaai::neat::internal {
namespace {

using nlohmann::json;

constexpr int kSentinelTimeoutSeconds = 5;
constexpr int kDieSensorCount = 7; // rtsn_0..6; rtsn_7..13 are trip channels for the same sites.

struct SentinelReply {
  int status = 0;
  json body;
};

std::string errno_text() {
  return std::generic_category().message(errno);
}

/// One HTTP/1.1 request over Sentinel's Unix socket. Sentinel closes the connection after
/// every reply, so the response is read to EOF.
SentinelReply sentinel_request_once(const std::string& socket_path, const std::string& method,
                                    const std::string& path, const std::string& payload) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    throw std::runtime_error("Sentinel socket path is too long: " + socket_path);
  }
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);

  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    throw std::runtime_error("cannot create a socket for Sentinel: " + errno_text());
  }
  struct Closer {
    int fd;
    ~Closer() {
      ::close(fd);
    }
  } closer{fd};
  const timeval timeout{kSentinelTimeoutSeconds, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    throw std::runtime_error("Sentinel is not reachable at " + socket_path + ": " + errno_text());
  }

  const std::string request = method + " " + path +
                              " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                              "Content-Length: " +
                              std::to_string(payload.size()) + "\r\nConnection: close\r\n\r\n" +
                              payload;
  for (std::size_t sent = 0; sent < request.size();) {
    const ssize_t n = ::send(fd, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0) {
      throw std::runtime_error("cannot send a request to Sentinel: " + errno_text());
    }
    sent += static_cast<std::size_t>(n);
  }

  std::string response;
  char buffer[8192];
  for (;;) {
    const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
    if (n == 0) {
      break;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      throw std::runtime_error("Sentinel did not answer " + method + " " + path + " within " +
                               std::to_string(kSentinelTimeoutSeconds) + " s");
    }
    if (n < 0) {
      throw std::runtime_error("cannot read the Sentinel reply: " + errno_text());
    }
    response.append(buffer, static_cast<std::size_t>(n));
  }

  const std::size_t header_end = response.find("\r\n\r\n");
  if (response.rfind("HTTP/1.", 0) != 0 || header_end == std::string::npos) {
    throw std::runtime_error("Sentinel sent a malformed reply to " + method + " " + path);
  }
  SentinelReply reply;
  reply.status = std::atoi(response.c_str() + response.find(' ') + 1);
  reply.body = json::parse(response.begin() + static_cast<std::ptrdiff_t>(header_end) + 4,
                           response.end(), nullptr, /*allow_exceptions=*/false);
  if (reply.body.is_discarded()) {
    throw std::runtime_error("Sentinel sent invalid JSON for " + method + " " + path);
  }
  return reply;
}

SentinelReply sentinel_request(const std::string& socket_path, const std::string& method,
                               const std::string& path, const std::string& payload = {}) {
  SentinelReply reply = sentinel_request_once(socket_path, method, path, payload);
  if (reply.status == 500) {
    // Sentinel can fail transiently while it rewrites its sample cache.
    reply = sentinel_request_once(socket_path, method, path, payload);
  }
  return reply;
}

const json& field(const json& object, const char* key) {
  static const json kNull;
  if (!object.is_object()) {
    return kNull;
  }
  const auto it = object.find(key);
  return it == object.end() ? kNull : *it;
}

json require_ok(SentinelReply reply, const std::string& action) {
  if (reply.status != 200) {
    const json& error = field(reply.body, "error");
    throw std::runtime_error("Sentinel could not " + action + " (HTTP " +
                             std::to_string(reply.status) +
                             (error.is_string() ? "): " + error.get<std::string>() : ")"));
  }
  return std::move(reply.body);
}

std::string benchmark_trace_name() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm utc{};
  ::gmtime_r(&seconds, &utc);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%S", &utc);
  char name[64];
  std::snprintf(name, sizeof(name), "neat-benchmark-%s.%03dZ", stamp, static_cast<int>(millis));
  return name;
}

std::optional<double> number(const json& values, const std::string& key) {
  const json& value = field(values, key.c_str());
  if (!value.is_number()) {
    return std::nullopt;
  }
  const double result = value.get<double>();
  return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}

BenchmarkMetric summarize(std::string key, std::string label, std::string unit,
                          const std::vector<double>& values) {
  BenchmarkMetric metric;
  metric.key = std::move(key);
  metric.label = std::move(label);
  metric.unit = std::move(unit);
  if (values.empty()) {
    return metric;
  }
  metric.samples = static_cast<int>(values.size());
  metric.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  metric.minimum = *std::min_element(values.begin(), values.end());
  metric.maximum = *std::max_element(values.begin(), values.end());
  return metric;
}

/// Read access to a saved Sentinel run: metric definitions plus samples.
class SentinelRun {
public:
  explicit SentinelRun(const json& run) : run_(run) {}

  const json& samples() const {
    return field(run_, "samples");
  }
  const json& definitions() const {
    return field(run_, "metrics");
  }

  std::vector<double> values(const std::string& key) const {
    std::vector<double> out;
    if (samples().is_array()) {
      for (const json& sample : samples()) {
        if (const auto value = number(field(sample, "values"), key)) {
          out.push_back(*value);
        }
      }
    }
    return out;
  }

  std::string definition(const std::string& key, const char* attribute) const {
    if (definitions().is_array()) {
      for (const json& metric : definitions()) {
        const json& value = field(metric, attribute);
        if (field(metric, "key") == key && value.is_string()) {
          return value.get<std::string>();
        }
      }
    }
    return {};
  }

  BenchmarkMetric metric(const std::string& key) const {
    std::string label = definition(key, "label");
    return summarize(key, label.empty() ? key : std::move(label), definition(key, "unit"),
                     values(key));
  }

  std::vector<BenchmarkMetric> metrics(const std::vector<std::string>& keys) const {
    std::vector<BenchmarkMetric> out;
    for (const std::string& key : keys) {
      out.push_back(metric(key));
    }
    return out;
  }

  std::vector<std::string> keys_like(const std::string& prefix, const std::string& suffix) const {
    std::vector<std::string> out;
    if (definitions().is_array()) {
      for (const json& metric : definitions()) {
        const json& key = field(metric, "key");
        if (!key.is_string()) {
          continue;
        }
        const std::string& text = key.get_ref<const std::string&>();
        if (text.size() > prefix.size() + suffix.size() && text.rfind(prefix, 0) == 0 &&
            text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
          out.push_back(text);
        }
      }
    }
    return out;
  }

  BenchmarkMetric hottest(const std::vector<std::string>& keys, std::string key,
                          std::string label) const {
    std::vector<double> per_sample;
    if (samples().is_array()) {
      for (const json& sample : samples()) {
        std::optional<double> hottest;
        for (const std::string& sensor : keys) {
          if (const auto value = number(field(sample, "values"), sensor)) {
            hottest = std::max(hottest.value_or(*value), *value);
          }
        }
        if (hottest) {
          per_sample.push_back(*hottest);
        }
      }
    }
    return summarize(std::move(key), std::move(label), definition(keys.front(), "unit"),
                     per_sample);
  }

private:
  const json& run_;
};

} // namespace

BenchmarkTelemetry summarize_benchmark_telemetry(const json& run) {
  const SentinelRun saved(run);
  const json& metadata = field(run, "metadata");

  BenchmarkTelemetry telemetry;
  telemetry.samples = saved.samples().is_array() ? static_cast<int>(saved.samples().size()) : 0;
  telemetry.status = telemetry.samples > 0
                         ? "ok"
                         : "Sentinel recorded no samples during the throughput window; "
                           "increase min_duration_ms";
  if (field(metadata, "id").is_string()) {
    telemetry.trace_id = field(metadata, "id").get<std::string>();
  }
  if (field(metadata, "sentinel_version").is_string()) {
    telemetry.sentinel_version = field(metadata, "sentinel_version").get<std::string>();
  }
  telemetry.sample_interval_ms =
      static_cast<int>(number(metadata, "sample_interval_ms").value_or(0.0));

  std::vector<std::string> die_sensors;
  for (int i = 0; i < kDieSensorCount; ++i) {
    die_sensors.push_back("rtsn_" + std::to_string(i));
  }

  telemetry.power_som_watts = saved.metric("power_current_watts");
  telemetry.power_mla_watts = saved.metric("power_rail_mla_watts");
  telemetry.power_rails_watts = saved.metrics(saved.keys_like("power_rail_", "_watts"));
  telemetry.die_temperatures_c = saved.metrics(die_sensors);
  telemetry.die_temperature_max_c = saved.hottest(die_sensors, "die_max", "Die max");
  telemetry.board_temperatures_c = saved.metrics({"lm96163_temp1", "lm96163_temp2"});
  telemetry.cpu_usage_pct = saved.metric("cpu_usage_pct");
  // Sentinel converts memory to MiB but labels it "MB".
  const auto memory = [&](const char* key) {
    BenchmarkMetric metric = saved.metric(key);
    metric.unit = "MiB";
    return metric;
  };
  telemetry.mla_memory_mib = memory("mla_mem_allocated_mb");
  telemetry.cma_memory_mib = memory("ev74_cma_used_mb");
  telemetry.linux_memory_used_mib = memory("linux_mem_used_mb");
  return telemetry;
}

BenchmarkTelemetryRecorder::BenchmarkTelemetryRecorder(std::string note, std::string socket_path)
    : note_(std::move(note)), socket_path_(std::move(socket_path)) {}

BenchmarkTelemetryRecorder::~BenchmarkTelemetryRecorder() {
  stop();
}

template <typename Step> void BenchmarkTelemetryRecorder::attempt(Step&& step) {
  if (!failure_.empty()) {
    return;
  }
  try {
    step();
  } catch (const std::exception& e) {
    failure_ = e.what();
  }
}

void BenchmarkTelemetryRecorder::start() {
  attempt([&] {
    const json request = {{"name", benchmark_trace_name()},
                          {"note", note_},
                          {"tags", json::array({"neat-benchmark"})}};
    const json reply = require_ok(
        sentinel_request(socket_path_, "POST", "/v1/traces", request.dump()), "start a trace");
    trace_id_ = field(field(reply, "trace"), "id").get<std::string>();
    recording_ = true;
  });
}

void BenchmarkTelemetryRecorder::stop() noexcept {
  if (!recording_) {
    return;
  }
  recording_ = false;
  try {
    SentinelReply reply =
        sentinel_request(socket_path_, "POST", "/v1/traces/" + trace_id_ + "/stop");
    if (reply.status == 404) {
      // Sentinel 0.1.0 has no stop-by-ID endpoint; the trace it holds is the one started here.
      reply = sentinel_request(socket_path_, "POST", "/v1/traces/stop");
    }
    require_ok(std::move(reply), "stop trace " + trace_id_);
  } catch (const std::exception& e) {
    if (failure_.empty()) {
      failure_ = e.what();
    }
  }
}

BenchmarkTelemetry BenchmarkTelemetryRecorder::collect() {
  stop();
  BenchmarkTelemetry telemetry;
  attempt([&] {
    const json run = require_ok(sentinel_request(socket_path_, "GET", "/v1/runs/" + trace_id_),
                                "read trace " + trace_id_);
    telemetry = summarize_benchmark_telemetry(run);
  });
  if (!failure_.empty()) {
    telemetry = BenchmarkTelemetry{};
    telemetry.status = failure_;
    telemetry.trace_id = trace_id_;
  }
  return telemetry;
}

} // namespace simaai::neat::internal

#include "pipeline/internal/SentinelClient.h"

#include <httplib.h>
#include <sys/socket.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

using nlohmann::json;

/// GET `path`; throw when Sentinel does not answer.
httplib::Result get(const std::string& socket_path, std::chrono::seconds timeout,
                    const std::string& path) {
  httplib::Client client(socket_path);
  client.set_address_family(AF_UNIX);
  client.set_connection_timeout(timeout);
  client.set_read_timeout(timeout);
  client.set_write_timeout(timeout);
  httplib::Result result = client.Get(path);
  if (result) {
    return result;
  }
  if (result.error() == httplib::Error::Connection) {
    throw std::runtime_error(
        "Sentinel is not running on this board, so no telemetry is recorded. Install it on the "
        "board with 'sima-cli neat install sentinel'. If it is already installed, start it with "
        "'sudo systemctl start simaai-sentinel'.");
  }
  throw std::runtime_error("Sentinel did not answer GET " + path + ": " +
                           httplib::to_string(result.error()));
}

/// Return the JSON object of a 200 reply; otherwise throw with Sentinel's error message.
json expect_ok(const httplib::Result& result, const std::string& action) {
  if (result->status == 404) {
    throw std::runtime_error("Sentinel is too old to " + action +
                             ". Upgrade it on the board with 'sima-cli neat install sentinel'.");
  }
  json body = json::parse(result->body, nullptr, /*allow_exceptions=*/false);
  if (result->status != 200) {
    const bool has_error = body.is_object() && body.contains("error") && body["error"].is_string();
    throw std::runtime_error("Sentinel could not " + action + " (HTTP " +
                             std::to_string(result->status) +
                             (has_error ? "): " + body["error"].get<std::string>() : ")"));
  }
  if (!body.is_object()) {
    throw std::runtime_error("Sentinel sent invalid JSON when asked to " + action);
  }
  return body;
}

} // namespace

SentinelClient::SentinelClient(std::string socket_path, std::chrono::seconds timeout)
    : socket_path_(std::move(socket_path)), timeout_(timeout) {}

json SentinelClient::latest_sample() const {
  const std::string action = "read the latest sample";
  const json reply = expect_ok(get(socket_path_, timeout_, "/v1/samples/latest"), action);
  if (!reply.contains("sample") || !reply["sample"].is_object()) {
    throw std::runtime_error("Sentinel sent no sample when asked to " + action);
  }
  return reply["sample"];
}

json SentinelClient::cache() const {
  const std::string action = "read its sample cache";
  json reply = expect_ok(get(socket_path_, timeout_, "/v1/cache"), action);
  if (!reply.contains("samples") || !reply["samples"].is_array() || !reply.contains("metrics") ||
      !reply["metrics"].is_array()) {
    throw std::runtime_error("Sentinel sent no samples or metrics when asked to " + action);
  }
  return reply;
}

} // namespace simaai::neat::pipeline_internal

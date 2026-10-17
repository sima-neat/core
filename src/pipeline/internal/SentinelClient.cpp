#include "pipeline/internal/SentinelClient.h"

#include <httplib.h>
#include <sys/socket.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

using nlohmann::json;

constexpr int kTimeoutSeconds = 5;

/// Send one request; throw when Sentinel cannot be reached.
httplib::Result send(const std::string& socket_path, const std::string& method,
                     const std::string& path, const std::string& body = {}) {
  httplib::Client client(socket_path);
  client.set_address_family(AF_UNIX);
  client.set_connection_timeout(kTimeoutSeconds);
  client.set_read_timeout(kTimeoutSeconds);
  client.set_write_timeout(kTimeoutSeconds);
  httplib::Result result =
      method == "GET" ? client.Get(path) : client.Post(path, body, "application/json");
  if (!result) {
    throw std::runtime_error("Sentinel is not reachable at " + socket_path + ": " +
                             httplib::to_string(result.error()) +
                             "; install it with 'sima-cli neat install sentinel'");
  }
  return result;
}

/// Return the JSON body of a 200 reply; otherwise throw with Sentinel's error message.
json expect_ok(const httplib::Result& result, const std::string& action) {
  json body = json::parse(result->body, nullptr, /*allow_exceptions=*/false);
  if (result->status != 200) {
    const bool has_error = body.is_object() && body.contains("error") && body["error"].is_string();
    throw std::runtime_error("Sentinel could not " + action + " (HTTP " +
                             std::to_string(result->status) +
                             (has_error ? "): " + body["error"].get<std::string>() : ")"));
  }
  if (body.is_discarded()) {
    throw std::runtime_error("Sentinel sent invalid JSON when asked to " + action);
  }
  return body;
}

} // namespace

SentinelClient::SentinelClient(std::string socket_path) : socket_path_(std::move(socket_path)) {}

std::string SentinelClient::start_trace(const std::string& name, const std::string& note,
                                        const std::vector<std::string>& tags) const {
  const json request = {{"name", name}, {"note", note}, {"tags", tags}};
  const json reply =
      expect_ok(send(socket_path_, "POST", "/v1/traces", request.dump()), "start a trace");
  const json::json_pointer id("/trace/id");
  if (!reply.contains(id) || !reply[id].is_string()) {
    throw std::runtime_error("Sentinel started a trace without returning its ID");
  }
  return reply[id].get<std::string>();
}

void SentinelClient::stop_trace(const std::string& trace_id) const {
  const std::string action = "stop trace " + trace_id;
  const httplib::Result result = send(socket_path_, "POST", "/v1/traces/" + trace_id + "/stop");
  if (result->status == 404) {
    // Sentinel 0.1.0 has no stop-by-ID endpoint and stops the trace it is recording.
    expect_ok(send(socket_path_, "POST", "/v1/traces/stop"), action);
    return;
  }
  expect_ok(result, action);
}

json SentinelClient::read_run(const std::string& trace_id) const {
  return expect_ok(send(socket_path_, "GET", "/v1/runs/" + trace_id), "read trace " + trace_id);
}

json SentinelClient::latest_sample() const {
  const std::string action = "read the latest sample";
  const json reply = expect_ok(send(socket_path_, "GET", "/v1/samples/latest"), action);
  if (!reply.contains("sample") || !reply["sample"].is_object()) {
    throw std::runtime_error("Sentinel sent no sample when asked to " + action);
  }
  return reply["sample"];
}

} // namespace simaai::neat::pipeline_internal

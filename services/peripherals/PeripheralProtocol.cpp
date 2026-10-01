#include "PeripheralProtocol.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

nlohmann::json encode_issue(const CatalogIssue& issue) {
  return {
      {"provider", issue.provider},
      {"code", issue.code},
      {"reason", issue.reason},
      {"retained_last_good", issue.retained_last_good},
  };
}

nlohmann::json encode_base(const CatalogMetadata& metadata) {
  nlohmann::json result = {
      {"schema_version", kPeripheralSchemaVersion},
      {"instance_id", metadata.instance_id},
      {"state", metadata.degraded ? "degraded" : (metadata.initialized ? "ready" : "starting")},
      {"ready", metadata.initialized},
      {"stale", metadata.stale},
      {"revision", metadata.revision},
      {"sequence", metadata.sequence},
      {"scan_sequence", metadata.scan_sequence},
      {"last_success_at", metadata.last_success_at.empty()
                              ? nlohmann::json(nullptr)
                              : nlohmann::json(metadata.last_success_at)},
      {"last_attempt_at", metadata.last_attempt_at.empty()
                              ? nlohmann::json(nullptr)
                              : nlohmann::json(metadata.last_attempt_at)},
      {"error", metadata.degraded ? metadata.error : nlohmann::json(nullptr)},
  };
  result["issues"] = nlohmann::json::array();
  for (const auto& issue : metadata.issues)
    result["issues"].push_back(encode_issue(issue));
  return result;
}

nlohmann::json encode_event(const PeripheralEvent& event) {
  nlohmann::json result = {
      {"sequence", event.sequence},
      {"revision", event.revision},
      {"kind", event.kind},
  };
  if (!event.device_id.empty()) {
    result["device_id"] = event.device_id;
    result["device_type"] = event.device_type;
  }
  if (event.previous)
    result["previous"] = encode_peripheral_record(*event.previous);
  if (event.current)
    result["current"] = encode_peripheral_record(*event.current);
  if (!event.error.is_null())
    result["error"] = event.error;
  return result;
}

} // namespace

nlohmann::json encode_peripheral_record(const PeripheralRecord& record) {
  return {
      {"id", record.id},
      {"type", record.type},
      {"provider", record.provider},
      {record.type, record.details},
  };
}

nlohmann::json encode_health(const CatalogMetadata& metadata, std::size_t device_count) {
  nlohmann::json result = encode_base(metadata);
  result["api_version"] = kPeripheralApiVersion;
  result["device_count"] = device_count;
  return result;
}

nlohmann::json encode_catalog(const CatalogMetadata& metadata,
                              const std::vector<PeripheralRecord>& devices) {
  nlohmann::json result = encode_base(metadata);
  result["devices"] = nlohmann::json::array();
  for (const auto& device : devices)
    result["devices"].push_back(encode_peripheral_record(device));
  return result;
}

nlohmann::json encode_events(const CatalogMetadata& metadata, bool resync_required,
                             bool shutting_down, const std::vector<PeripheralEvent>& events) {
  nlohmann::json result = {
      {"schema_version", kPeripheralSchemaVersion},
      {"instance_id", metadata.instance_id},
      {"revision", metadata.revision},
      {"sequence", metadata.sequence},
      {"scan_sequence", metadata.scan_sequence},
      {"resync_required", resync_required},
      {"shutting_down", shutting_down},
      {"events", nlohmann::json::array()},
  };
  if (!resync_required) {
    for (const auto& event : events)
      result["events"].push_back(encode_event(event));
  }
  return result;
}

nlohmann::json encode_error(std::string code) {
  return {{"error", std::move(code)}};
}

nlohmann::json encode_error(std::string code, std::string reason) {
  return {{"error", std::move(code)}, {"reason", std::move(reason)}};
}

nlohmann::json encode_refresh_accepted(std::uint64_t target_scan_sequence) {
  return {{"accepted", true}, {"target_scan_sequence", target_scan_sequence}};
}

} // namespace simaai::neat::peripherals_internal

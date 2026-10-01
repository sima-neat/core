#pragma once

#include "peripherals/internal/ProtocolContract.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::peripherals_internal {

struct PeripheralRecord {
  std::string id;
  std::string type;
  std::string provider;
  nlohmann::json details;
};

struct CatalogIssue {
  std::string provider;
  std::string code;
  std::string reason;
  bool retained_last_good = false;

  bool operator==(const CatalogIssue&) const = default;
};

struct PeripheralEvent {
  std::uint64_t sequence = 0;
  std::uint64_t revision = 0;
  std::string kind;
  std::string device_id;
  std::string device_type;
  std::optional<PeripheralRecord> previous;
  std::optional<PeripheralRecord> current;
  nlohmann::json error;
};

struct CatalogMetadata {
  std::string instance_id;
  bool initialized = false;
  bool degraded = false;
  bool stale = false;
  std::uint64_t revision = 0;
  std::uint64_t sequence = 0;
  std::uint64_t scan_sequence = 0;
  std::string last_success_at;
  std::string last_attempt_at;
  nlohmann::json error;
  std::vector<CatalogIssue> issues;
};

nlohmann::json encode_peripheral_record(const PeripheralRecord& record);
nlohmann::json encode_health(const CatalogMetadata& metadata, std::size_t device_count);
nlohmann::json encode_catalog(const CatalogMetadata& metadata,
                              const std::vector<PeripheralRecord>& devices);
nlohmann::json encode_events(const CatalogMetadata& metadata, bool resync_required,
                             bool shutting_down, const std::vector<PeripheralEvent>& events);
nlohmann::json encode_error(std::string code);
nlohmann::json encode_error(std::string code, std::string reason);
nlohmann::json encode_refresh_accepted(std::uint64_t target_scan_sequence);

} // namespace simaai::neat::peripherals_internal

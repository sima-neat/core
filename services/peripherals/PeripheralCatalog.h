#pragma once

#include "PeripheralProtocol.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::peripherals_internal {

class PeripheralCatalog {
public:
  explicit PeripheralCatalog(std::string instance_id, std::size_t event_capacity = 256);

  PeripheralCatalog(const PeripheralCatalog&) = delete;
  PeripheralCatalog& operator=(const PeripheralCatalog&) = delete;

  void apply_success(std::vector<PeripheralRecord> devices, std::vector<CatalogIssue> issues = {});
  void apply_provider_failure(std::vector<CatalogIssue> issues);
  void apply_scan_error(std::string code, std::string reason);
  void apply_error(std::string code, std::string reason);
  std::uint64_t scan_sequence() const;

  nlohmann::json health_json() const;
  nlohmann::json catalog_json() const;
  nlohmann::json events_json(std::uint64_t after_sequence, std::chrono::milliseconds wait,
                             const std::optional<std::string>& client_instance_id);

  void shutdown();

private:
  void append_event_locked(PeripheralEvent event);
  void apply_error_locked(std::string code, std::string reason, bool scan_attempted);
  CatalogMetadata metadata_locked() const;

  const std::string instance_id_;
  const std::size_t event_capacity_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<PeripheralRecord> devices_;
  std::vector<CatalogIssue> issues_;
  std::deque<PeripheralEvent> events_;
  std::uint64_t revision_ = 0;
  std::uint64_t sequence_ = 0;
  std::uint64_t scan_sequence_ = 0;
  bool initialized_ = false;
  bool shutting_down_ = false;
  std::string last_success_at_;
  std::string last_attempt_at_;
  nlohmann::json error_;
};

} // namespace simaai::neat::peripherals_internal

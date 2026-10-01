#include "PeripheralCatalog.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

std::string utc_now() {
  const auto now = std::chrono::system_clock::now();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  std::ostringstream value;
  value << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
        << milliseconds.count() << 'Z';
  return value.str();
}

bool same_record(const PeripheralRecord& left, const PeripheralRecord& right) {
  return std::tie(left.id, left.type, left.provider, left.details) ==
         std::tie(right.id, right.type, right.provider, right.details);
}

void canonicalize(std::vector<PeripheralRecord>& devices) {
  std::sort(devices.begin(), devices.end(),
            [](const auto& left, const auto& right) { return left.id < right.id; });
  const auto duplicate =
      std::adjacent_find(devices.begin(), devices.end(),
                         [](const auto& left, const auto& right) { return left.id == right.id; });
  if (duplicate != devices.end())
    throw std::invalid_argument("duplicate peripheral identity: " + duplicate->id);
}

void canonicalize(std::vector<CatalogIssue>& issues) {
  std::sort(issues.begin(), issues.end(), [](const auto& left, const auto& right) {
    return std::tie(left.provider, left.code, left.reason, left.retained_last_good) <
           std::tie(right.provider, right.code, right.reason, right.retained_last_good);
  });
  const auto invalid = std::find_if(issues.begin(), issues.end(), [](const auto& issue) {
    return issue.provider.empty() || issue.code.empty() || issue.reason.empty();
  });
  if (invalid != issues.end())
    throw std::invalid_argument("peripheral provider issue fields must not be empty");
  const auto duplicate =
      std::adjacent_find(issues.begin(), issues.end(), [](const auto& left, const auto& right) {
        return left.provider == right.provider;
      });
  if (duplicate != issues.end())
    throw std::invalid_argument("duplicate peripheral provider issue: " + duplicate->provider);
}

nlohmann::json issue_error(const std::vector<CatalogIssue>& issues) {
  nlohmann::json encoded = nlohmann::json::array();
  for (const auto& issue : issues) {
    encoded.push_back({
        {"provider", issue.provider},
        {"code", issue.code},
        {"reason", issue.reason},
        {"retained_last_good", issue.retained_last_good},
    });
  }
  return {
      {"code", "peripherals.provider_degraded"},
      {"reason", "One or more peripheral providers could not be refreshed."},
      {"issues", std::move(encoded)},
  };
}

} // namespace

PeripheralCatalog::PeripheralCatalog(std::string instance_id, std::size_t event_capacity)
    : instance_id_(std::move(instance_id)), event_capacity_(event_capacity) {
  if (instance_id_.empty())
    throw std::invalid_argument("peripheral catalog instance ID must not be empty");
  if (event_capacity_ == 0)
    throw std::invalid_argument("peripheral catalog event capacity must be greater than zero");
}

void PeripheralCatalog::append_event_locked(PeripheralEvent event) {
  event.sequence = ++sequence_;
  if (events_.size() == event_capacity_)
    events_.pop_front();
  events_.push_back(std::move(event));
}

void PeripheralCatalog::apply_success(std::vector<PeripheralRecord> devices,
                                      std::vector<CatalogIssue> issues) {
  canonicalize(devices);
  canonicalize(issues);
  const std::string attempt_time = utc_now();
  std::lock_guard lock(mutex_);
  ++scan_sequence_;
  const bool recovered = (!error_.is_null() || !issues_.empty()) && issues.empty();
  const bool issues_changed = issues_ != issues;
  last_attempt_at_ = attempt_time;
  if (issues.empty())
    last_success_at_ = attempt_time;
  error_ = nullptr;

  auto append_status_events = [&] {
    if (recovered) {
      PeripheralEvent event;
      event.revision = revision_;
      event.kind = "recovered";
      append_event_locked(std::move(event));
    }
    if (issues_changed && !issues.empty()) {
      PeripheralEvent event;
      event.revision = revision_;
      event.kind = "error";
      event.error = issue_error(issues);
      append_event_locked(std::move(event));
    }
  };

  if (!initialized_) {
    devices_ = std::move(devices);
    initialized_ = true;
    revision_ = 1;
    append_status_events();
    issues_ = std::move(issues);
    changed_.notify_all();
    return;
  }

  if (devices_.size() == devices.size() &&
      std::equal(devices_.begin(), devices_.end(), devices.begin(), same_record)) {
    append_status_events();
    issues_ = std::move(issues);
    changed_.notify_all();
    return;
  }

  ++revision_;
  append_status_events();
  std::size_t previous_index = 0;
  std::size_t current_index = 0;
  while (previous_index < devices_.size() || current_index < devices.size()) {
    if (current_index == devices.size() ||
        (previous_index < devices_.size() &&
         devices_[previous_index].id < devices[current_index].id)) {
      PeripheralEvent event;
      event.revision = revision_;
      event.kind = "removed";
      event.device_id = devices_[previous_index].id;
      event.device_type = devices_[previous_index].type;
      event.previous = devices_[previous_index++];
      append_event_locked(std::move(event));
      continue;
    }
    if (previous_index == devices_.size() ||
        devices[current_index].id < devices_[previous_index].id) {
      PeripheralEvent event;
      event.revision = revision_;
      event.kind = "added";
      event.device_id = devices[current_index].id;
      event.device_type = devices[current_index].type;
      event.current = devices[current_index++];
      append_event_locked(std::move(event));
      continue;
    }
    if (!same_record(devices_[previous_index], devices[current_index])) {
      PeripheralEvent event;
      event.revision = revision_;
      event.kind = "changed";
      event.device_id = devices[current_index].id;
      event.device_type = devices[current_index].type;
      event.previous = devices_[previous_index];
      event.current = devices[current_index];
      append_event_locked(std::move(event));
    }
    ++previous_index;
    ++current_index;
  }
  devices_ = std::move(devices);
  issues_ = std::move(issues);
  changed_.notify_all();
}

void PeripheralCatalog::apply_error_locked(std::string code, std::string reason,
                                           bool scan_attempted) {
  if (scan_attempted)
    last_attempt_at_ = utc_now();
  error_ = {{"code", std::move(code)}, {"reason", std::move(reason)}};
  PeripheralEvent event;
  event.revision = revision_;
  event.kind = "error";
  event.error = error_;
  append_event_locked(std::move(event));
  changed_.notify_all();
}

void PeripheralCatalog::apply_provider_failure(std::vector<CatalogIssue> issues) {
  canonicalize(issues);
  if (issues.empty())
    throw std::invalid_argument("a provider failure scan must contain an issue");
  std::lock_guard lock(mutex_);
  ++scan_sequence_;
  last_attempt_at_ = utc_now();
  error_ = nullptr;
  if (issues_ != issues) {
    PeripheralEvent event;
    event.revision = revision_;
    event.kind = "error";
    event.error = issue_error(issues);
    append_event_locked(std::move(event));
  }
  issues_ = std::move(issues);
  changed_.notify_all();
}

void PeripheralCatalog::apply_scan_error(std::string code, std::string reason) {
  std::lock_guard lock(mutex_);
  ++scan_sequence_;
  apply_error_locked(std::move(code), std::move(reason), true);
}

void PeripheralCatalog::apply_error(std::string code, std::string reason) {
  std::lock_guard lock(mutex_);
  apply_error_locked(std::move(code), std::move(reason), false);
}

CatalogMetadata PeripheralCatalog::metadata_locked() const {
  const bool retained_provider_data = std::any_of(
      issues_.begin(), issues_.end(), [](const auto& issue) { return issue.retained_last_good; });
  return {
      .instance_id = instance_id_,
      .initialized = initialized_,
      .degraded = !error_.is_null() || !issues_.empty(),
      .stale = initialized_ && (!error_.is_null() || retained_provider_data),
      .revision = revision_,
      .sequence = sequence_,
      .scan_sequence = scan_sequence_,
      .last_success_at = last_success_at_,
      .last_attempt_at = last_attempt_at_,
      .error = error_,
      .issues = issues_,
  };
}

nlohmann::json PeripheralCatalog::health_json() const {
  std::lock_guard lock(mutex_);
  return encode_health(metadata_locked(), devices_.size());
}

nlohmann::json PeripheralCatalog::catalog_json() const {
  std::lock_guard lock(mutex_);
  return encode_catalog(metadata_locked(), devices_);
}

std::uint64_t PeripheralCatalog::scan_sequence() const {
  std::lock_guard lock(mutex_);
  return scan_sequence_;
}

nlohmann::json
PeripheralCatalog::events_json(std::uint64_t after_sequence, std::chrono::milliseconds wait,
                               const std::optional<std::string>& client_instance_id) {
  std::unique_lock lock(mutex_);
  auto requires_resync = [&] {
    if (client_instance_id && *client_instance_id != instance_id_)
      return true;
    if (after_sequence > sequence_)
      return true;
    return !events_.empty() && after_sequence < events_.front().sequence - 1;
  };
  if (!requires_resync() && after_sequence == sequence_ && wait.count() > 0) {
    changed_.wait_for(lock, wait, [&] {
      return shutting_down_ || sequence_ != after_sequence || requires_resync();
    });
  }

  const bool resync = requires_resync();
  std::vector<PeripheralEvent> result_events;
  if (!resync) {
    for (const auto& event : events_) {
      if (event.sequence > after_sequence)
        result_events.push_back(event);
    }
  }
  return encode_events(metadata_locked(), resync, shutting_down_, result_events);
}

void PeripheralCatalog::shutdown() {
  std::lock_guard lock(mutex_);
  shutting_down_ = true;
  changed_.notify_all();
}

} // namespace simaai::neat::peripherals_internal

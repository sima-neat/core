#include "PeripheralCatalogManager.h"

#include "pipeline/NeatError.h"

#include <libudev.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace simaai::neat::peripherals_internal {
namespace {

struct FileDescriptor {
  int value = -1;

  FileDescriptor() = default;
  explicit FileDescriptor(int fd) : value(fd) {}
  ~FileDescriptor() {
    if (value >= 0)
      ::close(value);
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
};

struct UdevDeleter {
  void operator()(udev* context) const {
    if (context)
      udev_unref(context);
  }
};

struct UdevMonitorDeleter {
  void operator()(udev_monitor* monitor) const {
    if (monitor)
      udev_monitor_unref(monitor);
  }
};

void notify_eventfd(int fd) {
  const std::uint64_t value = 1;
  while (::write(fd, &value, sizeof(value)) < 0) {
    if (errno == EINTR)
      continue;
    if (errno == EAGAIN)
      return;
    throw std::runtime_error("failed to signal peripheral catalog worker: " +
                             std::string(std::strerror(errno)));
  }
}

void drain_eventfd(int fd) {
  while (true) {
    std::uint64_t value = 0;
    const ssize_t result = ::read(fd, &value, sizeof(value));
    if (result == sizeof(value))
      continue;
    if (result < 0 && errno == EINTR)
      continue;
    if (result < 0 && errno == EAGAIN)
      return;
    throw std::runtime_error("failed to drain peripheral catalog signal: " +
                             std::string(std::strerror(result < 0 ? errno : EIO)));
  }
}

class InvalidProviderResult : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

bool valid_type_token(const std::string& type) {
  if (type.empty() || type.size() > 64 || type == "id" || type == "type" || type == "provider" ||
      type.front() < 'a' || type.front() > 'z') {
    return false;
  }
  return std::all_of(type.begin() + 1, type.end(), [](const unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
  });
}

CatalogIssue provider_issue(const PeripheralProvider& provider, std::string code,
                            std::string reason, bool retained_last_good) {
  if (code.empty())
    code = "peripherals.discovery_failed";
  if (reason.empty())
    reason = "The provider failed without reporting a reason.";
  return {provider.name, std::move(code), std::move(reason), retained_last_good};
}

void validate_provider_result(const PeripheralProvider& provider,
                              const std::vector<PeripheralRecord>& records) {
  std::set<std::string> record_ids;
  for (const auto& record : records) {
    if (record.id.empty() || !valid_type_token(record.type) || record.provider.empty() ||
        !record.details.is_object()) {
      throw InvalidProviderResult(
          "provider returned a record without a non-empty id, valid type token, provider, and "
          "details object");
    }
    if (record.provider != provider.name) {
      throw InvalidProviderResult("provider returned a record owned by " + record.provider);
    }
    if (!record_ids.insert(record.id).second) {
      throw InvalidProviderResult("provider returned duplicate peripheral identity: " + record.id);
    }
  }
}

} // namespace

struct PeripheralCatalogManager::Impl {
  struct ProviderState {
    PeripheralProvider provider;
    std::vector<PeripheralRecord> last_good;
    bool has_last_good = false;
  };

  Impl(PeripheralCatalog& catalog_value, std::vector<PeripheralProvider> providers_value,
       std::uint32_t debounce_value)
      : catalog(catalog_value), debounce(debounce_value),
        refresh_fd(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)),
        stop_fd(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (providers_value.empty())
      throw std::invalid_argument("at least one peripheral provider must be set");
    std::set<std::string> provider_names;
    std::set<std::string> subsystems;
    providers.reserve(providers_value.size());
    for (auto& provider : providers_value) {
      if (provider.name.empty() || !provider.discover)
        throw std::invalid_argument("peripheral provider name and callback must be set");
      if (!provider_names.insert(provider.name).second)
        throw std::invalid_argument("duplicate peripheral provider: " + provider.name);
      for (const auto& subsystem : provider.udev_subsystems) {
        if (subsystem.empty())
          throw std::invalid_argument("peripheral provider udev subsystem must not be empty");
        subsystems.insert(subsystem);
      }
      providers.push_back({.provider = std::move(provider)});
    }
    if (subsystems.empty())
      throw std::invalid_argument("at least one peripheral udev subsystem must be set");
    if (refresh_fd.value < 0 || stop_fd.value < 0)
      throw std::runtime_error("failed to create peripheral catalog eventfd: " +
                               std::string(std::strerror(errno)));

    context.reset(udev_new());
    if (!context)
      throw std::runtime_error("failed to create udev context for peripheral monitoring");
    monitor.reset(udev_monitor_new_from_netlink(context.get(), "udev"));
    if (!monitor)
      throw std::runtime_error("failed to create udev peripheral monitor");
    for (const auto& subsystem : subsystems) {
      if (udev_monitor_filter_add_match_subsystem_devtype(monitor.get(), subsystem.c_str(),
                                                          nullptr) < 0)
        throw std::runtime_error("failed to add udev peripheral filter for " + subsystem);
    }
    if (udev_monitor_filter_update(monitor.get()) < 0 ||
        udev_monitor_enable_receiving(monitor.get()) < 0) {
      throw std::runtime_error("failed to enable udev peripheral monitoring");
    }
  }

  void scan() {
    try {
      {
        std::lock_guard lock(schedule_mutex);
        scan_in_progress = true;
        active_scan_sequence = catalog.scan_sequence() + 1;
      }
      struct ScanResult {
        ProviderState* state;
        std::vector<PeripheralRecord> discovered;
        bool accepted = false;
        std::optional<CatalogIssue> issue;
      };

      std::map<std::string, std::string> established_owners;
      for (const auto& state : providers) {
        if (!state.has_last_good)
          continue;
        for (const auto& record : state.last_good)
          established_owners.emplace(record.id, state.provider.name);
      }

      std::vector<ScanResult> results;
      results.reserve(providers.size());
      for (auto& state : providers) {
        ScanResult result{.state = &state};
        try {
          result.discovered = state.provider.discover();
          validate_provider_result(state.provider, result.discovered);
          result.accepted = true;
        } catch (const NeatError& error) {
          result.issue = provider_issue(state.provider, error.report().error_code, error.what(),
                                        state.has_last_good);
        } catch (const InvalidProviderResult& error) {
          result.issue = provider_issue(state.provider, "peripherals.invalid_provider_result",
                                        error.what(), state.has_last_good);
        } catch (const std::exception& error) {
          result.issue = provider_issue(state.provider, "peripherals.discovery_failed",
                                        error.what(), state.has_last_good);
        } catch (...) {
          result.issue = provider_issue(
              state.provider, "peripherals.discovery_failed",
              "The provider failed with an unknown non-standard exception.", state.has_last_good);
        }
        results.push_back(std::move(result));
      }

      const auto effective_records =
          [](const ScanResult& result) -> const std::vector<PeripheralRecord>& {
        return result.accepted ? result.discovered : result.state->last_good;
      };

      bool ownership_changed = false;
      do {
        ownership_changed = false;
        std::map<std::string, std::vector<std::size_t>> claims;
        for (std::size_t index = 0; index < results.size(); ++index) {
          for (const auto& record : effective_records(results[index]))
            claims[record.id].push_back(index);
        }
        for (const auto& [id, claimants] : claims) {
          if (claimants.size() < 2)
            continue;

          std::optional<std::size_t> established_owner;
          const auto prior = established_owners.find(id);
          if (prior != established_owners.end()) {
            for (const auto index : claimants) {
              if (results[index].state->provider.name == prior->second) {
                established_owner = index;
                break;
              }
            }
          }

          for (const auto index : claimants) {
            auto& result = results[index];
            if (!result.accepted || (established_owner && *established_owner == index))
              continue;
            result.accepted = false;
            result.issue = provider_issue(
                result.state->provider, "peripherals.invalid_provider_result",
                established_owner
                    ? "peripheral identity " + id + " remains owned by provider " +
                          results[*established_owner].state->provider.name
                    : "peripheral identity " + id +
                          " was returned by multiple providers without an established owner",
                result.state->has_last_good);
            ownership_changed = true;
          }
        }
      } while (ownership_changed);

      std::vector<PeripheralRecord> devices;
      std::vector<CatalogIssue> issues;
      bool has_provider_result = false;
      for (auto& result : results) {
        if (result.accepted) {
          result.state->last_good = std::move(result.discovered);
          result.state->has_last_good = true;
        }
        if (result.issue)
          issues.push_back(std::move(*result.issue));
        has_provider_result = has_provider_result || result.state->has_last_good;
        devices.insert(devices.end(), result.state->last_good.begin(),
                       result.state->last_good.end());
      }
      if (has_provider_result)
        catalog.apply_success(std::move(devices), std::move(issues));
      else
        catalog.apply_provider_failure(std::move(issues));
    } catch (...) {
      std::lock_guard lock(schedule_mutex);
      scan_in_progress = false;
      throw;
    }
    std::lock_guard lock(schedule_mutex);
    scan_in_progress = false;
  }

  void consume_udev_events() {
    while (udev_device* device = udev_monitor_receive_device(monitor.get()))
      udev_device_unref(device);
  }

  void run() {
    using Clock = std::chrono::steady_clock;
    Clock::time_point refresh_at{};
    bool refresh_pending = false;
    const int monitor_fd = udev_monitor_get_fd(monitor.get());
    if (monitor_fd < 0) {
      const std::string reason = "udev monitor has no pollable file descriptor";
      catalog.apply_error("peripherals.monitor_failed", reason);
      throw std::runtime_error(reason);
    }
    while (!stopping.load()) {
      int timeout_ms = -1;
      if (refresh_pending) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(refresh_at - Clock::now());
        timeout_ms = static_cast<int>(
            std::clamp<std::int64_t>(remaining.count(), 0, std::numeric_limits<int>::max()));
      }
      std::array<pollfd, 3> descriptors = {
          pollfd{monitor_fd, POLLIN, 0},
          pollfd{refresh_fd.value, POLLIN, 0},
          pollfd{stop_fd.value, POLLIN, 0},
      };
      const int result = ::poll(descriptors.data(), descriptors.size(), timeout_ms);
      if (result < 0) {
        if (errno == EINTR)
          continue;
        const std::string reason = "udev poll failed: " + std::string(std::strerror(errno));
        catalog.apply_error("peripherals.monitor_failed", reason);
        throw std::runtime_error(reason);
      }
      if (descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
        const std::string reason = "udev monitor became unavailable";
        catalog.apply_error("peripherals.monitor_failed", reason);
        throw std::runtime_error(reason);
      }
      if (descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL)) {
        const std::string reason = "peripheral refresh signal became unavailable";
        catalog.apply_error("peripherals.monitor_failed", reason);
        throw std::runtime_error(reason);
      }
      if (descriptors[2].revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("peripheral stop signal became unavailable");
      if (descriptors[2].revents & POLLIN) {
        drain_eventfd(stop_fd.value);
        return;
      }
      if (descriptors[0].revents & POLLIN) {
        consume_udev_events();
        refresh_at = Clock::now() + debounce;
        refresh_pending = true;
      }
      if (descriptors[1].revents & POLLIN) {
        drain_eventfd(refresh_fd.value);
        refresh_at = Clock::now();
        refresh_pending = true;
      }
      if (refresh_pending && Clock::now() >= refresh_at) {
        refresh_pending = false;
        scan();
      }
    }
  }

  PeripheralCatalog& catalog;
  std::vector<ProviderState> providers;
  std::chrono::milliseconds debounce;
  FileDescriptor refresh_fd;
  FileDescriptor stop_fd;
  std::unique_ptr<udev, UdevDeleter> context;
  std::unique_ptr<udev_monitor, UdevMonitorDeleter> monitor;
  std::thread worker;
  std::atomic<bool> stopping{false};
  mutable std::mutex schedule_mutex;
  bool scan_in_progress = false;
  std::uint64_t active_scan_sequence = 0;
  mutable std::mutex failure_mutex;
  std::exception_ptr worker_failure;

  void clear_failure() {
    std::lock_guard lock(failure_mutex);
    worker_failure = nullptr;
  }

  void record_failure(std::exception_ptr failure) noexcept {
    std::lock_guard lock(failure_mutex);
    worker_failure = std::move(failure);
  }

  void throw_if_failed() const {
    std::exception_ptr failure;
    {
      std::lock_guard lock(failure_mutex);
      failure = worker_failure;
    }
    if (failure)
      std::rethrow_exception(failure);
  }
};

PeripheralCatalogManager::PeripheralCatalogManager(PeripheralCatalog& catalog,
                                                   std::vector<PeripheralProvider> providers,
                                                   std::uint32_t debounce_ms)
    : impl_(std::make_unique<Impl>(catalog, std::move(providers), debounce_ms)) {}

PeripheralCatalogManager::~PeripheralCatalogManager() {
  try {
    stop();
  } catch (...) {
  }
}

void PeripheralCatalogManager::initial_scan() {
  impl_->scan();
}

void PeripheralCatalogManager::start() {
  if (impl_->worker.joinable())
    throw std::logic_error("peripheral catalog manager is already running");
  impl_->clear_failure();
  impl_->stopping = false;
  impl_->worker = std::thread([this] {
    try {
      impl_->run();
    } catch (...) {
      impl_->record_failure(std::current_exception());
    }
  });
}

std::uint64_t PeripheralCatalogManager::request_refresh() {
  std::lock_guard lock(impl_->schedule_mutex);
  if (impl_->stopping.load())
    throw std::runtime_error("peripheral catalog manager is stopping");
  const std::uint64_t target_scan_sequence = impl_->scan_in_progress
                                                 ? impl_->active_scan_sequence + 1
                                                 : impl_->catalog.scan_sequence() + 1;
  notify_eventfd(impl_->refresh_fd.value);
  return target_scan_sequence;
}

void PeripheralCatalogManager::request_stop() {
  if (!impl_ || !impl_->worker.joinable())
    return;
  std::lock_guard lock(impl_->schedule_mutex);
  impl_->stopping = true;
  notify_eventfd(impl_->stop_fd.value);
}

void PeripheralCatalogManager::join() {
  if (!impl_ || !impl_->worker.joinable())
    return;
  impl_->worker.join();
}

void PeripheralCatalogManager::stop() {
  request_stop();
  join();
}

void PeripheralCatalogManager::throw_if_failed() const {
  impl_->throw_if_failed();
}

} // namespace simaai::neat::peripherals_internal

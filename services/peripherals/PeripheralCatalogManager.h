#pragma once

#include "PeripheralCatalog.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::peripherals_internal {

using PeripheralDiscovery = std::function<std::vector<PeripheralRecord>()>;

struct PeripheralProvider {
  std::string name;
  std::vector<std::string> udev_subsystems;
  PeripheralDiscovery discover;
};

class PeripheralCatalogManager {
public:
  PeripheralCatalogManager(PeripheralCatalog& catalog, std::vector<PeripheralProvider> providers,
                           std::uint32_t debounce_ms = 250);
  ~PeripheralCatalogManager();

  PeripheralCatalogManager(const PeripheralCatalogManager&) = delete;
  PeripheralCatalogManager& operator=(const PeripheralCatalogManager&) = delete;

  void initial_scan();
  void start();
  std::uint64_t request_refresh();
  void request_stop();
  void join();
  void stop();
  int failure_fd() const noexcept;
  void throw_if_failed() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::peripherals_internal

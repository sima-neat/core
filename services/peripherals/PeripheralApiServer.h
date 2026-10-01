#pragma once

#include "PeripheralCatalog.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace simaai::neat::peripherals_internal {

class PeripheralApiServer {
public:
  using RefreshRequest = std::function<std::uint64_t()>;

  PeripheralApiServer(std::string socket_path, PeripheralCatalog& catalog,
                      RefreshRequest refresh_request, std::size_t max_clients = 64);
  ~PeripheralApiServer();

  PeripheralApiServer(const PeripheralApiServer&) = delete;
  PeripheralApiServer& operator=(const PeripheralApiServer&) = delete;

  void start();
  void stop();
  int failure_fd() const noexcept;
  void throw_if_failed() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::peripherals_internal

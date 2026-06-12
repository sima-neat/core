#pragma once

#include "pipeline/PowerTelemetry.h"

#include <cstdint>
#include <memory>
#include <string>

namespace simaai::neat::pipeline_internal {

class PowerI2cBackend {
public:
  virtual ~PowerI2cBackend() = default;
  virtual bool set_page(int bus, std::uint8_t addr, std::uint8_t page, std::string& error) = 0;
  virtual bool read_register(int bus, std::uint8_t addr, std::uint8_t reg, std::uint8_t& value,
                             std::string& error) = 0;
};

std::shared_ptr<PowerI2cBackend> make_native_power_i2c_backend();

PowerSnapshot read_power_snapshot_with_backend(const PowerMonitorOptions& options,
                                               const std::shared_ptr<PowerI2cBackend>& backend);

} // namespace simaai::neat::pipeline_internal

#pragma once

#include "PeripheralProtocol.h"

#include <filesystem>
#include <vector>

namespace simaai::neat::peripherals_internal {

struct AlsaDiscoveryRoots {
  std::filesystem::path proc_asound = "/proc/asound";
  std::filesystem::path sys = "/sys";
  std::filesystem::path dev = "/dev";
};

std::vector<PeripheralRecord> discover_alsa_capture_peripherals(const AlsaDiscoveryRoots& roots);
std::vector<PeripheralRecord> discover_alsa_capture_peripherals();

} // namespace simaai::neat::peripherals_internal

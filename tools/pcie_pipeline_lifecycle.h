#pragma once

#include <string_view>

namespace simaai::neat::pcie_builder {

enum class CleanStopReason {
  Signal,
  HostEndOfStream,
};

struct CleanStopStatus {
  std::string_view stopping_message;
  std::string_view final_state;
  std::string_view final_message;
};

constexpr CleanStopStatus clean_stop_status(const CleanStopReason reason) noexcept {
  if (reason == CleanStopReason::HostEndOfStream) {
    return {
        .stopping_message = "host requested end of stream",
        .final_state = "exited",
        .final_message = "pipeline stopped after host EOS",
    };
  }
  return {
      .stopping_message = "signal received",
      .final_state = "exited",
      .final_message = "pipeline stopped",
  };
}

} // namespace simaai::neat::pcie_builder

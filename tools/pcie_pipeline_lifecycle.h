#pragma once

#include <string_view>

namespace simaai::neat::pcie_builder {

enum class CleanStopReason {
  Signal,
  EndOfStream,
};

struct CleanStopStatus {
  std::string_view stopping_message;
  std::string_view final_state;
  std::string_view final_message;
};

constexpr CleanStopStatus clean_stop_status(const CleanStopReason reason) noexcept {
  if (reason == CleanStopReason::EndOfStream) {
    return {
        .stopping_message = "pipeline reached end of stream",
        .final_state = "exited",
        .final_message = "pipeline stopped after end of stream",
    };
  }
  return {
      .stopping_message = "signal received",
      .final_state = "exited",
      .final_message = "pipeline stopped",
  };
}

} // namespace simaai::neat::pcie_builder

#include "tools/pcie_pipeline_lifecycle.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using simaai::neat::pcie_builder::clean_stop_status;
using simaai::neat::pcie_builder::CleanStopReason;

void require(const bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    const auto eos = clean_stop_status(CleanStopReason::EndOfStream);
    require(eos.final_state == std::string_view("exited"),
            "end of stream must be recorded as a clean exit");
    require(eos.stopping_message == std::string_view("pipeline reached end of stream"),
            "end of stream must remain distinguishable from a signal");

    const auto signal = clean_stop_status(CleanStopReason::Signal);
    require(signal.final_state == std::string_view("exited"),
            "signal shutdown must remain a clean exit");
    require(signal.stopping_message == std::string_view("signal received"),
            "signal shutdown must retain its diagnostic message");

    std::cout << "[PASS] PCIe pipeline lifecycle status\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}

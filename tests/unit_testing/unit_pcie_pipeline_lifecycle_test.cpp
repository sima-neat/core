#include "tools/pcie_pipeline_lifecycle.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using simaai::neat::pcie_builder::CleanStopReason;
using simaai::neat::pcie_builder::clean_stop_status;

void require(const bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    const auto host_eos = clean_stop_status(CleanStopReason::HostEndOfStream);
    require(host_eos.final_state == std::string_view("exited"),
            "host-requested EOS must be recorded as a clean exit");
    require(host_eos.stopping_message == std::string_view("host requested end of stream"),
            "host-requested EOS must remain distinguishable from a signal");

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

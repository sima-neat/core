#include "Service.h"

#include <stdexcept>

namespace simaai::neat::pcie::genai::wire {
namespace {
[[noreturn]] void unavailable() {
  throw std::runtime_error(
      "GenAI over PCIe requires the Windows platform service API; the installed "
      "Windows PCIe provider exposes vision queues only");
}
} // namespace

struct Service::Impl {};

Service::Service(int, bool) : impl_(std::make_unique<Impl>()) {
  unavailable();
}

Service::~Service() = default;

void Service::interrupt() noexcept {}

void Service::subscribe(const std::string&) {
  unavailable();
}

void Service::send(const std::string&, const std::string&) {
  unavailable();
}

std::optional<std::string> Service::receive(int) {
  unavailable();
}

void Service::fetch(const std::string&, const std::string&, const std::string&) {
  unavailable();
}

void Service::put(const std::string&, const std::string&) {
  unavailable();
}
} // namespace simaai::neat::pcie::genai::wire

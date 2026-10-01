#include "PeripheralApiServer.h"
#include "PeripheralCatalog.h"
#include "PeripheralCatalogManager.h"
#include "peripherals/internal/ProtocolContract.h"
#include "providers/CameraProvider.h"

#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace peripherals = simaai::neat::peripherals_internal;

namespace {

struct Options {
  std::string socket_path = peripherals::kPeripheralSocketPath;
  std::uint32_t debounce_ms = 250;
  std::uint32_t event_capacity = 256;
};

std::optional<std::uint32_t> parse_u32(std::string_view value) {
  std::uint32_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
    return std::nullopt;
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    auto next_value = [&]() -> std::string_view {
      if (++index >= argc)
        throw std::invalid_argument("missing value after " + std::string(argument));
      return argv[index];
    };
    if (argument == "--socket") {
      options.socket_path = next_value();
    } else if (argument == "--debounce-ms") {
      const auto value = parse_u32(next_value());
      if (!value)
        throw std::invalid_argument("invalid --debounce-ms value");
      options.debounce_ms = *value;
    } else if (argument == "--event-capacity") {
      const auto value = parse_u32(next_value());
      if (!value || *value == 0)
        throw std::invalid_argument("invalid --event-capacity value");
      options.event_capacity = *value;
    } else if (argument == "--help") {
      std::cout << "Usage: simaai-peripherals [--socket PATH] [--debounce-ms N] "
                   "[--event-capacity N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

std::string new_instance_id() {
  std::ifstream input("/proc/sys/kernel/random/uuid");
  std::string value;
  std::getline(input, value);
  if (value.empty())
    throw std::runtime_error("failed to generate daemon instance ID");
  return value;
}

int create_signal_fd() {
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  sigaddset(&signals, SIGHUP);
  if (pthread_sigmask(SIG_BLOCK, &signals, nullptr) != 0)
    throw std::runtime_error("failed to block daemon signals");
  const int fd = signalfd(-1, &signals, SFD_CLOEXEC);
  if (fd < 0)
    throw std::runtime_error("failed to create daemon signal fd: " +
                             std::string(std::strerror(errno)));
  return fd;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int signal_fd = create_signal_fd();

    peripherals::PeripheralCatalog catalog(new_instance_id(), options.event_capacity);
    peripherals::PeripheralCatalogManager manager(catalog,
                                                  {{"daemon.camera.libcamera",
                                                    {"media", "video4linux"},
                                                    peripherals::discover_camera_peripherals}},
                                                  options.debounce_ms);
    manager.initial_scan();
    manager.start();

    peripherals::PeripheralApiServer api(options.socket_path, catalog,
                                         [&manager] { return manager.request_refresh(); });
    api.start();
    std::cout << "simaai-peripherals listening on " << options.socket_path << std::endl;

    bool running = true;
    while (running) {
      std::array<pollfd, 3> descriptors = {
          pollfd{signal_fd, POLLIN, 0},
          pollfd{manager.failure_fd(), POLLIN, 0},
          pollfd{api.failure_fd(), POLLIN, 0},
      };
      const int result = ::poll(descriptors.data(), descriptors.size(), -1);
      if (result < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("daemon signal poll failed: " + std::string(std::strerror(errno)));
      }
      if (descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("peripheral catalog failure signal became unavailable");
      if (descriptors[2].revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("peripheral API failure signal became unavailable");
      if (descriptors[1].revents & POLLIN)
        manager.throw_if_failed();
      if (descriptors[2].revents & POLLIN)
        api.throw_if_failed();
      if (descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("daemon signal fd became unavailable");
      if (descriptors[0].revents & POLLIN) {
        signalfd_siginfo signal{};
        if (::read(signal_fd, &signal, sizeof(signal)) != sizeof(signal))
          continue;
        if (signal.ssi_signo == SIGHUP)
          manager.request_refresh();
        else if (signal.ssi_signo == SIGINT || signal.ssi_signo == SIGTERM)
          running = false;
      }
    }

    manager.request_stop();
    catalog.shutdown();
    api.stop();
    manager.join();
    ::close(signal_fd);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "simaai-peripherals: " << error.what() << std::endl;
    return 1;
  }
}

#include "genai/GenAIServer.h"
#include "test_main.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using simaai::neat::genai::GenAIServer;
using simaai::neat::genai::GenAIServerOptions;

// Ends the process with a clear message if the test does not finish in time, so that a server
// that cannot be stopped fails the test instead of hanging it.
void start_watchdog(std::chrono::seconds deadline) {
  std::thread([deadline] {
    std::this_thread::sleep_for(deadline);
    std::cerr << "[FAIL] GenAIServer start()/stop() did not return within " << deadline.count()
              << " s\n";
    std::_Exit(1);
  }).detach();
}

// A plain listening socket on an ephemeral loopback port. It deliberately does not set
// SO_REUSEPORT, so a server asked to listen on the same port cannot share it.
class PortHolder {
public:
  PortHolder() {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "socket() failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "inet_pton failed");
    require(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "bind() failed");
    require(::listen(fd_, 1) == 0, "listen() failed");
    socklen_t len = sizeof(addr);
    require(::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len) == 0,
            "getsockname() failed");
    port_ = ntohs(addr.sin_port);
  }

  PortHolder(const PortHolder&) = delete;
  PortHolder& operator=(const PortHolder&) = delete;

  ~PortHolder() {
    release();
  }

  uint16_t port() const {
    return port_;
  }

  void release() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  int fd_ = -1;
  uint16_t port_ = 0;
};

// True when a TCP connection to the loopback port is accepted right now.
bool can_connect(uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "socket() failed");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "inet_pton failed");
  const bool ok = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  ::close(fd);
  return ok;
}

// A free loopback port: bind an ephemeral one and release it again.
uint16_t pick_free_port() {
  PortHolder holder;
  const uint16_t port = holder.port();
  holder.release();
  return port;
}

// A bind failure must reach the caller of start() as an exception, the process must survive, and
// the same object must be startable once the port is free.
void start_reports_port_in_use_and_can_be_retried() {
  PortHolder holder;

  GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = holder.port();
  GenAIServer server(options);

  const std::string expected = "failed to listen on 127.0.0.1:" + std::to_string(holder.port());
  bool threw = false;
  try {
    server.start();
  } catch (const std::runtime_error& e) {
    threw = true;
    require_contains(e.what(), expected, "unexpected start() error");
  }
  require(threw, "start() on a busy port must throw std::runtime_error");

  holder.release();
  server.start();
  server.stop();
}

// A port that another server already listens on must be reported as in use, not shared with it.
void start_on_port_of_another_server_throws() {
  GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = pick_free_port();
  GenAIServer first(options);
  first.start();

  GenAIServer second(options);
  const std::string expected = "failed to listen on 127.0.0.1:" + std::to_string(options.port);
  bool threw = false;
  try {
    second.start();
  } catch (const std::runtime_error& e) {
    threw = true;
    require_contains(e.what(), expected, "unexpected start() error");
  }
  first.stop();
  require(threw, "start() on a port another server listens on must throw std::runtime_error");
}

// stop() right after start() must return instead of waiting on a server that never closes.
void stop_right_after_start_returns() {
  GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = 0;

  for (int i = 0; i < 20; ++i) {
    GenAIServer server(options);
    server.start();
    server.stop();
  }
}

// Once start() returns the server must already accept connections, and serve() must refuse to
// run a second listener on the same object.
void started_server_accepts_connections_and_rejects_serve() {
  GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = pick_free_port();
  GenAIServer server(options);
  server.start();

  require(can_connect(options.port), "start() returned before the server accepted connections");

  bool threw = false;
  try {
    server.serve();
  } catch (const std::runtime_error& e) {
    threw = true;
    require_contains(e.what(), "already running", "unexpected serve() error");
  }
  require(threw, "serve() on a started server must throw std::runtime_error");
  server.stop();
}

// start() while serve() blocks on another thread must throw on the caller's thread.
void start_while_serve_is_running_throws() {
  GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = pick_free_port();
  GenAIServer server(options);

  std::thread serving([&server] { server.serve(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!can_connect(options.port) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const bool listening = can_connect(options.port);

  bool threw = false;
  if (listening) {
    try {
      server.start();
    } catch (const std::runtime_error& e) {
      threw = true;
      require_contains(e.what(), "already running", "unexpected start() error");
    }
  }
  server.stop();
  serving.join();
  require(listening, "serve() did not start listening");
  require(threw, "start() while serve() is running must throw std::runtime_error");
}

} // namespace

RUN_TEST("unit_genai_server_port_in_use_regression_test", ([] {
           start_watchdog(std::chrono::seconds(30));
           start_reports_port_in_use_and_can_be_retried();
           start_on_port_of_another_server_throws();
           stop_right_after_start_returns();
           started_server_accepts_connections_and_rejects_serve();
           start_while_serve_is_running_throws();
         }));

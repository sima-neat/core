#include "genai/GenAIServer.h"
#include "test_main.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using simaai::neat::genai::GenAIServer;
using simaai::neat::genai::GenAIServerOptions;

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
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "socket() failed");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "inet_pton failed");
  socklen_t len = sizeof(addr);
  const bool ok = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
                  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0;
  ::close(fd);
  require(ok, "could not reserve a free loopback port");
  return ntohs(addr.sin_port);
}

// A port that another server listens on must be reported to the caller of start() instead of
// being shared, and the same object must start once that port is free.
void start_reports_busy_port_and_can_be_retried() {
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

  second.start();
  second.stop();
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
           start_reports_busy_port_and_can_be_retried();
           stop_right_after_start_returns();
           started_server_accepts_connections_and_rejects_serve();
           start_while_serve_is_running_throws();
         }));

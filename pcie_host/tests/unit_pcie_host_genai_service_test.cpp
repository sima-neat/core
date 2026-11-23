#include "Protocol.h"
#include "Service.h"

#include <simaai_svc.h>

#include <cerrno>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace wire = simaai::neat::pcie::genai::wire;

#ifdef _WIN32
void test_genai_open_error(int);
#endif

namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

template <class Fn> void rejects(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected rejection");
}
} // namespace

int main() {
  try {
    const std::string session(24, 'a');
    wire::Service host(0);
#ifdef _WIN32
    wire::Service worker(0, true);
#else
    wire::Service worker(0);
#endif
    host.subscribe(wire::tag(session, true));
    worker.subscribe(wire::tag(session, false));

    require(!host.receive(1), "Empty service receive must time out");
    std::exception_ptr remote_error;
    std::jthread remote([&] {
      try {
        auto receive = [&] {
          const auto payload = worker.receive(2000);
          require(payload.has_value(), "Worker request timed out");
          return wire::parse(*payload, session);
        };
        auto hello = receive();
        require(hello.at("kind") == "hello", "Worker received hello");
        hello["capabilities"] = {{"text", true}, {"image", true}, {"audio", false}};
        worker.send(wire::tag(session, true), hello.dump());

        const auto generate = receive();
        require(generate.at("kind") == "generate" && generate.at("body").at("prompt") == "hello",
                "Worker received generation request");
        auto sample = wire::envelope(session, 1, "sample");
        sample["sequence"] = 1;
        sample["body"] = {{"text", "answer"}, {"reasoning", ""},
                          {"final", true},    {"finish_reason", "stop"},
                          {"language", ""},   {"tool_calls", wire::Json::array()},
                          {"tokens", 1},      {"ttft", 0.0},
                          {"tps", 0.0}};
        worker.send(wire::tag(session, true), sample.dump());

        const auto cancel = receive();
        require(cancel.at("kind") == "cancel", "Worker received cancellation");
        auto cancelled = sample;
        cancelled["request"] = 2;
        cancelled["body"]["text"] = "";
        cancelled["body"]["finish_reason"] = "interrupted";
        worker.send(wire::tag(session, true), cancelled.dump());
      } catch (...) {
        remote_error = std::current_exception();
      }
    });

    host.send(wire::tag(session, false), wire::envelope(session, 0, "hello").dump());
    auto reply = host.receive(2000);
    require(reply.has_value() && wire::parse(*reply, session).at("capabilities").at("text"),
            "service transports readiness values");

    auto request = wire::envelope(session, 1, "generate");
    request["body"] = {{"prompt", "hello"}};
    host.send(wire::tag(session, false), request.dump());
    reply = host.receive(2000);
    require(reply.has_value(), "service transports generation result");
    const auto event = wire::parse(*reply, session);
    const auto sample = wire::decode(event.at("body"));
    require(sample.text == "answer" && sample.is_final && sample.metrics.generated_tokens == 1,
            "service preserves GenAI value fields");

    host.send(wire::tag(session, false), wire::envelope(session, 2, "cancel").dump());
    reply = host.receive(2000);
    require(reply.has_value() &&
                wire::decode(wire::parse(*reply, session).at("body")).finish_reason ==
                    "interrupted",
            "service transports terminal cancellation");
    remote.join();
    if (remote_error)
      std::rethrow_exception(remote_error);
    rejects([&] { wire::Service invalid(-1); });
#ifdef _WIN32
    // A stopped SiMaMlaDaemon service fails the open with a bare code: name the fix.
    std::string stopped;
    test_genai_open_error(-ENOENT);
    try {
      wire::Service unavailable(0);
    } catch (const std::runtime_error& error) {
      stopped = error.what();
    }
    test_genai_open_error(0);
    require(stopped.starts_with("PCIe service open card failed: -2. ") &&
                stopped.find("Get-Service SiMaMlaDaemon") != std::string::npos &&
                stopped.find("Start-Service SiMaMlaDaemon from an elevated PowerShell") !=
                    std::string::npos &&
                stopped.find(R"(\SimaPCIeHost\bin\sima-pcie-host-doctor.cmd")") !=
                    std::string::npos,
            "service open failure names the stopped Platform service and its fix");
#endif
    rejects([&] {
      host.send(wire::tag(session, false), std::string(SIMAAI_SVC_PAYLOAD_MAX + 1, 'x'));
    });
    std::cout << "GenAI service transport and values: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

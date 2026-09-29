// The bug this guards: pcie-genai checked Ctrl-C only after
// GenerationStream::next() returned, and next() blocks until the card sends
// something. With a silent card (long prefill, or stuck), Ctrl-C did nothing
// until the idle timeout. The CLI now cancels from a CancelWatcher thread.
#include "genai/GenAIModelInternal.h"
#include "genai/GenAIProtocol.h"
#include "genai/SvcTransport.h"
#include "genai_fake_svc_client.h"
#include "pcie_genai_cancel_watcher.h"

#include "simaai/neat/pcie/genai/GenAIModel.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace pgenai = simaai::neat::pcie::genai;
namespace pgi = simaai::neat::pcie::genai::internal;
using pgenai::tools::CancelWatcher;
using pgi::test::FakeSvcClient;
using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

pgi::SvcTransportOptions options() {
  pgi::SvcTransportOptions o;
  o.model_id = "llama";
  o.recv_timeout_ms = 10;
  o.idle_timeout_ms = 5000; // long on purpose: the fix must not rely on it
  o.cancel_timeout_ms = 2000;
  return o;
}

pgenai::GenerationRequest request(const std::string& prompt) {
  pgenai::GenerationRequest r;
  r.prompt = prompt;
  return r;
}

std::string id_of(const std::string& payload) {
  return nlohmann::json::parse(payload)["id"].get<std::string>();
}

std::string final_json(const std::string& id, const std::string& reason) {
  return R"({"id":")" + id + R"(","finish_reason":")" + reason +
         R"(","generated_tokens":0,"ttft":0.0,"tps":0.0})";
}

std::size_t count_tag(const FakeSvcClient& client, const char* tag) {
  const auto sent = client.sent();
  return static_cast<std::size_t>(
      std::count_if(sent.begin(), sent.end(), [&](const pgi::SvcNote& n) { return n.tag == tag; }));
}

} // namespace

int main() {
  try {
    // A card that stays silent after the prompt and answers only a cancel.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* card = fake.get();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagCancel) {
          c.push(pgi::kTagFinal, final_json(id_of(payload), "cancelled"));
        }
      };
      pgenai::GenAIModel model = pgi::make_model_with_transport(
          std::make_unique<pgi::SvcTransport>(std::move(fake), options()));

      std::atomic<bool> interrupt{false};
      std::thread ctrl_c([&] {
        std::this_thread::sleep_for(100ms);
        interrupt.store(true); // the user presses Ctrl-C while no token arrives
      });
      const auto start = clock_type::now();
      std::optional<pgenai::TokenSample> last;
      bool fired = false;
      std::string error;
      {
        pgenai::GenerationStream stream = model.stream(request("Hi"));
        const CancelWatcher watcher([&] { return interrupt.load(); },
                                    [&stream] { stream.cancel(); }, 10ms);
        // Catch here, not in main: ctrl_c must be joined before any throw,
        // or its std::thread destructor calls std::terminate.
        try {
          while (const std::optional<pgenai::TokenSample> sample = stream.next()) {
            last = sample;
          }
        } catch (const std::exception& e) {
          error = e.what(); // without the fix: the idle timeout fires here
        }
        fired = watcher.fired();
      }
      ctrl_c.join();
      const auto waited = clock_type::now() - start;

      require(error.empty(), "a silent answer must not end in an error, got: " + error);
      require(fired, "the watcher must have cancelled the stream");
      require(last && last->is_final && last->finish_reason == "cancelled",
              "a silent answer must end with the card's cancelled final");
      require(waited < 1500ms,
              "Ctrl-C must end a silent answer quickly, not after idle_timeout_ms (5 s)");
      // Review focus 1: one cancel on the wire.
      require(count_tag(*card, pgi::kTagCancel) == 1, "exactly one genai.cancel must be sent");
    }

    // Review focus 3: the card never answers the cancel -> the answer still
    // ends after cancel_timeout_ms, well before idle_timeout_ms.
    {
      pgi::SvcTransportOptions o = options();
      o.cancel_timeout_ms = 200;
      pgenai::GenAIModel model = pgi::make_model_with_transport(
          std::make_unique<pgi::SvcTransport>(std::make_unique<FakeSvcClient>(), o));
      std::atomic<bool> interrupt{true}; // Ctrl-C already pressed
      const auto start = clock_type::now();
      {
        pgenai::GenerationStream stream = model.stream(request("Hi"));
        const CancelWatcher watcher([&] { return interrupt.load(); },
                                    [&stream] { stream.cancel(); }, 10ms);
        while (stream.next()) {
        }
      }
      require(clock_type::now() - start < 1500ms,
              "a mute card must be given up after cancel_timeout_ms");
    }

    // Review focus 2: a cancel that lands after the final does not leak into
    // the next prompt (SvcTransport clears its cancel flag per run).
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* card = fake.get();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "Hi");
          c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
        }
      };
      pgenai::GenAIModel model = pgi::make_model_with_transport(
          std::make_unique<pgi::SvcTransport>(std::move(fake), options()));
      {
        pgenai::GenerationStream first = model.stream(request("one"));
        while (first.next()) {
        }
        first.cancel(); // what a late-firing watcher does
      }
      std::optional<pgenai::TokenSample> last;
      {
        pgenai::GenerationStream second = model.stream(request("two"));
        while (const std::optional<pgenai::TokenSample> sample = second.next()) {
          last = sample;
        }
      }
      require(last && last->is_final && last->finish_reason == "stop",
              "the next prompt must run normally after a late cancel");
      require(count_tag(*card, pgi::kTagCancel) == 0,
              "a cancel after the final must not send genai.cancel");
    }

    std::cout << "[PASS] pcie-genai Ctrl-C cancels a silent answer\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

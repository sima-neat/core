#include "genai/GenAITypes.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace genai = simaai::neat::genai;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    // The public factory drives tokens end-to-end: a producer emits two pieces
    // of text and finishes, and next() delivers them in order plus a final
    // sample. This is the seam the PCIe host backend uses to turn card-side
    // token notifications into a GenerationStream.
    {
      genai::GenerationStream stream = genai::GenerationStream::make(
          [](genai::GenerationStream::Producer& producer) {
            producer.record_text("Hello", false);
            producer.record_text(" World", false);
            producer.finish("stop", 2);
          },
          [] {});

      std::vector<std::string> texts;
      bool saw_final = false;
      std::string final_reason;
      while (const std::optional<genai::TokenSample> sample = stream.next()) {
        if (sample->is_final) {
          saw_final = true;
          final_reason = sample->finish_reason;
        } else {
          texts.push_back(sample->text);
        }
      }

      require(texts.size() == 2, "factory stream must deliver both text tokens");
      require(texts[0] == "Hello" && texts[1] == " World",
              "factory stream must deliver tokens in producer order");
      require(saw_final, "factory stream must deliver a final sample");
      require(final_reason == "stop", "final sample must carry the producer finish reason");
    }

    // The cancel callback wired through the factory fires on cancel(), and a
    // producer that watches cancelled() can stop early.
    {
      std::atomic<bool> cancel_called{false};
      std::atomic<bool> producer_started{false};
      genai::GenerationStream stream = genai::GenerationStream::make(
          [&producer_started](genai::GenerationStream::Producer& producer) {
            producer_started = true;
            while (!producer.cancelled()) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            producer.finish("cancelled", 0);
          },
          [&cancel_called] { cancel_called = true; });

      while (!producer_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      stream.cancel();

      require(cancel_called.load(), "factory stream must invoke the caller's cancel callback");

      // Draining after cancel must terminate (producer observed cancellation).
      while (stream.next()) {
      }
    }

    std::cout << "[PASS] GenerationStream public factory\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

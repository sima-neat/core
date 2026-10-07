// Run one text request, optionally streaming the answer, from the host over PCIe.
#include "tutorial_args.h"

namespace genai = simaai::neat::pcie::genai;

int main(int argc, char** argv) {
  try {
    const auto args = tutorial::parse_args(argc, argv, "llm");

    // STEP load-llm
    genai::GenAIModel model(args.model, args.connection);
    if (!model.accepts_text() || model.task() == genai::GenAITask::ASR)
      throw std::invalid_argument("choose a text-generation model directory");
    // END STEP

    // STEP text-request
    genai::GenerationRequest request;
    request.prompt = args.prompt;
    request.max_new_tokens = args.max_tokens;
    if (!args.stream) {
      const auto result = model.run(request);
      std::cout << result.text << '\n';
      std::cout << "Generated tokens: " << result.metrics.generated_tokens << '\n';
    }
    // END STEP
    else {
      // STEP stream-answer
      auto stream = model.stream(request);
      for (const auto& sample : stream) {
        std::cout << sample.text << std::flush;
        if (sample.is_final)
          std::cout << "\nGenerated tokens: " << sample.metrics.generated_tokens << '\n';
      }
      // END STEP
    }
    // STEP close-model
    model.close();
    // END STEP
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "LLM tutorial: " << error.what() << '\n';
    return 1;
  }
}

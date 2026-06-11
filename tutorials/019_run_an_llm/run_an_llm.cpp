#include "neat/genai.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace genai = simaai::neat::genai;

struct Args {
  std::filesystem::path model;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      args.model = argv[++i];
    } else {
      throw std::runtime_error("usage: run_an_llm --model <llima_model_dir>");
    }
  }
  if (args.model.empty()) {
    throw std::runtime_error("missing required --model <llima_model_dir>");
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    // STEP load-model
    genai::GenAIModel model(args.model);
    // END STEP

    // STEP send-prompt
    genai::GenerationRequest request;
    request.prompt = "Give me three practical tips for designing a small REST API.";
    request.max_new_tokens = 96;

    const genai::GenerationResult first = model.run(request);
    std::cout << "assistant: " << first.text << "\n\n";
    // END STEP

    // STEP system-prompt
    const std::string system_prompt = "You are concise and practical.";

    genai::GenerationRequest concise_request;
    concise_request.system_prompt = system_prompt;
    concise_request.prompt = "Give me one rule of thumb for designing a small REST API.";
    concise_request.max_new_tokens = 64;

    const genai::GenerationResult concise = model.run(concise_request);
    std::cout << "assistant: " << concise.text << "\n\n";
    // END STEP

    // STEP store-history
    std::vector<genai::ChatMessage> messages;
    messages.push_back(genai::ChatMessage{
        .role = "system",
        .content = system_prompt,
    });
    messages.push_back(genai::ChatMessage{
        .role = "user",
        .content = "Give me three practical tips for writing API documentation.",
    });

    genai::GenerationRequest chat_request;
    chat_request.messages = messages;
    chat_request.max_new_tokens = 96;

    const genai::GenerationResult chat_result = model.run(chat_request);
    std::cout << "assistant: " << chat_result.text << "\n\n";
    messages.push_back(genai::ChatMessage{.role = "assistant", .content = chat_result.text});
    // END STEP

    // STEP follow-up
    messages.push_back(genai::ChatMessage{
        .role = "user",
        .content = "Which tip should I apply first for a prototype?",
    });

    genai::GenerationRequest follow_up;
    follow_up.messages = messages;
    follow_up.max_new_tokens = 96;

    const genai::GenerationResult second = model.run(follow_up);
    std::cout << "assistant: " << second.text << "\n\n";
    messages.push_back(genai::ChatMessage{.role = "assistant", .content = second.text});
    // END STEP

    // STEP stream-answer
    messages.push_back(genai::ChatMessage{
        .role = "user",
        .content = "Turn that advice into a short checklist.",
    });

    genai::GenerationRequest streaming_request;
    streaming_request.messages = messages;
    streaming_request.max_new_tokens = 96;

    genai::GenerationStream stream_handle = model.stream(streaming_request);
    std::cout << "assistant: ";
    for (const genai::TokenSample& token : stream_handle) {
      std::cout << token.text << std::flush;
    }
    std::cout << "\n";
    // END STEP

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

#include "neat.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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
      throw std::runtime_error("usage: compose_genai_into_graph --model <llima_model_dir>");
    }
  }
  if (args.model.empty()) {
    throw std::runtime_error("missing required --model <llima_model_dir>");
  }
  return args;
}

simaai::neat::Sample make_text_sample(const std::string& port, const std::string& text) {
  return simaai::neat::make_tensor_sample(port, simaai::neat::Tensor::from_text(text));
}

std::string sample_text(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return sample.tensor->to_text();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && sample.tensors.size() == 1U) {
    return sample.tensors.front().to_text();
  }
  return {};
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    // STEP create-fragment
    auto model = std::make_shared<genai::VisionLanguageModel>(args.model);

    genai::VisionLanguageOptions options;
    options.system_prompt = "You are concise.";
    options.max_new_tokens = 96;
    options.streaming = true;

    simaai::neat::Graph genai_fragment =
        genai::graphs::VisionLanguage(model, options, "genai_stage");
    // END STEP

    // STEP compose-graph
    simaai::neat::Graph app("genai_app");
    app.add(genai_fragment);
    std::cout << app.describe() << "\n";
    // END STEP

    // STEP push-prompt
    simaai::neat::Run run = app.build();
    if (!run.push("prompt", make_text_sample("prompt", "Explain what an API gateway does."))) {
      throw std::runtime_error("push(prompt) failed: " + run.last_error());
    }
    // END STEP

    // STEP pull-results
    std::cout << "assistant: ";
    for (int i = 0; i < 256; ++i) {
      if (auto token = run.pull("tokens", 250)) {
        std::cout << sample_text(*token) << std::flush;
        continue;
      }
      if (auto done = run.pull("done", 10)) {
        (void)done;
        break;
      }
      if (auto error = run.pull("error", 10)) {
        throw std::runtime_error(sample_text(*error));
      }
    }
    std::cout << "\n";
    run.close();
    // END STEP

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

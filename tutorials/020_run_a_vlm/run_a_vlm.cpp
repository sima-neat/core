#include "neat/genai.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace genai = simaai::neat::genai;

struct Args {
  std::filesystem::path model;
  std::filesystem::path image;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      args.model = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      args.image = argv[++i];
    } else {
      throw std::runtime_error("usage: run_a_vlm --model <vlm_model_dir> --image <image>");
    }
  }
  if (args.model.empty() || args.image.empty()) {
    throw std::runtime_error("missing required --model <vlm_model_dir> or --image <image>");
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    // STEP load-inputs
    genai::VisionLanguageModel model(args.model);
    cv::Mat image = cv::imread(args.image.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to read image: " + args.image.string());
    }
    // END STEP

    // STEP direct-image
    genai::GenerationRequest direct;
    direct.prompt = "Describe this image in one sentence.";
    direct.images = {image};
    direct.max_new_tokens = 96;

    const genai::GenerationResult first = model.run(direct);
    std::cout << "direct image: " << first.text << "\n\n";
    // END STEP

    // STEP cache-image
    if (!model.encode(image)) {
      throw std::runtime_error("VLM did not accept the image for caching");
    }
    std::cout << "cached_images=" << model.cached_image_count() << "\n";
    // END STEP

    // STEP follow-up
    genai::GenerationRequest cached;
    cached.prompt = "What details should I inspect more closely?";
    cached.use_cached_images = true;
    cached.max_new_tokens = 96;

    const genai::GenerationResult follow_up = model.run(cached);
    std::cout << "cached image: " << follow_up.text << "\n\n";

    genai::GenerationRequest second_cached;
    second_cached.prompt = "Summarize the image in three keywords.";
    second_cached.use_cached_images = true;
    second_cached.max_new_tokens = 48;

    const genai::GenerationResult second_follow_up = model.run(second_cached);
    std::cout << "cached image keywords: " << second_follow_up.text << "\n\n";
    // END STEP

    // STEP message-image
    genai::ChatMessage image_message;
    image_message.role = "user";
    image_message.content = "What is the main subject of this image?";
    image_message.images = {image};

    genai::GenerationRequest message_request;
    message_request.messages = {image_message};
    message_request.max_new_tokens = 96;

    const genai::GenerationResult message_result = model.run(message_request);
    std::cout << "message image: " << message_result.text << "\n";
    // END STEP

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

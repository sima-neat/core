#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
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
      throw std::runtime_error(
          "usage: compose_genai_into_graph --model <vlm_model_dir> --image <image>");
    }
  }
  if (args.model.empty() || args.image.empty()) {
    throw std::runtime_error("missing required --model <vlm_model_dir> or --image <image>");
  }
  return args;
}

simaai::neat::Sample make_text_sample(const std::string& port, const std::string& text) {
  return simaai::neat::make_tensor_sample(port, simaai::neat::Tensor::from_text(text));
}

simaai::neat::Sample make_image_sample(const std::filesystem::path& image_path) {
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + image_path.string());
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return simaai::neat::make_tensor_sample(
      "image", simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                                 simaai::neat::TensorMemory::CPU));
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
    options.encode_images_on_input = false;

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
    if (!run.push("image", make_image_sample(args.image))) {
      throw std::runtime_error("push(image) failed: " + run.last_error());
    }
    if (!run.push("prompt", make_text_sample("prompt", "Describe this image in one sentence."))) {
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

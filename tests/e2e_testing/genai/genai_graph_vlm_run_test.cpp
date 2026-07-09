#include "genai/VisionLanguageModel.h"
#include "genai/GraphFragments.h"
#include "genai_test_utils.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "pipeline/TensorCore.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

// Exercises the GenAI VisionLanguage graph node end-to-end against a real LLiMa
// VLM, including image encode, cached image generation, EV74 tensors, and errors.
// Model fixture:
//   export LLIMA_MODELS_PATH=/media/nvme/llima/models
//   export SIMA_TEST_LLIMA_VLM_MODEL=LFM2.5-VL-450M-a16w4
//   tests/tools/prepare_genai_models.sh
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kPrompt = "Describe this image in a short phrase.";
constexpr const char* kExpectedText = "Skier in the air.";

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

fs::path resolve_model_dir() {
  return simaai::neat::test::resolve_genai_model_dir(
      kModelEnv, simaai::neat::test::kDefaultVlmModelName, "LLiMa VLM", "devkit/vlm_config.json");
}

fs::path resolve_image_path(const fs::path& repo_root) {
  const fs::path image = repo_root / "tests" / "images" / "people.jpg";
  if (!fs::is_regular_file(image)) {
    throw std::runtime_error("missing VLM e2e image fixture: " + image.string());
  }
  return image;
}

simaai::neat::Sample make_text_input(const char* port_name, std::string text,
                                     int64_t frame_id = 1) {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample(port_name, simaai::neat::Tensor::from_text(text));
  sample.frame_id = frame_id;
  sample.stream_id = "genai-graph-vlm";
  sample.pts_ns = frame_id * 1000;
  sample.duration_ns = 1000;
  return sample;
}

simaai::neat::Tensor rgb_tensor_from_bgr(const cv::Mat& bgr, simaai::neat::TensorMemory memory) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, memory);
}

simaai::neat::Sample make_image_input(const cv::Mat& image_bgr, simaai::neat::TensorMemory memory,
                                      int64_t frame_id) {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample("image", rgb_tensor_from_bgr(image_bgr, memory));
  sample.frame_id = frame_id;
  sample.stream_id = "genai-graph-vlm";
  sample.pts_ns = frame_id * 1000;
  sample.duration_ns = 1000;
  return sample;
}

simaai::neat::Sample make_invalid_image_input() {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample("image", simaai::neat::Tensor::from_text("not-an-image"));
  sample.frame_id = 99;
  sample.stream_id = "genai-graph-vlm";
  return sample;
}

std::string sample_text(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    require(sample.tensor.has_value(), "Tensor sample missing tensor");
    return sample.tensor->to_text();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    require(sample.tensors.size() == 1U, "TensorSet sample should carry one tensor");
    return sample.tensors.front().to_text();
  }
  throw std::runtime_error("sample is not text");
}

std::string bundle_field_text(const simaai::neat::Sample& bundle, const std::string& name) {
  require(bundle.kind == simaai::neat::SampleKind::Bundle, "bundle sample expected");
  for (const auto& field : bundle.fields) {
    if (field.port_name == name || field.stream_label == name) {
      return sample_text(field);
    }
  }
  throw std::runtime_error("missing bundle field: " + name);
}

struct GraphOutputs {
  std::string tokens;
  simaai::neat::Sample done;
  simaai::neat::Sample encoded;
  std::string error;
  int token_samples = 0;
  bool saw_done = false;
  bool saw_encoded = false;
  bool saw_error = false;
};

GraphOutputs pull_until_done_or_error(simaai::neat::Run& run) {
  GraphOutputs outputs;
  for (int i = 0; i < 256; ++i) {
    if (auto sample = run.pull("tokens", 250)) {
      outputs.tokens += sample_text(*sample);
      outputs.token_samples += 1;
      continue;
    }
    if (auto sample = run.pull("encoded", 10)) {
      outputs.encoded = *sample;
      outputs.saw_encoded = true;
      continue;
    }
    if (auto sample = run.pull("done", 10)) {
      outputs.done = *sample;
      outputs.saw_done = true;
      break;
    }
    if (auto sample = run.pull("error", 10)) {
      outputs.error = sample_text(*sample);
      outputs.saw_error = true;
      break;
    }
  }
  return outputs;
}

simaai::neat::Sample pull_encoded(simaai::neat::Run& run) {
  for (int i = 0; i < 16; ++i) {
    if (auto sample = run.pull("encoded", 60000)) {
      return *sample;
    }
    if (auto sample = run.pull("error", 10)) {
      throw std::runtime_error("GenAI graph VLM image input emitted error: " +
                               sample_text(*sample));
    }
  }
  throw std::runtime_error("GenAI graph VLM image input did not emit encoded sample");
}

void require_vlm_outputs(const GraphOutputs& outputs, const std::string& label,
                         bool expect_streaming) {
  require(outputs.saw_done, label + " did not emit done");
  require(!outputs.saw_error, label + " emitted error: " + outputs.error);
  require(outputs.token_samples > 0, label + " should emit token samples");
  if (!expect_streaming) {
    require(outputs.token_samples == 1, label + " should emit one sync token sample");
  }
  std::cout << label << " text=" << outputs.tokens << "\n";
  require(trim_text(outputs.tokens) == kExpectedText,
          label + " generated unexpected text: " + outputs.tokens);
  const std::string finish_reason = bundle_field_text(outputs.done, "finish_reason");
  require(finish_reason == "stop" || finish_reason == "interrupted",
          label + " returned unexpected finish reason: " + finish_reason);
  require(std::stoul(bundle_field_text(outputs.done, "generated_tokens")) > 0,
          label + " done should report generated tokens");
}

simaai::neat::Run build_vlm_run(std::shared_ptr<simaai::neat::genai::VisionLanguageModel> model,
                                simaai::neat::genai::VisionLanguageOptions options,
                                const std::string& label) {
  simaai::neat::Graph graph;
  graph.add(
      simaai::neat::genai::graphs::VisionLanguage(std::move(model), std::move(options), label));
  return graph.build();
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_graph_vlm_run_test requires repository root argument");
    }

    const fs::path model_dir = resolve_model_dir();
    const fs::path image_path = resolve_image_path(fs::path(argv[1]));

    std::cout << "GENAI_GRAPH_VLM model_dir=" << model_dir << "\n";
    std::cout << "GENAI_GRAPH_VLM image=" << image_path << "\n";

    cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    require(!image_bgr.empty(), "failed to load VLM e2e image: " + image_path.string());

    auto model = std::make_shared<simaai::neat::genai::VisionLanguageModel>(model_dir);
    require(model->accepts_image(), "VLM model should accept image input");

    simaai::neat::Run direct_streaming_run =
        build_vlm_run(model,
                      simaai::neat::genai::VisionLanguageOptions{
                          .max_new_tokens = 48, .streaming = true, .encode_images_on_input = false},
                      "vision_language_direct_streaming");
    simaai::neat::Run direct_sync_run = build_vlm_run(
        model,
        simaai::neat::genai::VisionLanguageOptions{
            .max_new_tokens = 48, .streaming = false, .encode_images_on_input = false},
        "vision_language_direct_sync");
    simaai::neat::Run cached_encode_run =
        build_vlm_run(model,
                      simaai::neat::genai::VisionLanguageOptions{
                          .max_new_tokens = 48, .streaming = true, .encode_images_on_input = true},
                      "vision_language_cached_encode");
    simaai::neat::Run missing_image_run =
        build_vlm_run(model,
                      simaai::neat::genai::VisionLanguageOptions{
                          .max_new_tokens = 48, .streaming = true, .encode_images_on_input = false},
                      "vision_language_missing_image");

    require(direct_streaming_run.push(
                "image", make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 1)),
            "Run::push CPU image failed");
    simaai::neat::Sample encoded = pull_encoded(direct_streaming_run);
    require(bundle_field_text(encoded, "mode") == "direct", "CPU image should stay direct");

    require(direct_streaming_run.push("prompt", make_text_input("prompt", kPrompt, 2)),
            "Run::push direct streaming prompt failed");
    require_vlm_outputs(pull_until_done_or_error(direct_streaming_run),
                        "GENAI_GRAPH_VLM_DIRECT_STREAMING", true);

    require(direct_streaming_run.push(
                "image", make_image_input(image_bgr, simaai::neat::TensorMemory::EV74, 3)),
            "Run::push EV74 image failed");
    encoded = pull_encoded(direct_streaming_run);
    require(bundle_field_text(encoded, "mode") == "direct", "EV74 image should stay direct");

    require(direct_streaming_run.push("prompt", make_text_input("prompt", kPrompt, 4)),
            "Run::push EV74 direct prompt failed");
    require_vlm_outputs(pull_until_done_or_error(direct_streaming_run),
                        "GENAI_GRAPH_VLM_EV74_DIRECT_STREAMING", true);

    require(direct_sync_run.push("image",
                                 make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 5)),
            "Run::push direct image failed");
    encoded = pull_encoded(direct_sync_run);
    require(bundle_field_text(encoded, "mode") == "direct", "direct image should stay direct");

    require(
        direct_sync_run.push("use_cached_image", make_text_input("use_cached_image", "false", 6)),
        "Run::push direct control failed");
    require(direct_sync_run.push("prompt", make_text_input("prompt", kPrompt, 7)),
            "Run::push direct sync prompt failed");
    require_vlm_outputs(pull_until_done_or_error(direct_sync_run), "GENAI_GRAPH_VLM_DIRECT_SYNC",
                        false);

    require(cached_encode_run.push("image",
                                   make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 8)),
            "Run::push cached encode image failed");
    encoded = pull_encoded(cached_encode_run);
    require(bundle_field_text(encoded, "mode") == "cached", "cached image should emit cached mode");

    require(cached_encode_run.push("prompt", make_text_input("prompt", kPrompt, 9)),
            "Run::push cached prompt failed");
    require_vlm_outputs(pull_until_done_or_error(cached_encode_run), "GENAI_GRAPH_VLM_CACHED",
                        true);

    require(direct_streaming_run.push("image", make_invalid_image_input()),
            "Run::push invalid image failed");
    GraphOutputs error_outputs = pull_until_done_or_error(direct_streaming_run);
    require(error_outputs.saw_error, "invalid image should emit error");
    require(!error_outputs.error.empty(), "invalid image error should be non-empty");

    require(missing_image_run.push("prompt", make_text_input("prompt", kPrompt, 10)),
            "Run::push missing image prompt failed");
    error_outputs = pull_until_done_or_error(missing_image_run);
    require(error_outputs.saw_error, "missing image prompt should emit error");
    require(!error_outputs.error.empty(), "missing image error should be non-empty");

    direct_streaming_run.stop();
    direct_sync_run.stop();
    cached_encode_run.stop();
    missing_image_run.stop();
    std::cout << "[OK] genai_graph_vlm_run_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

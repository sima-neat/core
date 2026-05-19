#include "genai/VisionLanguageModel.h"
#include "genai/nodes/VisionLanguage.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"
#include "graph/GraphSession.h"
#include "pipeline/SessionOptions.h"
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
//   hf download simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4 --local-dir <model-dir>
//   export SIMA_TEST_LLIMA_VLM_MODEL=<model-dir>
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kPrompt = "Describe this image in a short phrase.";
constexpr const char* kExpectedText =
    "A skier soars through the air above a snowy slope, with spectators watching below.";

bool has_llima_vlm_config(const fs::path& model_dir) {
  std::error_code ec;
  return fs::is_regular_file(model_dir / "devkit" / "vlm_config.json", ec) && !ec;
}

std::string trim_env_value(const char* value) {
  if (value == nullptr) {
    return {};
  }

  std::string out(value);
  const auto first = out.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

fs::path resolve_model_dir() {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (env_model_dir.empty()) {
    skip_long_test_exception("set " + std::string(kModelEnv) +
                             " to an existing LLiMa VLM model directory");
  }

  fs::path model_dir(env_model_dir);
  if (has_llima_vlm_config(model_dir)) {
    return model_dir;
  }

  skip_long_test_exception(std::string(kModelEnv) +
                           " does not point to a LLiMa VLM model directory: " + model_dir.string());
  return {};
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

GraphOutputs pull_until_done_or_error(simaai::neat::graph::GraphRun& run,
                                      simaai::neat::graph::NodeId node_id) {
  GraphOutputs outputs;
  for (int i = 0; i < 128; ++i) {
    auto sample = run.pull(node_id, 60000);
    require(sample.has_value(), "GraphRun::pull timed out");
    if (sample->port_name == "tokens" || sample->stream_label == "tokens") {
      outputs.tokens += sample_text(*sample);
      outputs.token_samples += 1;
    } else if (sample->port_name == "done" || sample->stream_label == "done") {
      outputs.done = *sample;
      outputs.saw_done = true;
      break;
    } else if (sample->port_name == "encoded" || sample->stream_label == "encoded") {
      outputs.encoded = *sample;
      outputs.saw_encoded = true;
    } else if (sample->port_name == "error" || sample->stream_label == "error") {
      outputs.error = sample_text(*sample);
      outputs.saw_error = true;
      break;
    } else {
      throw std::runtime_error("unexpected graph output port: " + sample->port_name);
    }
  }
  return outputs;
}

simaai::neat::Sample pull_encoded(simaai::neat::graph::GraphRun& run,
                                  simaai::neat::graph::NodeId node_id) {
  for (int i = 0; i < 16; ++i) {
    auto sample = run.pull(node_id, 60000);
    require(sample.has_value(), "GraphRun::pull encoded timed out");
    if (sample->port_name == "encoded" || sample->stream_label == "encoded") {
      return *sample;
    }
    if (sample->port_name == "error" || sample->stream_label == "error") {
      throw std::runtime_error("GenAI graph VLM image input emitted error: " +
                               sample_text(*sample));
    }
    throw std::runtime_error("unexpected graph output before encoded: " + sample->port_name);
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
    require(model->accepts_image(), "Qwen3-VL model should accept image input");

    simaai::neat::graph::Graph graph;
    const auto prompt_port = graph.intern_port("prompt");
    const auto image_port = graph.intern_port("image");
    const auto use_cached_image_port = graph.intern_port("use_cached_image");

    const auto direct_streaming_node = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .max_new_tokens = 48, .streaming = true, .encode_images_on_input = false},
        "vision_language_direct_streaming"));
    const auto direct_sync_node = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .max_new_tokens = 48, .streaming = false, .encode_images_on_input = false},
        "vision_language_direct_sync"));
    const auto unsupported_cached_node = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .max_new_tokens = 48, .streaming = true, .encode_images_on_input = true},
        "vision_language_unsupported_cached"));
    const auto missing_image_node = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .max_new_tokens = 48, .streaming = true, .encode_images_on_input = false},
        "vision_language_missing_image"));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(direct_streaming_node, image_port,
                     make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 1)),
            "GraphRun::push CPU image failed");
    simaai::neat::Sample encoded = pull_encoded(run, direct_streaming_node);
    require(bundle_field_text(encoded, "mode") == "direct", "CPU image should stay direct");

    require(run.push(direct_streaming_node, prompt_port, make_text_input("prompt", kPrompt, 2)),
            "GraphRun::push direct streaming prompt failed");
    require_vlm_outputs(pull_until_done_or_error(run, direct_streaming_node),
                        "GENAI_GRAPH_VLM_DIRECT_STREAMING", true);

    require(run.push(direct_streaming_node, image_port,
                     make_image_input(image_bgr, simaai::neat::TensorMemory::EV74, 3)),
            "GraphRun::push EV74 image failed");
    encoded = pull_encoded(run, direct_streaming_node);
    require(bundle_field_text(encoded, "mode") == "direct", "EV74 image should stay direct");

    require(run.push(direct_streaming_node, prompt_port, make_text_input("prompt", kPrompt, 4)),
            "GraphRun::push EV74 direct prompt failed");
    require_vlm_outputs(pull_until_done_or_error(run, direct_streaming_node),
                        "GENAI_GRAPH_VLM_EV74_DIRECT_STREAMING", true);

    require(run.push(direct_sync_node, image_port,
                     make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 5)),
            "GraphRun::push direct image failed");
    encoded = pull_encoded(run, direct_sync_node);
    require(bundle_field_text(encoded, "mode") == "direct", "direct image should stay direct");

    require(run.push(direct_sync_node, use_cached_image_port,
                     make_text_input("use_cached_image", "false", 6)),
            "GraphRun::push direct control failed");
    require(run.push(direct_sync_node, prompt_port, make_text_input("prompt", kPrompt, 7)),
            "GraphRun::push direct sync prompt failed");
    require_vlm_outputs(pull_until_done_or_error(run, direct_sync_node),
                        "GENAI_GRAPH_VLM_DIRECT_SYNC", false);

    require(run.push(unsupported_cached_node, image_port,
                     make_image_input(image_bgr, simaai::neat::TensorMemory::CPU, 8)),
            "GraphRun::push unsupported cached image failed");
    GraphOutputs error_outputs = pull_until_done_or_error(run, unsupported_cached_node);
    require(error_outputs.saw_error, "unsupported cached image should emit error");
    require(error_outputs.error.find("cached reuse is not supported") != std::string::npos,
            "unsupported cached image error should explain cached reuse support");

    require(run.push(direct_streaming_node, image_port, make_invalid_image_input()),
            "GraphRun::push invalid image failed");
    error_outputs = pull_until_done_or_error(run, direct_streaming_node);
    require(error_outputs.saw_error, "invalid image should emit error");
    require(!error_outputs.error.empty(), "invalid image error should be non-empty");

    require(run.push(missing_image_node, prompt_port, make_text_input("prompt", kPrompt, 9)),
            "GraphRun::push missing image prompt failed");
    error_outputs = pull_until_done_or_error(run, missing_image_node);
    require(error_outputs.saw_error, "missing image prompt should emit error");
    require(!error_outputs.error.empty(), "missing image error should be non-empty");

    run.stop();
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

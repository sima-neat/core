#include "genai/GenAITypes.h"
#include "genai/GenAIModel.h"
#include "genai/VisionLanguageModel.h"
#include "genai_test_utils.h"
#include "pipeline/TensorCore.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

// Exercises direct VisionLanguageModel image generation against a real LLiMa
// VLM using tests/images/people.jpg.
// Model fixture:
//   export LLIMA_MODELS_PATH=/media/nvme/llima/models
//   export SIMA_TEST_LLIMA_VLM_MODEL=LFM2.5-VL-450M-Autoround-a16w4
//   tests/tools/prepare_genai_models.sh
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kPrompt = "Describe this image in a short phrase.";

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

simaai::neat::Tensor rgb_tensor_from_bgr(const cv::Mat& bgr, simaai::neat::TensorMemory memory) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, memory);
}

void require_generation_result(const simaai::neat::genai::GenerationResult& result,
                               const std::string& label) {
  const std::string text = trim_text(result.text);
  require(!text.empty(), label + " expected non-empty generated text");
  std::cout << label << " generated_tokens=" << result.metrics.generated_tokens
            << " ttft_s=" << result.metrics.time_to_first_token_s
            << " tps=" << result.metrics.tokens_per_second
            << " finish_reason=" << result.finish_reason << "\n";
  std::cout << label << " text=" << result.text << "\n";
  require(result.metrics.generated_tokens > 0, label + " expected generated tokens");
  require(result.finish_reason == "stop" || result.finish_reason == "length" ||
              result.finish_reason == "interrupted",
          label + " returned unexpected finish reason: " + result.finish_reason);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_vlm_run_test requires repository root argument");
    }

    const fs::path model_dir = resolve_model_dir();
    const fs::path image_path = resolve_image_path(fs::path(argv[1]));

    std::cout << "GENAI_VLM model_dir=" << model_dir << "\n";
    std::cout << "GENAI_VLM image=" << image_path << "\n";

    cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    require(!image_bgr.empty(), "failed to load VLM e2e image: " + image_path.string());

    {
      simaai::neat::genai::VisionLanguageModel model(model_dir);
      require(model.accepts_image(), "VLM model should accept image input");

      simaai::neat::genai::GenerationRequest cv_request;
      cv_request.prompt = std::string{kPrompt};
      cv_request.images = {image_bgr};
      cv_request.max_new_tokens = 48;
      require_generation_result(model.run(cv_request), "GENAI_VLM_CVMAT");

      simaai::neat::genai::GenerationRequest tensor_request;
      tensor_request.prompt = std::string{kPrompt};
      tensor_request.images = {rgb_tensor_from_bgr(image_bgr, simaai::neat::TensorMemory::CPU)};
      tensor_request.max_new_tokens = 48;
      require_generation_result(model.run(tensor_request), "GENAI_VLM_TENSOR");

      simaai::neat::genai::GenerationRequest ev74_tensor_request;
      ev74_tensor_request.prompt = std::string{kPrompt};
      ev74_tensor_request.images = {
          rgb_tensor_from_bgr(image_bgr, simaai::neat::TensorMemory::EV74)};
      ev74_tensor_request.max_new_tokens = 48;
      require_generation_result(model.run(ev74_tensor_request), "GENAI_VLM_EV74_TENSOR");

      auto stream = model.stream(tensor_request);
      std::string streamed_text;
      bool saw_stream_chunk = false;
      bool saw_stream_final = false;
      simaai::neat::genai::TokenSample final_sample;
      while (auto sample = stream.next()) {
        if (sample->is_final) {
          saw_stream_final = true;
          final_sample = *sample;
          break;
        }
        saw_stream_chunk = true;
        streamed_text += sample->text;
      }
      require(saw_stream_chunk, "GENAI_VLM_STREAM expected at least one text chunk");
      require(saw_stream_final, "GENAI_VLM_STREAM expected final sample");
      require(final_sample.metrics.generated_tokens > 0,
              "GENAI_VLM_STREAM expected generated tokens");
      require(final_sample.finish_reason == "stop" || final_sample.finish_reason == "length" ||
                  final_sample.finish_reason == "interrupted",
              "GENAI_VLM_STREAM returned unexpected finish reason: " + final_sample.finish_reason);
      std::cout << "GENAI_VLM_STREAM generated_tokens=" << final_sample.metrics.generated_tokens
                << " ttft_s=" << final_sample.metrics.time_to_first_token_s
                << " tps=" << final_sample.metrics.tokens_per_second
                << " finish_reason=" << final_sample.finish_reason << "\n";
      std::cout << "GENAI_VLM_STREAM text=" << streamed_text << "\n";
      require(!trim_text(streamed_text).empty(),
              "GENAI_VLM_STREAM expected non-empty generated text");

      // The completed handle stays alive while the next stream owns the same model.
      auto next = model.stream(tensor_request);
      auto chunk = next.next();
      require(chunk && !chunk->is_final, "expected an active second stream");
      stream.cancel();
      std::string next_text = chunk->text;
      while ((chunk = next.next()) && !chunk->is_final)
        next_text += chunk->text;
      require(chunk && chunk->finish_reason == final_sample.finish_reason &&
                  chunk->metrics.generated_tokens == final_sample.metrics.generated_tokens &&
                  next_text == streamed_text,
              "cancelling the completed handle interrupted the second stream");

      require(model.encode(std::vector<cv::Mat>{image_bgr}), "GENAI_VLM_ENCODE failed");
      require(model.cached_image_count() == 1U, "GENAI_VLM_ENCODE expected one cached image");

      simaai::neat::genai::GenerationRequest cached_request;
      cached_request.prompt = std::string{kPrompt};
      cached_request.use_cached_images = true;
      cached_request.max_new_tokens = 48;
      const auto cached_result = model.run(cached_request);
      require_generation_result(cached_result, "GENAI_VLM_CACHED");

      auto a = model.create_session();
      auto b = model.create_session();
      auto other = cached_request;
      other.prompt = "Which objects are visible?";
      const auto expected_b = model.run(other);
      require(a.run(cached_request).text == cached_result.text &&
                  b.run(other).text == expected_b.text,
              "independent session output differed from the default session");
      a.reset();
      require(b.run(other).text == expected_b.text &&
                  a.run(cached_request).text == cached_result.text,
              "resetting one session affected another conversation");

      auto retained = [&] {
        auto temporary = model.create_session();
        return temporary.stream(cached_request);
      }();
      std::string retained_text;
      while (auto sample = retained.next()) {
        if (sample->is_final) {
          require(sample->finish_reason == cached_result.finish_reason,
                  "session stream ended early");
          break;
        }
        retained_text += sample->text;
      }
      require(retained_text == cached_result.text,
              "stream lost its session when the handle was released");
      auto next_session = model.create_session();
      next_session.reset();
    }

    simaai::neat::genai::GenAIModel generic_model(model_dir);
    require(generic_model.task() == simaai::neat::genai::GenAITask::VisionLanguage,
            "GenAIModel VLM task mismatch");
    require(generic_model.accepts_text(), "GenAIModel VLM should accept text");
    require(generic_model.accepts_image(), "GenAIModel VLM should accept images");
    require(!generic_model.accepts_audio(), "GenAIModel VLM should not accept audio");

    simaai::neat::genai::GenerationRequest generic_request;
    generic_request.prompt = std::string{kPrompt};
    generic_request.images = {rgb_tensor_from_bgr(image_bgr, simaai::neat::TensorMemory::CPU)};
    generic_request.max_new_tokens = 48;
    require_generation_result(generic_model.run(generic_request), "GENAI_MODEL_VLM");

    std::cout << "[OK] genai_vlm_run_test passed\n";
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

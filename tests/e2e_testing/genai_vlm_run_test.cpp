#include "genai/GenAITypes.h"
#include "genai/GenAIModel.h"
#include "genai/VisionLanguageModel.h"
#include "pipeline/TensorCore.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <unistd.h>

// Exercises direct VisionLanguageModel image generation against a real LLiMa
// VLM using tests/images/people.jpg.
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kRepoId = "simaai/LFM2-VL-450M-a16w4";
constexpr const char* kModelName = "LFM2-VL-450M-a16w4";
constexpr const char* kPrompt = "Describe this image in a short phrase.";
constexpr const char* kExpectedText = "Skier jumping high in the air.";

std::string shell_quote(const fs::path& path) {
  std::string in = path.string();
  std::string out = "'";
  for (char c : in) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool command_exists(const char* command) {
  std::string cmd = "command -v ";
  cmd += command;
  cmd += " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

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

class AutoModelDir {
public:
  AutoModelDir() = default;

  explicit AutoModelDir(fs::path path) : path_(std::move(path)), owned_(true) {}

  AutoModelDir(const AutoModelDir&) = delete;
  AutoModelDir& operator=(const AutoModelDir&) = delete;

  AutoModelDir(AutoModelDir&& other) noexcept
      : path_(std::move(other.path_)), owned_(other.owned_) {
    other.owned_ = false;
  }

  AutoModelDir& operator=(AutoModelDir&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    cleanup();
    path_ = std::move(other.path_);
    owned_ = other.owned_;
    other.owned_ = false;
    return *this;
  }

  ~AutoModelDir() {
    cleanup();
  }

  const fs::path& path() const {
    return path_;
  }

private:
  void cleanup() {
    if (!owned_ || path_.empty()) {
      return;
    }
    std::error_code ec;
    fs::remove_all(path_, ec);
    owned_ = false;
  }

  fs::path path_;
  bool owned_ = false;
};

AutoModelDir download_model_to_nvme() {
  if (!command_exists("hf")) {
    skip_long_test_exception("missing Hugging Face CLI command 'hf'; set " +
                             std::string(kModelEnv) + " to an existing " + kRepoId +
                             " model directory");
  }

  const fs::path root =
      fs::path("/media/nvme/tmp") /
      ("neat-genai-vlm-e2e-" + std::to_string(static_cast<long long>(::getpid())));
  const fs::path model_dir = root / kModelName;

  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    skip_long_test_exception("failed to create temporary model directory " + root.string() + ": " +
                             ec.message());
  }

  std::ostringstream cmd;
  cmd << "hf download " << kRepoId << " --local-dir " << shell_quote(model_dir);
  const int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    skip_long_test_exception("failed to download " + std::string(kRepoId) + " with hf");
  }

  if (!has_llima_vlm_config(model_dir)) {
    throw std::runtime_error("downloaded model is missing devkit/vlm_config.json: " +
                             model_dir.string());
  }

  return AutoModelDir(root);
}

fs::path resolve_model_dir(AutoModelDir& auto_dir) {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (!env_model_dir.empty()) {
    fs::path model_dir(env_model_dir);
    if (has_llima_vlm_config(model_dir)) {
      return model_dir;
    }

    std::cout << "[WARN] " << kModelEnv
              << " does not point to a LLiMa VLM model directory, falling back to temporary "
                 "download: "
              << model_dir << "\n";
  }

  auto_dir = download_model_to_nvme();
  return auto_dir.path() / kModelName;
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
  require(text == kExpectedText, label + " generated unexpected text: " + result.text);
  require(result.metrics.generated_tokens > 0, label + " expected generated tokens");
  require(result.finish_reason == "stop" || result.finish_reason == "interrupted",
          label + " returned unexpected finish reason: " + result.finish_reason);

  std::cout << label << " generated_tokens=" << result.metrics.generated_tokens
            << " ttft_s=" << result.metrics.time_to_first_token_s
            << " tps=" << result.metrics.tokens_per_second
            << " finish_reason=" << result.finish_reason << "\n";
  std::cout << label << " text=" << result.text << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_vlm_run_test requires repository root argument");
    }

    AutoModelDir auto_dir;
    const fs::path model_dir = resolve_model_dir(auto_dir);
    const fs::path image_path = resolve_image_path(fs::path(argv[1]));

    std::cout << "GENAI_VLM model_dir=" << model_dir << "\n";
    std::cout << "GENAI_VLM image=" << image_path << "\n";

    cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    require(!image_bgr.empty(), "failed to load VLM e2e image: " + image_path.string());

    {
      simaai::neat::genai::VisionLanguageModel model(model_dir);
      require(model.accepts_image(), "LFM2-VL model should accept image input");

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
      require(final_sample.finish_reason == "stop" || final_sample.finish_reason == "interrupted",
              "GENAI_VLM_STREAM returned unexpected finish reason: " + final_sample.finish_reason);
      require(trim_text(streamed_text) == kExpectedText,
              "GENAI_VLM_STREAM generated unexpected text: " + streamed_text);
      std::cout << "GENAI_VLM_STREAM generated_tokens=" << final_sample.metrics.generated_tokens
                << " ttft_s=" << final_sample.metrics.time_to_first_token_s
                << " tps=" << final_sample.metrics.tokens_per_second
                << " finish_reason=" << final_sample.finish_reason << "\n";
      std::cout << "GENAI_VLM_STREAM text=" << streamed_text << "\n";

      require(model.encode(std::vector<cv::Mat>{image_bgr}), "GENAI_VLM_ENCODE failed");
      require(model.cached_image_count() == 1U, "GENAI_VLM_ENCODE expected one cached image");

      simaai::neat::genai::GenerationRequest cached_request;
      cached_request.prompt = std::string{kPrompt};
      cached_request.use_cached_images = true;
      cached_request.max_new_tokens = 48;
      require_generation_result(model.run(cached_request), "GENAI_VLM_CACHED");
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

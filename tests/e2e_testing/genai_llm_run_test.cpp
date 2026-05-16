#include "genai/GenAITypes.h"
#include "genai/VisionLanguageModel.h"
#include "test_utils.h"

#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kRepoId = "simaai/LFM2-350M-a16w4";
constexpr const char* kModelName = "LFM2-350M-a16w4";

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

  const fs::path& path() const { return path_; }

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
                             std::string(kModelEnv) + " to an existing model directory");
  }

  const fs::path root =
      fs::path("/media/nvme/tmp") /
      ("neat-genai-e2e-" + std::to_string(static_cast<long long>(::getpid())));
  const fs::path model_dir = root / kModelName;

  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    skip_long_test_exception("failed to create temporary model directory " + root.string() +
                             ": " + ec.message());
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

} // namespace

int main() {
  try {
    AutoModelDir auto_dir;
    const fs::path model_dir = resolve_model_dir(auto_dir);

    std::cout << "GENAI_LLM model_dir=" << model_dir << "\n";

    simaai::neat::genai::VisionLanguageModel model(model_dir);
    require(!model.accepts_image(), "Text-only LLiMa model should not accept image input");

    simaai::neat::genai::GenerationRequest request;
    request.system_prompt = "You are concise.";
    request.prompt = "What is the capital of Germany?";
    request.max_new_tokens = 24;

    const auto result = model.run(request);
    require(!result.text.empty(), "GenAI LLM e2e expected non-empty generated text");
    const std::string normalized_text = trim_text(result.text);
    require(normalized_text == "The capital of Germany is Berlin.",
            "GenAI LLM e2e generated unexpected text: " + result.text);
    require(result.finish_reason == "stop" || result.finish_reason == "interrupted",
            "GenAI LLM e2e returned unexpected finish reason: " + result.finish_reason);

    std::cout << "GENAI_LLM generated_tokens=" << result.metrics.generated_tokens
              << " ttft_s=" << result.metrics.time_to_first_token_s
              << " tps=" << result.metrics.tokens_per_second
              << " finish_reason=" << result.finish_reason << "\n";
    std::cout << "GENAI_LLM text=" << result.text << "\n";

    auto stream = model.stream(request);
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

    require(saw_stream_chunk, "GenAI LLM stream e2e expected at least one text chunk");
    require(saw_stream_final, "GenAI LLM stream e2e expected final sample");
    require(final_sample.finish_reason == "stop" || final_sample.finish_reason == "interrupted",
            "GenAI LLM stream e2e returned unexpected finish reason: " +
                final_sample.finish_reason);
    require(trim_text(streamed_text) == "The capital of Germany is Berlin.",
            "GenAI LLM stream e2e generated unexpected text: " + streamed_text);

    std::cout << "GENAI_LLM_STREAM generated_tokens=" << final_sample.metrics.generated_tokens
              << " ttft_s=" << final_sample.metrics.time_to_first_token_s
              << " tps=" << final_sample.metrics.tokens_per_second
              << " finish_reason=" << final_sample.finish_reason << "\n";
    std::cout << "GENAI_LLM_STREAM text=" << streamed_text << "\n";
    std::cout << "[OK] genai_llm_run_test passed\n";
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

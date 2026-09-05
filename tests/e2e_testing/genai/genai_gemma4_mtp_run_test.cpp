#include "genai/VisionLanguageModel.h"
#include "genai_test_utils.h"
#include "test_utils.h"

#include <filesystem>
#include <iostream>
#include <string>

// Exercises Core's method-aware dispatch against a deployed Gemma4 MTP target/draft package.
// Model fixture:
//   export LLIMA_MODELS_PATH=/media/nvme/llima/models
//   export SIMA_TEST_LLIMA_GEMMA4_MTP_MODEL=Gemma-4-E4B-it-TextOnly-MTP-deployed
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_GEMMA4_MTP_MODEL";
constexpr const char* kDefaultModelName = "Gemma-4-E4B-it-TextOnly-MTP-deployed";
constexpr const char* kTargetConfig =
    "Gemma-4-E4B-it-TextOnly-GPTQ-Safetensors/devkit/vlm_config.json";

} // namespace

int main() {
  try {
    const fs::path model_dir = simaai::neat::test::resolve_genai_model_dir(
        kModelEnv, kDefaultModelName, "Gemma4 MTP", kTargetConfig);

    simaai::neat::genai::GenAIModelOptions options;
    options.max_kv_cache_slots = 1;
    simaai::neat::genai::VisionLanguageModel model(model_dir, options);
    require(!model.accepts_image(), "Gemma4 MTP text-only model should not accept images");

    simaai::neat::genai::GenerationRequest request;
    request.prompt = std::string{"What is the capital of Germany? Answer in one word."};
    request.max_new_tokens = 24;
    request.cache_id = std::string{"gemma4-mtp-session"};

    const auto result = model.run(request);
    require(result.metrics.generated_tokens > 0U,
            "Gemma4 MTP generation should produce at least one token");
    require(!result.text.empty() || !result.reasoning.empty(),
            "Gemma4 MTP generation should produce decoded output");
    require(result.metrics.cache_created, "Gemma4 MTP generation should allocate a cache slot");
    require(model.kv_cache_count() == 1U, "Gemma4 MTP cache assignment was not retained");
    require(model.remove_kv_cache("gemma4-mtp-session"),
            "Gemma4 MTP cache assignment should be removable");

    std::cout << "GENAI_GEMMA4_MTP generated_tokens=" << result.metrics.generated_tokens
              << " ttft_s=" << result.metrics.time_to_first_token_s
              << " tps=" << result.metrics.tokens_per_second << "\n";
    std::cout << "[OK] genai_gemma4_mtp_run_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] genai_gemma4_mtp_run_test: " << e.what() << "\n";
    return 1;
  }
}

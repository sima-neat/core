#include "genai/GenAITypes.h"
#include "genai/GenAIModel.h"
#include "genai/VisionLanguageModel.h"
#include "test_utils.h"

#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

// Exercises direct VisionLanguageModel text generation and streaming against a
// real LLiMa text model.
// Model fixture:
//   hf download simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4 --local-dir <model-dir>
//   export SIMA_TEST_LLIMA_TEXT_MODEL=<model-dir>
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";

void require_throws(const std::function<void()>& fn, const std::string& label) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(label + " should throw");
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

std::string first_tool_call_name(const simaai::neat::genai::Json& tool_calls) {
  if (!tool_calls.is_array() || tool_calls.empty()) {
    return {};
  }
  const auto& first_call = tool_calls.at(0);
  if (first_call.contains("function") && first_call.at("function").is_object() &&
      first_call.at("function").contains("name") &&
      first_call.at("function").at("name").is_string()) {
    return first_call.at("function").at("name").get<std::string>();
  }
  if (first_call.contains("name") && first_call.at("name").is_string()) {
    return first_call.at("name").get<std::string>();
  }
  return {};
}

simaai::neat::genai::GenerationRequest make_tool_call_request() {
  using simaai::neat::genai::ChatMessage;
  using simaai::neat::genai::GenerationRequest;
  using simaai::neat::genai::Json;

  GenerationRequest request;
  request.messages = {ChatMessage{
      "user",
      "Use the available tool to set coolant flow to 80 percent for machine CNC-01.",
  }};
  request.max_new_tokens = 128;
  request.tools = Json::array({Json{
      {"type", "function"},
      {"function",
       {{"name", "set_coolant_flow"},
        {"description", "Set the coolant flow percentage for a CNC machine."},
        {"parameters",
         {{"type", "object"},
          {"properties",
           {{"machine_id", {{"type", "string"}}}, {"flow_percentage", {{"type", "integer"}}}}},
          {"required", Json::array({"machine_id", "flow_percentage"})}}}}},
  }});
  return request;
}

fs::path resolve_model_dir() {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (env_model_dir.empty()) {
    skip_long_test_exception("set " + std::string(kModelEnv) +
                             " to an existing LLiMa text model directory");
  }

  fs::path model_dir(env_model_dir);
  if (has_llima_vlm_config(model_dir)) {
    return model_dir;
  }

  skip_long_test_exception(
      std::string(kModelEnv) +
      " does not point to a LLiMa text model directory: " + model_dir.string());
  return {};
}

} // namespace

int main() {
  try {
    const fs::path model_dir = resolve_model_dir();

    std::cout << "GENAI_LLM model_dir=" << model_dir << "\n";

    simaai::neat::genai::VisionLanguageModel model(model_dir);
    require(!model.accepts_image(), "Text-only LLiMa model should not accept image input");

    simaai::neat::genai::GenerationRequest request;
    request.system_prompt = std::string{"You are concise."};
    request.prompt = std::string{"What is the capital of Germany?"};
    request.max_new_tokens = 24;

    const auto result = model.run(request);
    require(!result.text.empty(), "GenAI LLM e2e expected non-empty generated text");
    const std::string normalized_text = trim_text(result.text);
    std::cout << "GENAI_LLM generated_tokens=" << result.metrics.generated_tokens
              << " ttft_s=" << result.metrics.time_to_first_token_s
              << " tps=" << result.metrics.tokens_per_second
              << " finish_reason=" << result.finish_reason << "\n";
    std::cout << "GENAI_LLM text=" << result.text << "\n";
    require(normalized_text == "The capital of Germany is Berlin.",
            "GenAI LLM e2e generated unexpected text: " + result.text);
    require(result.finish_reason == "stop" || result.finish_reason == "interrupted",
            "GenAI LLM e2e returned unexpected finish reason: " + result.finish_reason);

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
    std::cout << "GENAI_LLM_STREAM generated_tokens=" << final_sample.metrics.generated_tokens
              << " ttft_s=" << final_sample.metrics.time_to_first_token_s
              << " tps=" << final_sample.metrics.tokens_per_second
              << " finish_reason=" << final_sample.finish_reason << "\n";
    std::cout << "GENAI_LLM_STREAM text=" << streamed_text << "\n";
    require(trim_text(streamed_text) == "The capital of Germany is Berlin.",
            "GenAI LLM stream e2e generated unexpected text: " + streamed_text);

    simaai::neat::genai::GenAIModel generic_model(model_dir);
    require(generic_model.task() == simaai::neat::genai::GenAITask::VisionLanguage,
            "GenAIModel text task mismatch");
    require(generic_model.accepts_text(), "GenAIModel text model should accept text");
    require(!generic_model.accepts_image(), "GenAIModel text model should not accept images");
    require(!generic_model.accepts_audio(), "GenAIModel text model should not accept audio");

    const auto generic_result = generic_model.run(request);
    std::cout << "GENAI_MODEL_LLM text=" << generic_result.text << "\n";
    require(trim_text(generic_result.text) == "The capital of Germany is Berlin.",
            "GenAIModel LLM generated unexpected text: " + generic_result.text);

    auto generic_stream = generic_model.stream(request);
    std::string generic_streamed_text;
    bool saw_generic_final = false;
    while (auto sample = generic_stream.next()) {
      if (sample->is_final) {
        saw_generic_final = true;
        break;
      }
      generic_streamed_text += sample->text;
    }
    require(saw_generic_final, "GenAIModel LLM stream expected final sample");
    std::cout << "GENAI_MODEL_LLM_STREAM text=" << generic_streamed_text << "\n";
    require(trim_text(generic_streamed_text) == "The capital of Germany is Berlin.",
            "GenAIModel LLM stream generated unexpected text: " + generic_streamed_text);

    const auto tool_result = model.run(make_tool_call_request());
    const std::string tool_name = first_tool_call_name(tool_result.tool_calls);
    std::cout << "GENAI_LLM_TOOL_CALL finish_reason=" << tool_result.finish_reason << "\n";
    std::cout << "GENAI_LLM_TOOL_CALL raw_text=" << tool_result.text << "\n";
    std::cout << "GENAI_LLM_TOOL_CALL function=" << tool_name << "\n";

    simaai::neat::genai::GenerationRequest bad_audio_request;
    bad_audio_request.audio_file = fs::path{"audio.wav"};
    require_throws([&] { (void)generic_model.run(bad_audio_request); },
                   "GenAIModel VisionLanguage audio request");

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

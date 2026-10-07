#include <simaai/neat/pcie/genai/GenAIModel.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace g = simaai::neat::pcie::genai;
namespace pcie = simaai::neat::pcie;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Fn> void rejects(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected invalid request to be rejected");
}

std::string env(const char* name, const std::string& fallback = {}) {
  const auto* value = std::getenv(name);
  return value && *value ? value : fallback;
}

std::string normalized(const std::string& text) {
  std::string result;
  bool space = false;
  for (unsigned char ch : text) {
    if (ch >= 0x80 || std::isalnum(ch)) {
      if (space && !result.empty())
        result += ' ';
      result += ch >= 0x80 ? static_cast<char>(ch) : static_cast<char>(std::tolower(ch));
      space = false;
    } else {
      space = true;
    }
  }
  return result;
}

g::ConnectionOptions connection() {
  g::ConnectionOptions result;
  result.card_id = std::stoi(env("SIMAPCIE_CARD_ID", "0"));
  result.card_host = env("SIMAPCIE_CARD_HOST");
  result.user = env("SIMAPCIE_USER", "sima");
  result.ssh_key = env("SIMAPCIE_SSH_KEY");
  result.card_receive_directory =
      env("SIMAPCIE_GENAI_RECEIVE_ROOT", result.card_receive_directory.string());
  result.startup_timeout_ms = std::stoi(env("SIMAPCIE_READINESS_TIMEOUT_MS", "180000"));
  result.request_timeout_ms = std::stoi(env("SIMAPCIE_GENAI_REQUEST_TIMEOUT_MS", "120000"));
  return result;
}

fs::path model_path(const std::string& kind) {
  const auto root = env("SIMAPCIE_GENAI_MODELS_PATH", env("HOME") + "/workspace/models_genai");
  const auto name =
      kind == "llm"   ? env("SIMA_TEST_LLIMA_TEXT_MODEL", "Qwen2.5-0.5B-Instruct-Autoround-a16w4")
      : kind == "vlm" ? env("SIMA_TEST_LLIMA_VLM_MODEL", "LFM2.5-VL-450M-Autoround-a16w4")
                      : env("SIMA_TEST_LLIMA_ASR_MODEL", "whisper-small-a16w8-layered-encoder");
  const auto path = fs::path(root) / name;
  require(fs::is_regular_file(path / "devkit" /
                              (kind == "asr" ? "whisper_config.json" : "vlm_config.json")),
          "Missing prepared model: " + path.string() + "; run prepare_pcie_genai_models.sh");
  return path;
}

g::GenerationResult consume(g::GenerationStream& stream) {
  g::GenerationResult result;
  int finals = 0;
  while (auto sample = stream.next()) {
    require(finals == 0, "Output after terminal sample");
    result.text += sample->text;
    if (sample->is_final) {
      ++finals;
      result.finish_reason = sample->finish_reason;
      result.metrics = sample->metrics;
      result.language = sample->language;
      result.no_speech_prob = sample->no_speech_prob;
      result.avg_logprob = sample->avg_logprob;
    }
  }
  require(finals == 1, "Expected exactly one terminal sample");
  require(!stream.next(), "Stream did not remain exhausted");
  return result;
}

void generated(const g::GenerationResult& result) {
  std::cout << "text=" << result.text << " finish=" << result.finish_reason << std::endl;
  require(!normalized(result.text).empty(), "Expected generated text");
  require(result.metrics.generated_tokens > 0, "Expected generated tokens");
  require(result.finish_reason == "stop" || result.finish_reason == "interrupted",
          "Unexpected finish reason");
}

void llm(g::GenAIModel& model) {
  require(model.task() == g::GenAITask::VisionLanguage && model.accepts_text() &&
              !model.accepts_image() && !model.accepts_audio(),
          "LLM capabilities");
  g::GenerationRequest request;
  request.system_prompt = "You are concise.";
  request.prompt = "What is the capital of Germany?";
  request.max_new_tokens = 24;
  auto thinking = request;
  thinking.enable_thinking = true;
  rejects([&] { model.run(thinking); });
  // Remote request validation is observed when consuming the stream.
  rejects([&] {
    auto invalid = model.stream(thinking);
    consume(invalid);
  });
  const auto check = [](const g::GenerationResult& result) {
    generated(result);
    require(normalized(result.text) == "the capital of germany is berlin", "Unexpected LLM answer");
  };
  check(model.run(request));
  auto stream = model.stream(request);
  check(consume(stream));

  g::GenerationRequest history;
  history.messages = {{"system", "You are concise."},
                      {"user", "What is the capital of Germany?"},
                      {"assistant", "The capital of Germany is Berlin."},
                      {"user", "Repeat your previous answer exactly."}};
  history.max_new_tokens = 24;
  const auto history_result = model.run(history);
  generated(history_result);
  require(normalized(history_result.text).find("berlin") != std::string::npos,
          "History should retain the previous answer");

  g::GenerationRequest tools;
  tools.messages = {
      {"user", "Use the available tool to set coolant flow to 80 percent for machine CNC-01."}};
  tools.max_new_tokens = 128;
  tools.tools = g::Json::parse(R"([{"type":"function","function":{
    "name":"set_coolant_flow","description":"Set the coolant flow percentage for a CNC machine.",
    "parameters":{"type":"object","properties":{"machine_id":{"type":"string"},
    "flow_percentage":{"type":"integer"}},"required":["machine_id","flow_percentage"]}}}])");
  const auto tool_result = model.run(tools);
  // As in standalone Core, tool selection is model-dependent; inspect returned calls.
  std::cout << "tool_calls=" << tool_result.tool_calls << " text=" << tool_result.text << '\n';
  g::GenerationRequest bad_audio;
  bad_audio.audio_file = "audio.wav";
  rejects([&] { model.run(bad_audio); });

  g::GenerationRequest cancel_request;
  cancel_request.prompt = "Count from one to one hundred.";
  cancel_request.max_new_tokens = 256;
  for (int iteration = 0; iteration < 3; ++iteration) {
    auto active = model.stream(cancel_request);
    const auto first = active.next();
    require(first && !first->is_final, "Need an active stream to cancel");
    active.cancel();
    require(consume(active).finish_reason == "interrupted", "Cancellation finish reason");
    std::cout << "cancel " << iteration + 1 << ": terminal received\n";
  }
  // Reuse after cancellation; independent requests have no implicit history.
  check(model.run(request));
}

void vlm(g::GenAIModel& model, const fs::path& assets) {
  require(model.task() == g::GenAITask::VisionLanguage && model.accepts_text() &&
              model.accepts_image() && !model.accepts_audio(),
          "VLM capabilities");
  const auto bgr = cv::imread((assets / "people.jpg").string(), cv::IMREAD_COLOR);
  require(!bgr.empty(), "Cannot load VLM image fixture");
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  g::GenerationRequest request;
  request.prompt = "Describe this image in a short phrase.";
  request.max_new_tokens = 48;
  request.images = {pcie::Tensor::from_vector(
      std::vector<std::uint8_t>(rgb.data, rgb.data + rgb.total() * rgb.elemSize()),
      {rgb.rows, rgb.cols, 3}, "", pcie::PixelFormat::RGB)};
  generated(model.run(request));
  auto stream = model.stream(request);
  generated(consume(stream));
  generated(model.run(request));
}

void asr_metadata(const g::GenerationResult& result, const std::string& language) {
  require(result.finish_reason == "stop", "ASR finish reason");
  require(result.language == language, "ASR language mismatch");
  require(result.no_speech_prob && std::isfinite(*result.no_speech_prob) &&
              *result.no_speech_prob >= 0 && *result.no_speech_prob <= 1,
          "ASR no_speech_prob");
  require(result.avg_logprob && std::isfinite(*result.avg_logprob), "ASR avg_logprob");
}

void asr(g::GenAIModel& model, const fs::path& assets) {
  require(model.task() == g::GenAITask::ASR && !model.accepts_text() && !model.accepts_image() &&
              model.accepts_audio(),
          "ASR capabilities");
  const auto check = [](const g::GenerationResult& result, const std::string& expected,
                        const std::string& language = "en") {
    std::cout << "ASR text=" << result.text << std::endl;
    require(normalized(result.text) == expected, "ASR transcript mismatch");
    asr_metadata(result, language);
  };
  g::GenerationRequest file;
  file.audio_file = assets / "audio.wav";
  const auto spoken = model.run(file);
  check(spoken, "tell me a joke please");
  g::GenerationRequest german;
  german.audio_file = assets / "audio_de.wav";
  check(model.run(german), "erzähl mir bitte einen witz", "de");
  german.asr_task = g::ASRTask::Translate;
  check(model.run(german), "please tell me a joke", "de");

  std::ifstream input(assets / "audio_16k_mono_f32le.raw", std::ios::binary | std::ios::ate);
  require(input.good(), "Cannot open PCM fixture");
  const auto bytes = input.tellg();
  require(bytes > 0 && bytes % sizeof(float) == 0, "Invalid PCM fixture size");
  std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
  input.seekg(0);
  require(static_cast<bool>(input.read(reinterpret_cast<char*>(samples.data()), bytes)),
          "Cannot read PCM fixture");
  g::GenerationRequest pcm;
  const auto count = static_cast<std::int64_t>(samples.size());
  pcm.audio = pcie::Tensor::from_vector(std::move(samples), {count});
  pcm.sample_rate = 16000;
  pcm.language = "english";
  check(model.run(pcm), "tell me a joke please");

  auto stream = model.stream(file);
  check(consume(stream), "tell me a joke please");
  g::GenerationRequest silence;
  silence.audio = pcie::Tensor::from_vector(std::vector<float>(32000, 0), {32000});
  const auto silent = model.run(silence);
  require(silent.finish_reason == "stop", "Silence finish reason");
  require(silent.no_speech_prob && std::isfinite(*silent.no_speech_prob) &&
              *silent.no_speech_prob <= 1 && *silent.no_speech_prob > *spoken.no_speech_prob,
          "Silence should have higher no_speech_prob than voiced audio");
  require(silent.avg_logprob && std::isfinite(*silent.avg_logprob), "Silence avg_logprob");
  if (!normalized(silent.text).empty())
    require(*spoken.avg_logprob > *silent.avg_logprob,
            "Voiced audio should have higher avg_logprob");
  check(model.run(file), "tell me a joke please");
  g::GenerationRequest bad_text;
  bad_text.prompt = "What is this audio?";
  rejects([&] { model.run(bad_text); });
  g::GenerationRequest bad_image;
  bad_image.images = {pcie::Tensor::from_vector(std::vector<std::uint8_t>(3), {1, 1, 3}, "",
                                                pcie::PixelFormat::RGB)};
  rejects([&] { model.run(bad_image); });
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: test_genai_run llm|vlm|asr ASSETS_DIR");
    const std::string kind = argv[1];
    require(kind == "llm" || kind == "vlm" || kind == "asr", "Unknown GenAI test kind");
    std::cout << std::unitbuf << "PCIe GenAI " << kind << " model=" << model_path(kind) << '\n';
    g::GenAIModel model(model_path(kind).string(), connection());
    const fs::path assets = env("SIMAPCIE_GENAI_TEST_ASSETS", argv[2]);
    if (kind == "llm")
      llm(model);
    else if (kind == "vlm")
      vlm(model, assets);
    else
      asr(model, assets);
    model.close();
    model.close();
    std::cout << "PASS: " << kind << ", reuse and idempotent close\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

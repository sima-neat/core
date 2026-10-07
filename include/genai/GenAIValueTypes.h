#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <string>
namespace simaai::neat::genai {
using Json = nlohmann::ordered_json;
enum class GenAITask {
  VisionLanguage,
  ASR,
};

/// Whisper decoding task for ASR requests.
enum class ASRTask {
  Transcribe,
  Translate,
};

struct GenerationMetrics {
  std::uint32_t generated_tokens = 0;
  double time_to_first_token_s = 0.0;
  double tokens_per_second = 0.0;
};

struct GenerationResult {
  std::string text;
  GenerationMetrics metrics;
  std::string finish_reason;
  /// Detected or explicitly selected ASR source language.
  std::string language;
  /// Probability that the ASR input contains no speech.
  std::optional<float> no_speech_prob;
  /// Mean log probability over generated ASR tokens.
  std::optional<float> avg_logprob;
  Json tool_calls = Json::array();
  /// Model reasoning, when thinking was enabled and the model emitted it.
  std::string reasoning;
};

struct TokenSample {
  std::string text;
  GenerationMetrics metrics;
  bool is_final = false;
  std::string finish_reason;
  /// Detected or explicitly selected ASR source language on the final sample.
  std::string language;
  /// Probability that the ASR input contains no speech, set on the final sample.
  std::optional<float> no_speech_prob;
  /// Mean log probability over generated ASR tokens, set on the final sample.
  std::optional<float> avg_logprob;
  Json tool_calls = Json::array();
  /// Reasoning fragment; mutually exclusive with text for generated samples.
  std::string reasoning;
};

} // namespace simaai::neat::genai

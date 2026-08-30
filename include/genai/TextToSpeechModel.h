/**
 * @file
 * @brief Public NEAT handle for packaged Qwen3-TTS raw-ELF models.
 */
#pragma once

#include "pipeline/TensorAudio.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace simaai::neat::genai {

struct TextToSpeechRequest {
  std::string prompt;
  std::string speaker = "Vivian";
  std::string language = "English";
  std::uint32_t seed = 1;
  std::uint32_t max_frames = 512;
  bool do_sample = false;
  bool subtalker_do_sample = false;
  /// Optional persistent WAV destination. A temporary WAV is used otherwise.
  std::optional<std::filesystem::path> output_wav;
};

struct TextToSpeechResult {
  PcmAudio audio;
  /// Set only when TextToSpeechRequest::output_wav was provided.
  std::optional<std::filesystem::path> output_wav;
};

class TextToSpeechModel {
public:
  explicit TextToSpeechModel(std::filesystem::path model_dir);
  ~TextToSpeechModel();

  TextToSpeechModel(TextToSpeechModel&&) noexcept;
  TextToSpeechModel& operator=(TextToSpeechModel&&) noexcept;

  TextToSpeechModel(const TextToSpeechModel&) = delete;
  TextToSpeechModel& operator=(const TextToSpeechModel&) = delete;

  bool accepts_text() const;
  std::string model_id() const;
  TextToSpeechResult run(const TextToSpeechRequest& request);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class GenAIModel;
};

} // namespace simaai::neat::genai

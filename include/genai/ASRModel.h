/**
 * @file
 * @brief Public NEAT handle for LLiMa speech-to-text models.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <memory>
#include <string>

namespace simaai::neat::genai {

class ASRModel {
public:
  explicit ASRModel(std::filesystem::path model_dir);
  ~ASRModel();

  ASRModel(ASRModel&&) noexcept;
  ASRModel& operator=(ASRModel&&) noexcept;

  ASRModel(const ASRModel&) = delete;
  ASRModel& operator=(const ASRModel&) = delete;

  bool accepts_audio() const;
  std::string model_id() const;
  std::string describe() const;
  GenerationResult run(const GenerationRequest& request);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::genai

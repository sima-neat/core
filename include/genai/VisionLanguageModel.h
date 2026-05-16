/**
 * @file
 * @brief Public NEAT handle for LLiMa text and vision-language models.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <memory>
#include <string>

namespace simaai::neat::genai {

class VisionLanguageModel {
public:
  explicit VisionLanguageModel(std::filesystem::path model_dir);
  ~VisionLanguageModel();

  VisionLanguageModel(VisionLanguageModel&&) noexcept;
  VisionLanguageModel& operator=(VisionLanguageModel&&) noexcept;

  VisionLanguageModel(const VisionLanguageModel&) = delete;
  VisionLanguageModel& operator=(const VisionLanguageModel&) = delete;

  bool accepts_image() const;
  std::string model_id() const;
  std::string describe() const;
  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class GenerationStream;
};

} // namespace simaai::neat::genai

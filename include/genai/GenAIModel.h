/**
 * @file
 * @brief Auto-detecting GenAI model handle.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <memory>
#include <string>

namespace simaai::neat::genai {

class GenAIModel {
public:
  explicit GenAIModel(std::filesystem::path model_dir);
  ~GenAIModel();

  GenAIModel(GenAIModel&&) noexcept;
  GenAIModel& operator=(GenAIModel&&) noexcept;

  GenAIModel(const GenAIModel&) = delete;
  GenAIModel& operator=(const GenAIModel&) = delete;

  GenAITask task() const;
  bool accepts_text() const;
  bool accepts_image() const;
  bool accepts_audio() const;
  std::string model_id() const;
  std::string describe() const;
  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::genai
